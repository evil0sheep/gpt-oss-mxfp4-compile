"""
Benchmark the impact of GPU scale conversion and fused routing kernel
on end-to-end compiled model forward time.

Tests three configurations:
  A) FULLY OPTIMIZED: GPU scale conversion + fused routing (current code)
  B) CPU SCALES: CPU batched scale conversion + fused routing
  C) PYTHON ROUTING: GPU scale conversion + Python argsort routing

Measures compiled model forward latency for each.
"""

import gc
import json
import os
import sys
import time

import torch

# Add deps/transformers/src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_path = os.path.join(current_dir, "deps", "transformers", "src")
sys.path.insert(0, transformers_path)

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

# Load cutlass_experts module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cutlass_experts", os.path.join(current_dir, "kernels", "cutlass_experts.py")
)
cutlass_experts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutlass_experts_mod)
sys.modules["cutlass_experts"] = cutlass_experts_mod

CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights
get_cutlass_extension = cutlass_experts_mod.get_cutlass_extension
ensure_custom_op_registered = cutlass_experts_mod.ensure_custom_op_registered
quantize_activations_to_fp8 = cutlass_experts_mod.quantize_activations_to_fp8
pad_fp8_activations = cutlass_experts_mod.pad_fp8_activations
swiglu = cutlass_experts_mod.swiglu


def get_checkpoint_path():
    snap_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots"
    )
    snapshots = os.listdir(snap_dir)
    return os.path.join(snap_dir, snapshots[0])


def replace_experts_with_cutlass(model, checkpoint_path, config, expert_cls):
    """Replace all expert modules with the given expert class."""
    from safetensors import safe_open

    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)["weight_map"]

    safetensor_files = {}

    def get_tensor(name):
        fname = weight_index[name]
        if fname not in safetensor_files:
            safetensor_files[fname] = safe_open(
                os.path.join(checkpoint_path, fname), framework="pt"
            )
        return safetensor_files[fname].get_tensor(name)

    num_replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ not in ("GptOssExperts", "Mxfp4GptOssExperts"):
            continue

        cutlass_module = expert_cls(config).to("cuda")

        prefix = name
        gu_bias = get_tensor(f"{prefix}.gate_up_proj_bias").to("cuda")
        dp_bias = get_tensor(f"{prefix}.down_proj_bias").to("cuda")
        cutlass_module.gate_up_proj_bias.data.copy_(gu_bias)
        cutlass_module.down_proj_bias.data.copy_(dp_bias)

        gu_blocks = get_tensor(f"{prefix}.gate_up_proj_blocks")
        gu_scales = get_tensor(f"{prefix}.gate_up_proj_scales")
        dp_blocks = get_tensor(f"{prefix}.down_proj_blocks")
        dp_scales = get_tensor(f"{prefix}.down_proj_scales")

        load_cutlass_weights(
            cutlass_module, gu_blocks, gu_scales, dp_blocks, dp_scales, "cuda"
        )

        model.set_submodule(name, cutlass_module)
        num_replaced += 1

        del gu_blocks, gu_scales, dp_blocks, dp_scales, gu_bias, dp_bias
        torch.cuda.empty_cache()

    for f in safetensor_files.values():
        del f
    safetensor_files.clear()

    print(f"  Replaced {num_replaced} expert modules with {expert_cls.__name__}")
    return num_replaced


# ---------------------------------------------------------------------------
# Variant A: Fully optimized (current code) - no changes needed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Variant B: CPU batched scale conversion (instead of GPU)
# ---------------------------------------------------------------------------

class CutlassExpertsCpuScales(CutlassGptOssExperts):
    """Uses CPU batched scale conversion instead of GPU kernel."""

    def _run_cutlass_gemm(
        self, activations, act_scales_rowmajor, weights, weight_scales,
        expert_token_counts, expert_offsets, N, K, K_padded,
    ):
        ext = get_cutlass_extension()
        total_tokens = activations.shape[0]
        num_experts = expert_token_counts.shape[0]
        device = activations.device

        output = torch.zeros((total_tokens, N), device=device, dtype=torch.bfloat16)

        if K_padded > K:
            activations, act_scales_rowmajor, _ = pad_fp8_activations(
                activations, act_scales_rowmajor, K
            )

        # CPU batched scale conversion (the old path)
        act_scales_blocked, sf_offsets = ext.batch_convert_a_scales_for_w4a8(
            act_scales_rowmajor, expert_token_counts, expert_offsets, N, K_padded
        )

        problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)
        problem_sizes[:, 0] = expert_token_counts
        problem_sizes[:, 1] = N
        problem_sizes[:, 2] = K_padded

        alphas = torch.ones((num_experts,), device=device, dtype=torch.float32)

        torch.ops.cutlass.fp4_group_mm(
            output, activations, weights, act_scales_blocked, weight_scales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )

        return output


# ---------------------------------------------------------------------------
# Variant C: Python argsort routing (instead of fused CUDA routing)
# ---------------------------------------------------------------------------

class CutlassExpertsPythonRouting(CutlassGptOssExperts):
    """Uses Python argsort routing instead of fused CUDA routing kernel."""

    def forward(self, hidden_states, router_indices, routing_weights):
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        top_k = router_indices.shape[1]

        # --- Python argsort routing (the old path) ---
        flat_expert_idx = router_indices.reshape(-1)
        flat_weights = routing_weights.float().reshape(-1)
        flat_token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k).reshape(-1)

        sorted_order = torch.argsort(flat_expert_idx, stable=True)
        sorted_token_idx = flat_token_idx[sorted_order]
        sorted_expert_idx = flat_expert_idx[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        expert_token_counts = torch.zeros(self.num_experts, device=device, dtype=torch.int32)
        expert_token_counts.scatter_add_(
            0, sorted_expert_idx.to(torch.int64),
            torch.ones_like(sorted_expert_idx, dtype=torch.int32),
        )
        expert_offsets = torch.zeros(self.num_experts, device=device, dtype=torch.int32)
        if self.num_experts > 1:
            torch.cumsum(expert_token_counts[:-1], dim=0, out=expert_offsets[1:])

        # Gather tokens in expert-sorted order
        gathered_hidden = hidden_states[sorted_token_idx]

        # --- Step 2: Quantize activations BF16 -> FP8 ---
        act_fp8, act_scales = quantize_activations_to_fp8(gathered_hidden)

        # --- Step 3+4: CUTLASS grouped GEMM for gate_up_proj ---
        N_gate_up = 2 * self.intermediate_size
        K_hidden = self.hidden_size

        gate_up_output = self._run_cutlass_gemm(
            act_fp8, act_scales,
            self.gate_up_proj_data, self.gate_up_proj_scales,
            expert_token_counts, expert_offsets,
            N=N_gate_up, K=K_hidden, K_padded=self.K_hidden_padded,
        )

        # --- Step 5: Add bias + SwiGLU ---
        expert_bias_gate_up = self.gate_up_proj_bias[sorted_expert_idx]
        gate_up_output = gate_up_output + expert_bias_gate_up
        intermediate = swiglu(gate_up_output, self.alpha, self.limit)

        # --- Step 6: Quantize intermediate -> FP8 ---
        inter_fp8, inter_scales = quantize_activations_to_fp8(intermediate)

        # --- Step 7: CUTLASS grouped GEMM for down_proj ---
        N_hidden = self.hidden_size
        K_inter = self.intermediate_size

        down_output = self._run_cutlass_gemm(
            inter_fp8, inter_scales,
            self.down_proj_data, self.down_proj_scales,
            expert_token_counts, expert_offsets,
            N=N_hidden, K=K_inter, K_padded=self.K_inter_padded,
        )

        # --- Step 8: Add bias, scale by routing weights, scatter back ---
        expert_bias_down = self.down_proj_bias[sorted_expert_idx]
        down_output = down_output + expert_bias_down
        down_output = down_output * sorted_weights.unsqueeze(-1)

        result = torch.zeros((num_tokens, self.hidden_size), device=device, dtype=hidden_states.dtype)
        result.scatter_add_(
            0,
            sorted_token_idx.unsqueeze(-1).expand(-1, self.hidden_size),
            down_output.to(hidden_states.dtype),
        )

        return result.view(input_shape)


def benchmark_compiled_forward(model, label, num_warmup=3, num_runs=10, seq_len=128):
    """Benchmark compiled model forward pass."""
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device="cuda")
    attention_mask = torch.ones((1, seq_len), device="cuda", dtype=torch.long)

    # Compile
    print(f"  Compiling {label}...")
    compiled_model = torch.compile(model, backend="inductor")

    # Warmup (includes compilation)
    with torch.no_grad():
        for i in range(num_warmup):
            t0 = time.perf_counter()
            _ = compiled_model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i == 0:
                print(f"  Compilation + first run: {t1 - t0:.2f}s")

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compiled_model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    std_ms = (sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

    print(f"  {label}: avg={avg_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms  std={std_ms:.2f}ms  (n={num_runs})")

    # Clean up compiled model
    del compiled_model
    gc.collect()
    torch.cuda.empty_cache()

    return avg_ms, min_ms, max_ms


def benchmark_eager_forward(model, label, num_warmup=3, num_runs=10, seq_len=128):
    """Benchmark eager model forward pass (no compile)."""
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device="cuda")
    attention_mask = torch.ones((1, seq_len), device="cuda", dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    std_ms = (sum((t * 1000 - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

    print(f"  {label}: avg={avg_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms  std={std_ms:.2f}ms  (n={num_runs})")

    return avg_ms, min_ms, max_ms


def load_model_base():
    """Load GPT-OSS-20B base model (dequantized BF16, no experts yet)."""
    model_name = "openai/gpt-oss-20b"
    checkpoint_path = get_checkpoint_path()

    print(f"Loading model config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model with dequantize=True (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        quantization_config=Mxfp4Config(dequantize=True),
    )

    print("Compiling CUTLASS extension...")
    get_cutlass_extension()
    ensure_custom_op_registered()

    return model, config, checkpoint_path


def main():
    print("=" * 70)
    print(" Benchmark: GPU Scale Conversion + Fused Routing Impact")
    print("=" * 70)

    NUM_WARMUP = 3
    NUM_RUNS = 20
    SEQ_LEN = 128

    results = {}

    # -----------------------------------------------------------------------
    # Config A: Fully optimized (GPU scales + fused routing)
    # -----------------------------------------------------------------------
    print("\n--- Loading Config A: Fully Optimized ---")
    model, config, checkpoint_path = load_model_base()
    replace_experts_with_cutlass(model, checkpoint_path, config, CutlassGptOssExperts)

    print("\n--- Benchmarking Config A (eager) ---")
    results["A_eager"] = benchmark_eager_forward(
        model, "A: GPU scales + fused routing (eager)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    print("\n--- Benchmarking Config A (compiled) ---")
    results["A_compiled"] = benchmark_compiled_forward(
        model, "A: GPU scales + fused routing (compiled)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Config B: CPU batched scales + fused routing
    # -----------------------------------------------------------------------
    print("\n--- Loading Config B: CPU Scales ---")
    model, config, checkpoint_path = load_model_base()
    replace_experts_with_cutlass(model, checkpoint_path, config, CutlassExpertsCpuScales)

    print("\n--- Benchmarking Config B (eager) ---")
    results["B_eager"] = benchmark_eager_forward(
        model, "B: CPU scales + fused routing (eager)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    print("\n--- Benchmarking Config B (compiled) ---")
    results["B_compiled"] = benchmark_compiled_forward(
        model, "B: CPU scales + fused routing (compiled)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Config C: GPU scales + Python argsort routing
    # -----------------------------------------------------------------------
    print("\n--- Loading Config C: Python Routing ---")
    model, config, checkpoint_path = load_model_base()
    replace_experts_with_cutlass(model, checkpoint_path, config, CutlassExpertsPythonRouting)

    print("\n--- Benchmarking Config C (eager) ---")
    results["C_eager"] = benchmark_eager_forward(
        model, "C: GPU scales + Python routing (eager)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    print("\n--- Benchmarking Config C (compiled) ---")
    results["C_compiled"] = benchmark_compiled_forward(
        model, "C: GPU scales + Python routing (compiled)",
        num_warmup=NUM_WARMUP, num_runs=NUM_RUNS, seq_len=SEQ_LEN
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Configuration':<50} {'Eager (ms)':<15} {'Compiled (ms)':<15}")
    print("-" * 80)

    for label, key_prefix in [
        ("A: GPU scales + fused routing (optimized)", "A"),
        ("B: CPU scales + fused routing", "B"),
        ("C: GPU scales + Python routing", "C"),
    ]:
        eager_avg = results[f"{key_prefix}_eager"][0]
        compiled_avg = results[f"{key_prefix}_compiled"][0]
        print(f"{label:<50} {eager_avg:>10.2f}     {compiled_avg:>10.2f}")

    # Impact analysis
    print("\n--- Impact Analysis (compiled) ---")
    a_compiled = results["A_compiled"][0]
    b_compiled = results["B_compiled"][0]
    c_compiled = results["C_compiled"][0]

    gpu_scale_impact = b_compiled - a_compiled
    fused_routing_impact = c_compiled - a_compiled

    print(f"GPU scale conversion saves:   {gpu_scale_impact:+.2f}ms ({gpu_scale_impact/b_compiled*100:+.1f}% of B)")
    print(f"Fused routing kernel saves:   {fused_routing_impact:+.2f}ms ({fused_routing_impact/c_compiled*100:+.1f}% of C)")

    print("\n--- Impact Analysis (eager) ---")
    a_eager = results["A_eager"][0]
    b_eager = results["B_eager"][0]
    c_eager = results["C_eager"][0]

    gpu_scale_impact_e = b_eager - a_eager
    fused_routing_impact_e = c_eager - a_eager

    print(f"GPU scale conversion saves:   {gpu_scale_impact_e:+.2f}ms ({gpu_scale_impact_e/b_eager*100:+.1f}% of B)")
    print(f"Fused routing kernel saves:   {fused_routing_impact_e:+.2f}ms ({fused_routing_impact_e/c_eager*100:+.1f}% of C)")


if __name__ == "__main__":
    main()
