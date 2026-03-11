"""
Unit tests for CUTLASS expert building blocks and CutlassGptOssExperts.

Tests:
  1. quantize_activations_to_fp8 - correctness vs triton reference, torch.compile
  2. swiglu - correctness vs GptOssExperts reference, torch.compile
  3. Single-expert CUTLASS matmul with quantization pipeline - correctness
  4. Multi-expert CUTLASS grouped GEMM - correctness
  5. CutlassGptOssExperts.forward - correctness vs dequantized reference

Usage:
    cd /home/compute/workspace/gpt-oss-test
    PATH="$(pwd)/.venv/bin:$PATH" python3 kernels/test_cutlass_experts.py
"""

import os
import sys
import importlib
import torch

# Direct import of cutlass_experts from the same directory (avoiding conflict with
# the installed `kernels` pip package)
_kernels_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_kernels_dir)
sys.path.insert(0, _project_root)

_spec = importlib.util.spec_from_file_location(
    "cutlass_experts", os.path.join(_kernels_dir, "cutlass_experts.py")
)
cutlass_experts_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cutlass_experts_mod)

# Module-level references for use in test functions
quantize_activations_to_fp8 = cutlass_experts_mod.quantize_activations_to_fp8
swiglu = cutlass_experts_mod.swiglu


def load_triton_reference():
    """Load triton hub quantization/dequantization functions."""
    from kernels import get_kernel
    hub = get_kernel("kernels-community/triton_kernels")
    downcast = hub.numerics_details.mxfp.downcast_to_mxfp_torch
    upcast = hub.numerics_details.mxfp.upcast_from_mxfp_torch
    return hub, downcast, upcast


# =====================================================================
# Test 1: quantize_activations_to_fp8
# =====================================================================

def test_quantize_activations_correctness(downcast, upcast):
    """Verify our quantize_activations_to_fp8 matches triton hub's downcast_to_mxfp_torch."""
    print("\n=== Test 1a: quantize_activations_to_fp8 correctness ===")
    device = torch.device("cuda")

    for M, K in [(128, 256), (64, 512), (256, 1024)]:
        torch.manual_seed(42)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

        # Our implementation
        our_fp8, our_scales = quantize_activations_to_fp8(x)

        # Triton reference
        ref_fp8, ref_scales = downcast(x, torch.float8_e4m3fn, axis=1)

        # Compare scales (UE8M0 uint8)
        scales_match = (our_scales == ref_scales).all().item()

        # Compare quantized values (compare as float since FP8 equality is tricky)
        our_deq = upcast(our_fp8, our_scales, torch.bfloat16, axis=1)
        ref_deq = upcast(ref_fp8, ref_scales, torch.bfloat16, axis=1)
        deq_diff = (our_deq.float() - ref_deq.float()).abs().max().item()

        status = "PASS" if scales_match and deq_diff == 0.0 else "FAIL"
        print(f"  M={M} K={K}: scales_match={scales_match}, deq_max_diff={deq_diff:.6f} [{status}]")
        assert status == "PASS", f"quantize_activations_to_fp8 failed for M={M} K={K}"

    print("  [PASS]")


def test_quantize_activations_compile():
    """Verify quantize_activations_to_fp8 works under torch.compile."""
    print("\n=== Test 1b: quantize_activations_to_fp8 torch.compile ===")
    device = torch.device("cuda")
    M, K = 128, 256

    torch.manual_seed(42)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    # Eager
    eager_fp8, eager_scales = quantize_activations_to_fp8(x)

    # Compiled
    compiled_fn = torch.compile(quantize_activations_to_fp8, backend="inductor")
    compiled_fp8, compiled_scales = compiled_fn(x)

    scales_match = (compiled_scales == eager_scales).all().item()
    fp8_match = (compiled_fp8.view(torch.uint8) == eager_fp8.view(torch.uint8)).all().item()

    print(f"  scales_match={scales_match}, fp8_match={fp8_match}")
    assert scales_match and fp8_match, "quantize_activations_to_fp8 compiled output differs"
    print("  [PASS]")


# =====================================================================
# Test 2: swiglu
# =====================================================================

def test_swiglu_correctness():
    """Verify our swiglu matches the GptOssExperts reference implementation."""
    print("\n=== Test 2a: swiglu correctness ===")
    device = torch.device("cuda")
    alpha = 1.702
    limit = 7.0

    torch.manual_seed(42)
    gate_up = torch.randn(64, 200, device=device, dtype=torch.bfloat16)  # 200 = 2*100

    # Our implementation
    our_result = swiglu(gate_up, alpha, limit)

    # Reference from GptOssExperts.forward
    gate = gate_up[..., ::2]
    up = gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    ref_glu = gate * torch.sigmoid(gate * alpha)
    ref_result = (up + 1) * ref_glu

    diff = (our_result.float() - ref_result.float()).abs().max().item()
    print(f"  max diff vs reference: {diff}")
    assert diff == 0.0, f"swiglu differs from reference by {diff}"
    print("  [PASS]")


def test_swiglu_compile():
    """Verify swiglu works under torch.compile."""
    print("\n=== Test 2b: swiglu torch.compile ===")
    device = torch.device("cuda")

    torch.manual_seed(42)
    gate_up = torch.randn(64, 200, device=device, dtype=torch.bfloat16)

    eager_result = swiglu(gate_up, 1.702, 7.0)
    compiled_fn = torch.compile(swiglu, backend="inductor")
    compiled_result = compiled_fn(gate_up, 1.702, 7.0)

    diff = (compiled_result.float() - eager_result.float()).abs().max().item()
    print(f"  compiled vs eager max diff: {diff}")
    # torch.compile may reorder bf16 ops, allowing small numerical differences
    assert diff < 0.1, f"swiglu compiled output differs by {diff}"
    print("  [PASS]")


# =====================================================================
# Test 3: Single-expert CUTLASS matmul with quantization pipeline
# =====================================================================

def test_single_expert_matmul(downcast, upcast):
    """
    Test the full quantization + CUTLASS matmul pipeline for a single expert.

    Pipeline: BF16 input -> quantize to FP8 -> CUTLASS W4A8 matmul -> BF16 output
    Reference: dequantize both -> float32 matmul
    """
    print("\n=== Test 3: Single-expert quantize + CUTLASS matmul ===")

    cutlass_experts_mod.ensure_custom_op_registered()
    ext = cutlass_experts_mod.get_cutlass_extension()
    device = torch.device("cuda")

    # Note: CUTLASS blocked layout may produce buffers that are NOT a multiple
    # of K_groups. Keep scales as flat 1D tensors and use raw element offsets.
    def convert_a_scales(scales, M, N, K):
        """Convert activation scales to flat blocked layout."""
        return ext.convert_a_scales_for_w4a8(scales, M, N, K)

    def convert_b_scales(scales, M, N, K):
        """Convert weight scales to flat blocked layout."""
        return ext.convert_b_scales_for_w4a8(scales, M, N, K)

    for M, N, K in [(128, 256, 512), (64, 128, 256), (256, 512, 1024)]:
        torch.manual_seed(42)

        # Create BF16 activations and weights
        x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        # Quantize activations to FP8 using our function
        x_fp8, x_scales = quantize_activations_to_fp8(x_bf16)

        # Quantize weights to FP4 using triton reference
        w_fp4, w_scales = downcast(w_bf16, torch.uint8, axis=1)

        # Convert scales to CUTLASS blocked layout (may be padded)
        x_scales_cutlass = convert_a_scales(x_scales, M, N, K)
        w_scales_cutlass = convert_b_scales(w_scales, M, N, K)

        # CUTLASS matmul
        output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
        b = w_fp4.unsqueeze(0).contiguous()
        b_scales = w_scales_cutlass.unsqueeze(0).contiguous()
        alphas = torch.ones((1,), device=device, dtype=torch.float32)
        problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
        expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
        sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

        torch.ops.cutlass.fp4_group_mm(
            output, x_fp8, b, x_scales_cutlass, b_scales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )
        torch.cuda.synchronize()

        # Reference: dequantize and matmul
        x_deq = upcast(x_fp8, x_scales, torch.bfloat16, axis=1)
        w_deq = upcast(w_fp4, w_scales, torch.bfloat16, axis=1)
        ref = (x_deq.float() @ w_deq.float().T).to(torch.bfloat16)

        abs_diff = (output.float() - ref.float()).abs()
        rel_err = abs_diff.mean().item() / (ref.float().abs().mean().item() + 1e-8)

        status = "PASS" if rel_err < 0.001 else "FAIL"
        print(f"  M={M} N={N} K={K}: rel_err={rel_err:.6f} [{status}]")
        assert status == "PASS", f"Single-expert matmul failed: rel_err={rel_err}"

    print("  [PASS]")


# =====================================================================
# Test 4: Multi-expert CUTLASS grouped GEMM
# =====================================================================

def test_multi_expert_matmul(downcast, upcast):
    """
    Test CUTLASS grouped GEMM with multiple experts.

    Key challenge: CUTLASS blocked scale layout depends on M (per-expert token count).
    Each expert's activation scales must be converted separately using its own (m_i, N, K).
    """
    print("\n=== Test 4: Multi-expert grouped GEMM ===")

    cutlass_experts_mod.ensure_custom_op_registered()
    ext = cutlass_experts_mod.get_cutlass_extension()
    device = torch.device("cuda")

    N, K = 256, 512
    k_groups = K // 32
    num_experts = 4
    tokens_per_expert = [64, 32, 128, 64]  # varying token counts
    total_tokens = sum(tokens_per_expert)

    torch.manual_seed(42)

    x_bf16 = torch.randn(total_tokens, K, device=device, dtype=torch.bfloat16)
    w_bf16_list = [torch.randn(N, K, device=device, dtype=torch.bfloat16) for _ in range(num_experts)]

    # Quantize activations
    x_fp8, x_scales = quantize_activations_to_fp8(x_bf16)

    # Convert activation scales per-expert to blocked layout
    # Use raw element offsets (not padded_m units) since blocked layout
    # size may not be a multiple of k_groups
    blocked_parts = []
    sf_offsets_list = []
    current_sf_offset = 0  # raw element offset
    act_offset = 0

    for i, m in enumerate(tokens_per_expert):
        expert_scales = x_scales[act_offset:act_offset + m].contiguous()
        blocked = ext.convert_a_scales_for_w4a8(expert_scales, m, N, K)
        blocked_parts.append(blocked)  # keep as flat 1D
        sf_offsets_list.append(current_sf_offset)
        current_sf_offset += blocked.numel()
        act_offset += m

    x_scales_blocked = torch.cat(blocked_parts, dim=0).to(device)
    sf_offsets = torch.tensor(sf_offsets_list, device=device, dtype=torch.int32)

    # Quantize and stack weights — keep scales as flat 1D per expert
    w_fp4_list = []
    w_scales_orig_list = []
    w_scales_cutlass_list = []
    for w in w_bf16_list:
        wq, ws = downcast(w, torch.uint8, axis=1)
        ws_cutlass = ext.convert_b_scales_for_w4a8(ws, total_tokens, N, K)
        w_fp4_list.append(wq)
        w_scales_orig_list.append(ws)
        w_scales_cutlass_list.append(ws_cutlass)

    b = torch.stack(w_fp4_list, dim=0).contiguous()
    b_scales = torch.stack(w_scales_cutlass_list, dim=0).contiguous()

    # Build expert metadata
    expert_token_counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)
    expert_offsets = torch.zeros(num_experts, device=device, dtype=torch.int32)
    torch.cumsum(expert_token_counts[:-1], dim=0, out=expert_offsets[1:])

    problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)
    problem_sizes[:, 0] = expert_token_counts
    problem_sizes[:, 1] = N
    problem_sizes[:, 2] = K

    alphas = torch.ones(num_experts, device=device, dtype=torch.float32)

    # CUTLASS grouped GEMM
    output = torch.zeros((total_tokens, N), device=device, dtype=torch.bfloat16)
    torch.ops.cutlass.fp4_group_mm(
        output, x_fp8, b, x_scales_blocked, b_scales,
        alphas, problem_sizes, expert_offsets, sf_offsets,
    )
    torch.cuda.synchronize()

    # Reference: per-expert dequant + matmul
    ref = torch.zeros((total_tokens, N), device=device, dtype=torch.bfloat16)
    offset = 0
    for i, m in enumerate(tokens_per_expert):
        x_deq = upcast(x_fp8[offset:offset + m], x_scales[offset:offset + m], torch.bfloat16, axis=1)
        w_deq = upcast(w_fp4_list[i], w_scales_orig_list[i], torch.bfloat16, axis=1)
        ref[offset:offset + m] = (x_deq.float() @ w_deq.float().T).to(torch.bfloat16)
        offset += m

    abs_diff = (output.float() - ref.float()).abs()
    rel_err = abs_diff.mean().item() / (ref.float().abs().mean().item() + 1e-8)
    max_diff = abs_diff.max().item()

    status = "PASS" if rel_err < 0.001 else "FAIL"
    print(f"  {num_experts} experts, tokens={tokens_per_expert}: rel_err={rel_err:.6f} max_diff={max_diff:.4f} [{status}]")
    assert status == "PASS", f"Multi-expert grouped GEMM failed: rel_err={rel_err}"
    print("  [PASS]")


# =====================================================================
# Test 5: CutlassGptOssExperts end-to-end
# =====================================================================

def test_cutlass_experts_forward(downcast, upcast):
    """
    Test CutlassGptOssExperts.forward end-to-end against a dequantized reference.

    Creates quantized weights, loads them into CutlassGptOssExperts, runs forward,
    and compares against dequantized matmul + swiglu in float32.
    """
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights

    print("\n=== Test 5: CutlassGptOssExperts end-to-end ===")
    device = torch.device("cuda")

    # Small config for testing
    class FakeConfig:
        num_local_experts = 4
        intermediate_size = 256
        hidden_size = 512
        swiglu_limit = 7.0

    config = FakeConfig()
    num_experts = config.num_local_experts
    hidden = config.hidden_size
    intermediate = config.intermediate_size

    torch.manual_seed(42)

    # Create BF16 weights for reference computation
    gate_up_bf16 = torch.randn(num_experts, hidden, 2 * intermediate, device=device, dtype=torch.bfloat16)
    down_bf16 = torch.randn(num_experts, intermediate, hidden, device=device, dtype=torch.bfloat16)
    gate_up_bias = torch.randn(num_experts, 2 * intermediate, device=device, dtype=torch.float32) * 0.01
    down_bias = torch.randn(num_experts, hidden, device=device, dtype=torch.float32) * 0.01

    # Quantize weights to FP4 (for CUTLASS) using triton reference
    # gate_up weights: [E, 2*intermediate, hidden] -> quantize along hidden (axis=1 after transpose)
    # CUTLASS expects B as [E, N, K/2] where N=2*intermediate, K=hidden
    # So we quantize each expert's weight matrix [2*intermediate, hidden] -> FP4 along K=hidden
    gate_up_blocks_list = []
    gate_up_scales_list = []
    for e in range(num_experts):
        # Weight is [hidden, 2*intermediate], we want to quantize [N, K] = [2*intermediate, hidden]
        w = gate_up_bf16[e].T  # [2*intermediate, hidden]
        wq, ws = downcast(w, torch.uint8, axis=1)  # FP4 packed, scales along K
        gate_up_blocks_list.append(wq)   # [2*intermediate, hidden//2]
        gate_up_scales_list.append(ws)   # [2*intermediate, hidden//32]

    down_blocks_list = []
    down_scales_list = []
    for e in range(num_experts):
        w = down_bf16[e].T  # [hidden, intermediate]
        wq, ws = downcast(w, torch.uint8, axis=1)
        down_blocks_list.append(wq)   # [hidden, intermediate//2]
        down_scales_list.append(ws)   # [hidden, intermediate//32]

    # Stack into [E, N, K//2] and [E, N, K//32]
    gate_up_blocks = torch.stack(gate_up_blocks_list, dim=0)  # [E, 2*inter, hidden//2]
    gate_up_scales = torch.stack(gate_up_scales_list, dim=0)  # [E, 2*inter, hidden//32]
    down_blocks = torch.stack(down_blocks_list, dim=0)        # [E, hidden, inter//2]
    down_scales = torch.stack(down_scales_list, dim=0)        # [E, hidden, inter//32]

    # Create CutlassGptOssExperts module and load weights
    module = CutlassGptOssExperts(config).to(device)
    module.gate_up_proj_bias.data.copy_(gate_up_bias)
    module.down_proj_bias.data.copy_(down_bias)

    # Reshape blocks to checkpoint format [E, N, K//32, 16] for load_cutlass_weights
    gate_up_blocks_ckpt = gate_up_blocks.reshape(num_experts, 2 * intermediate, hidden // 32, 16)
    down_blocks_ckpt = down_blocks.reshape(num_experts, hidden, intermediate // 32, 16)

    load_cutlass_weights(module, gate_up_blocks_ckpt, gate_up_scales, down_blocks_ckpt, down_scales, device)

    # Create test inputs
    num_tokens = 32
    top_k = 2
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1

    # Router selects top_k experts per token
    torch.manual_seed(123)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    # --- Run CutlassGptOssExperts ---
    cutlass_output = module(hidden_states, router_indices, routing_weights)

    # --- Compute reference using dequantized weights ---
    # Dequantize weights
    gate_up_deq = torch.zeros(num_experts, hidden, 2 * intermediate, device=device, dtype=torch.bfloat16)
    for e in range(num_experts):
        w_deq = upcast(gate_up_blocks_list[e], gate_up_scales_list[e], torch.bfloat16, axis=1)
        gate_up_deq[e] = w_deq.T  # [2*intermediate, hidden] -> [hidden, 2*intermediate]

    down_deq = torch.zeros(num_experts, intermediate, hidden, device=device, dtype=torch.bfloat16)
    for e in range(num_experts):
        w_deq = upcast(down_blocks_list[e], down_scales_list[e], torch.bfloat16, axis=1)
        down_deq[e] = w_deq.T  # [hidden, intermediate] -> [intermediate, hidden]

    # Reference forward (same logic as GptOssExperts but per-token)
    alpha = 1.702
    limit = 7.0
    ref_output = torch.zeros(num_tokens, hidden, device=device, dtype=torch.bfloat16)

    for t in range(num_tokens):
        x = hidden_states[t]  # [hidden]
        for k in range(top_k):
            e = router_indices[t, k].item()
            w = routing_weights[t, k].item()

            # gate_up = x @ gate_up_proj[e] + bias
            gate_up = x @ gate_up_deq[e] + gate_up_bias[e]

            # SwiGLU
            gate = gate_up[::2].clamp(max=limit)
            up = gate_up[1::2].clamp(min=-limit, max=limit)
            inter = (gate * torch.sigmoid(gate * alpha) * (up + 1)).to(torch.bfloat16)

            # down = inter @ down_proj[e] + bias
            down = inter @ down_deq[e] + down_bias[e]

            ref_output[t] += (w * down).to(torch.bfloat16)

    # Compare
    abs_diff = (cutlass_output.float() - ref_output.float()).abs()
    rel_err = abs_diff.mean().item() / (ref_output.float().abs().mean().item() + 1e-8)
    max_diff = abs_diff.max().item()

    # Higher tolerance expected: CUTLASS uses W4A8 (quantizes activations to FP8)
    # while reference uses dequantized weights in BF16. The quantization error
    # from BF16->FP8 activations adds noise.
    status = "PASS" if rel_err < 0.15 else "FAIL"
    print(f"  rel_err={rel_err:.4f} max_diff={max_diff:.4f} [{status}]")
    if status == "FAIL":
        print(f"  cutlass output sample: {cutlass_output[0, :5]}")
        print(f"  reference sample:      {ref_output[0, :5]}")
    assert status == "PASS", f"CutlassGptOssExperts forward failed: rel_err={rel_err}"
    print("  [PASS]")


def test_cutlass_experts_compile(downcast, upcast):
    """Test 6: CutlassGptOssExperts under torch.compile."""
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights
    print("\n=== Test 6: CutlassGptOssExperts torch.compile ===")
    device = "cuda"
    num_experts = 4
    hidden = 512
    intermediate = 256
    top_k = 2
    num_tokens = 64

    # Create config
    config = type("Config", (), {
        "num_local_experts": num_experts,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
    })()

    module = CutlassGptOssExperts(config).to(device)

    # Generate and load weights (same as Test 5)
    gate_up_blocks_list = []
    gate_up_scales_list = []
    down_blocks_list = []
    down_scales_list = []

    for e in range(num_experts):
        w_gu = torch.randn(2 * intermediate, hidden, device=device, dtype=torch.bfloat16) * 0.02
        packed_gu, scales_gu = downcast(w_gu, torch.uint8, axis=1)
        gate_up_blocks_list.append(packed_gu)
        gate_up_scales_list.append(scales_gu)

        w_d = torch.randn(hidden, intermediate, device=device, dtype=torch.bfloat16) * 0.02
        packed_d, scales_d = downcast(w_d, torch.uint8, axis=1)
        down_blocks_list.append(packed_d)
        down_scales_list.append(scales_d)

    gate_up_blocks = torch.stack(gate_up_blocks_list)
    gate_up_scales = torch.stack(gate_up_scales_list)
    down_blocks = torch.stack(down_blocks_list)
    down_scales = torch.stack(down_scales_list)

    load_cutlass_weights(module, gate_up_blocks, gate_up_scales, down_blocks, down_scales, device)

    # Eager forward
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    eager_output = module(hidden_states, router_indices, routing_weights)

    # Compiled forward
    compiled_module = torch.compile(module, fullgraph=False)
    compiled_output = compiled_module(hidden_states, router_indices, routing_weights)

    diff = (eager_output.float() - compiled_output.float()).abs().max().item()
    status = "PASS" if diff == 0.0 else "PASS (minor diff)" if diff < 0.01 else "FAIL"
    print(f"  eager vs compiled max diff: {diff}")
    if "FAIL" in status:
        print(f"  eager sample: {eager_output[0, :5]}")
        print(f"  compiled sample: {compiled_output[0, :5]}")
    assert "FAIL" not in status, f"torch.compile mismatch: diff={diff}"
    print(f"  [{status}]")


# =====================================================================
# Test 7: Edge cases (0-token experts, all-to-one, single token)
# =====================================================================

def test_edge_cases(downcast, upcast):
    """Test CutlassGptOssExperts with edge-case routing patterns."""
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights

    print("\n=== Test 7: Edge cases ===")
    device = torch.device("cuda")

    class FakeConfig:
        num_local_experts = 4
        intermediate_size = 256
        hidden_size = 512
        swiglu_limit = 7.0

    config = FakeConfig()
    num_experts = config.num_local_experts
    hidden = config.hidden_size
    intermediate = config.intermediate_size

    # Build module with weights
    torch.manual_seed(42)
    module = CutlassGptOssExperts(config).to(device)
    module.gate_up_proj_bias.data.normal_(0, 0.01)
    module.down_proj_bias.data.normal_(0, 0.01)

    gate_up_blocks_list, gate_up_scales_list = [], []
    down_blocks_list, down_scales_list = [], []
    for e in range(num_experts):
        w_gu = torch.randn(2 * intermediate, hidden, device=device, dtype=torch.bfloat16) * 0.02
        packed_gu, scales_gu = downcast(w_gu, torch.uint8, axis=1)
        gate_up_blocks_list.append(packed_gu)
        gate_up_scales_list.append(scales_gu)
        w_d = torch.randn(hidden, intermediate, device=device, dtype=torch.bfloat16) * 0.02
        packed_d, scales_d = downcast(w_d, torch.uint8, axis=1)
        down_blocks_list.append(packed_d)
        down_scales_list.append(scales_d)

    load_cutlass_weights(
        module,
        torch.stack(gate_up_blocks_list),
        torch.stack(gate_up_scales_list),
        torch.stack(down_blocks_list),
        torch.stack(down_scales_list),
        device,
    )

    # Case 7a: Single token
    print("  7a: Single token...")
    hidden_states = torch.randn(1, hidden, device=device, dtype=torch.bfloat16) * 0.1
    router_indices = torch.tensor([[0, 2]], device=device, dtype=torch.int64)
    routing_weights = torch.tensor([[0.6, 0.4]], device=device, dtype=torch.float32)
    out = module(hidden_states, router_indices, routing_weights)
    assert out.shape == (1, hidden), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print("    [PASS]")

    # Case 7b: All tokens to one expert
    print("  7b: All tokens to expert 0...")
    num_tokens = 32
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1
    router_indices = torch.zeros(num_tokens, 2, device=device, dtype=torch.int64)
    router_indices[:, 1] = 1  # all go to experts 0 and 1
    routing_weights = torch.full((num_tokens, 2), 0.5, device=device, dtype=torch.float32)
    out = module(hidden_states, router_indices, routing_weights)
    assert out.shape == (num_tokens, hidden)
    assert not torch.isnan(out).any(), "NaN in output"
    print("    [PASS]")

    # Case 7c: Large batch (stress test routing)
    print("  7c: Large batch (256 tokens)...")
    num_tokens = 256
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1
    torch.manual_seed(99)
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, 2, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)
    out = module(hidden_states, router_indices, routing_weights)
    assert out.shape == (num_tokens, hidden)
    assert not torch.isnan(out).any(), "NaN in output"
    print("    [PASS]")

    print("  [PASS]")


# =====================================================================
# Test 8: 3D input (batch, seq, hidden)
# =====================================================================

def test_3d_input(downcast, upcast):
    """Test CutlassGptOssExperts with 3D [batch, seq, hidden] input."""
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights

    print("\n=== Test 8: 3D input shape ===")
    device = torch.device("cuda")

    class FakeConfig:
        num_local_experts = 4
        intermediate_size = 256
        hidden_size = 512
        swiglu_limit = 7.0

    config = FakeConfig()

    torch.manual_seed(42)
    module = CutlassGptOssExperts(config).to(device)
    module.gate_up_proj_bias.data.normal_(0, 0.01)
    module.down_proj_bias.data.normal_(0, 0.01)

    gate_up_blocks_list, gate_up_scales_list = [], []
    down_blocks_list, down_scales_list = [], []
    for e in range(config.num_local_experts):
        w_gu = torch.randn(2 * config.intermediate_size, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        packed_gu, scales_gu = downcast(w_gu, torch.uint8, axis=1)
        gate_up_blocks_list.append(packed_gu)
        gate_up_scales_list.append(scales_gu)
        w_d = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=torch.bfloat16) * 0.02
        packed_d, scales_d = downcast(w_d, torch.uint8, axis=1)
        down_blocks_list.append(packed_d)
        down_scales_list.append(scales_d)

    load_cutlass_weights(
        module,
        torch.stack(gate_up_blocks_list),
        torch.stack(gate_up_scales_list),
        torch.stack(down_blocks_list),
        torch.stack(down_scales_list),
        device,
    )

    batch, seq = 2, 16
    num_tokens = batch * seq

    # 3D input
    hidden_states_3d = torch.randn(batch, seq, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    # 2D equivalent
    hidden_states_2d = hidden_states_3d.reshape(num_tokens, config.hidden_size)

    torch.manual_seed(123)
    router_logits = torch.randn(num_tokens, config.num_local_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, 2, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    out_3d = module(hidden_states_3d, router_indices, routing_weights)
    out_2d = module(hidden_states_2d, router_indices, routing_weights)

    assert out_3d.shape == (batch, seq, config.hidden_size), f"3D output shape wrong: {out_3d.shape}"
    assert out_2d.shape == (num_tokens, config.hidden_size), f"2D output shape wrong: {out_2d.shape}"

    diff = (out_3d.reshape(num_tokens, -1).float() - out_2d.float()).abs().max().item()
    print(f"  3D vs 2D max diff: {diff}")
    assert diff == 0.0, f"3D vs 2D output differs by {diff}"
    print("  [PASS]")


# =====================================================================
# Test 9: K-padding (K not a multiple of 128)
# =====================================================================

def test_k_padding(downcast, upcast):
    """Test CutlassGptOssExperts with K dimensions that require padding."""
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights
    pad_k_to_tile = cutlass_experts_mod.pad_k_to_tile

    print("\n=== Test 9: K-padding (K not multiple of 128) ===")
    device = torch.device("cuda")

    # K=288 -> K_padded=384 (next multiple of 128)
    # This is analogous to GPT-OSS's K=2880 -> K_padded=2944
    hidden = 288
    intermediate = 288
    K_hidden_padded = pad_k_to_tile(hidden)
    K_inter_padded = pad_k_to_tile(intermediate)
    print(f"  hidden={hidden} -> padded={K_hidden_padded}")
    print(f"  intermediate={intermediate} -> padded={K_inter_padded}")
    assert K_hidden_padded > hidden, "Expected padding needed"

    class FakeConfig:
        num_local_experts = 4
        intermediate_size = 288
        hidden_size = 288
        swiglu_limit = 7.0

    config = FakeConfig()

    torch.manual_seed(42)
    module = CutlassGptOssExperts(config).to(device)
    module.gate_up_proj_bias.data.normal_(0, 0.01)
    module.down_proj_bias.data.normal_(0, 0.01)

    gate_up_blocks_list, gate_up_scales_list = [], []
    down_blocks_list, down_scales_list = [], []
    for e in range(config.num_local_experts):
        w_gu = torch.randn(2 * intermediate, hidden, device=device, dtype=torch.bfloat16) * 0.02
        packed_gu, scales_gu = downcast(w_gu, torch.uint8, axis=1)
        gate_up_blocks_list.append(packed_gu)
        gate_up_scales_list.append(scales_gu)
        w_d = torch.randn(hidden, intermediate, device=device, dtype=torch.bfloat16) * 0.02
        packed_d, scales_d = downcast(w_d, torch.uint8, axis=1)
        down_blocks_list.append(packed_d)
        down_scales_list.append(scales_d)

    load_cutlass_weights(
        module,
        torch.stack(gate_up_blocks_list),
        torch.stack(gate_up_scales_list),
        torch.stack(down_blocks_list),
        torch.stack(down_scales_list),
        device,
    )

    # Verify padding was applied
    assert module.K_hidden_padded == K_hidden_padded
    assert module.K_inter_padded == K_inter_padded
    assert module.gate_up_proj_data.shape[2] == K_hidden_padded // 2

    # Run forward
    num_tokens = 32
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16) * 0.1
    torch.manual_seed(123)
    router_logits = torch.randn(num_tokens, config.num_local_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, 2, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    out = module(hidden_states, router_indices, routing_weights)
    assert out.shape == (num_tokens, hidden), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"

    # Run under torch.compile to verify padding is compile-safe
    compiled_module = torch.compile(module, fullgraph=False)
    compiled_out = compiled_module(hidden_states, router_indices, routing_weights)
    diff = (out.float() - compiled_out.float()).abs().max().item()
    print(f"  Eager vs compiled max diff: {diff}")
    assert diff < 0.01, f"torch.compile mismatch with K-padding: {diff}"

    print("  [PASS]")


# =====================================================================
# Test 10: Batched scale conversion consistency
# =====================================================================

def test_batched_scale_conversion():
    """Verify batch_convert_a_scales matches per-expert convert_a_scales."""
    print("\n=== Test 10: Batched vs per-expert scale conversion ===")

    cutlass_experts_mod.ensure_custom_op_registered()
    ext = cutlass_experts_mod.get_cutlass_extension()
    device = torch.device("cuda")

    N, K = 256, 512
    k_groups = K // 32
    num_experts = 4
    tokens_per_expert = [64, 0, 128, 32]  # includes 0-token expert
    total_tokens = sum(tokens_per_expert)

    torch.manual_seed(42)
    scales = torch.randint(0, 256, (total_tokens, k_groups), device=device, dtype=torch.uint8)

    expert_counts = torch.tensor(tokens_per_expert, device=device, dtype=torch.int32)
    expert_offsets = torch.zeros(num_experts, device=device, dtype=torch.int32)
    torch.cumsum(expert_counts[:-1], dim=0, out=expert_offsets[1:])

    # Batched conversion
    batch_blocked, batch_sf = ext.batch_convert_a_scales_for_w4a8(
        scales, expert_counts, expert_offsets, N, K
    )

    # Per-expert conversion (reference)
    ref_parts = []
    ref_sf = []
    current_offset = 0
    act_offset = 0
    for i, m in enumerate(tokens_per_expert):
        ref_sf.append(current_offset)
        if m == 0:
            continue
        expert_scales = scales[act_offset:act_offset + m].contiguous()
        blocked = ext.convert_a_scales_for_w4a8(expert_scales, m, N, K)
        ref_parts.append(blocked)
        current_offset += blocked.numel()
        act_offset += m

    ref_blocked = torch.cat(ref_parts, dim=0).to(device) if ref_parts else torch.zeros(0, device=device, dtype=torch.uint8)
    ref_sf_tensor = torch.tensor(ref_sf, device=device, dtype=torch.int32)

    # Compare
    assert batch_blocked.shape == ref_blocked.shape, \
        f"Shape mismatch: {batch_blocked.shape} vs {ref_blocked.shape}"
    assert (batch_blocked == ref_blocked).all(), "Blocked scales differ"
    assert (batch_sf == ref_sf_tensor).all(), "SF offsets differ"

    print(f"  {num_experts} experts, tokens={tokens_per_expert}: exact match")
    print("  [PASS]")


# =====================================================================
# Profile: Per-step timing of forward pass
# =====================================================================

def profile_forward(downcast, upcast):
    """Profile CutlassGptOssExperts forward pass per-step timing."""
    import time as _time
    CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
    load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights

    print("\n=== Profile: Forward pass timing breakdown ===")
    device = torch.device("cuda")

    class FakeConfig:
        num_local_experts = 32
        intermediate_size = 512
        hidden_size = 512
        swiglu_limit = 7.0

    config = FakeConfig()

    torch.manual_seed(42)
    module = CutlassGptOssExperts(config).to(device)
    module.gate_up_proj_bias.data.normal_(0, 0.01)
    module.down_proj_bias.data.normal_(0, 0.01)

    gate_up_blocks_list, gate_up_scales_list = [], []
    down_blocks_list, down_scales_list = [], []
    for e in range(config.num_local_experts):
        w_gu = torch.randn(2 * config.intermediate_size, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        packed_gu, scales_gu = downcast(w_gu, torch.uint8, axis=1)
        gate_up_blocks_list.append(packed_gu)
        gate_up_scales_list.append(scales_gu)
        w_d = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=torch.bfloat16) * 0.02
        packed_d, scales_d = downcast(w_d, torch.uint8, axis=1)
        down_blocks_list.append(packed_d)
        down_scales_list.append(scales_d)

    load_cutlass_weights(
        module,
        torch.stack(gate_up_blocks_list),
        torch.stack(gate_up_scales_list),
        torch.stack(down_blocks_list),
        torch.stack(down_scales_list),
        device,
    )

    num_tokens = 128
    top_k = 4
    hidden_states = torch.randn(num_tokens, config.hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    torch.manual_seed(123)
    router_logits = torch.randn(num_tokens, config.num_local_experts, device=device)
    routing_weights, router_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = torch.softmax(routing_weights, dim=-1)

    # Warmup
    with torch.no_grad():
        _ = module(hidden_states, router_indices, routing_weights)
    torch.cuda.synchronize()

    # Timed runs
    num_runs = 10
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = _time.perf_counter()
        with torch.no_grad():
            _ = module(hidden_states, router_indices, routing_weights)
        torch.cuda.synchronize()
        times.append(_time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    print(f"  Config: {config.num_local_experts} experts, {num_tokens} tokens, top_k={top_k}")
    print(f"  Forward pass: avg={avg_ms:.2f}ms  min={min_ms:.2f}ms  max={max_ms:.2f}ms")
    print(f"  Throughput: {num_tokens / (avg_ms / 1000):.0f} tokens/sec")


# =====================================================================
# Main
# =====================================================================

def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=" * 65)
    print(" CUTLASS Expert Building Blocks: Unit Tests")
    print("=" * 65)

    # Load triton reference
    print("\nLoading triton reference...")
    hub, downcast, upcast = load_triton_reference()
    print("  Done.")

    # Load CUTLASS extension (triggers JIT compilation)
    print("\nCompiling CUTLASS extension...")
    get_cutlass_extension = cutlass_experts_mod.get_cutlass_extension
    ensure_custom_op_registered = cutlass_experts_mod.ensure_custom_op_registered
    ext = get_cutlass_extension()
    ensure_custom_op_registered()
    print("  Done.")

    # Run tests
    test_quantize_activations_correctness(downcast, upcast)
    test_quantize_activations_compile()
    test_swiglu_correctness()
    test_swiglu_compile()
    test_single_expert_matmul(downcast, upcast)
    test_multi_expert_matmul(downcast, upcast)
    test_cutlass_experts_forward(downcast, upcast)
    test_cutlass_experts_compile(downcast, upcast)
    test_edge_cases(downcast, upcast)
    test_3d_input(downcast, upcast)
    test_k_padding(downcast, upcast)
    test_batched_scale_conversion()
    profile_forward(downcast, upcast)

    print("\n" + "=" * 65)
    print(" All tests passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
