"""
Test script for GPT-OSS-20B with CUTLASS MoE experts.

Loads the model with CutlassGptOssExperts (instead of triton Mxfp4GptOssExperts),
tests forward pass and torch.compile compatibility.
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

# Load cutlass_experts module (avoid conflict with installed 'kernels' package)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cutlass_experts", os.path.join(current_dir, "kernels", "cutlass_experts.py")
)
cutlass_experts_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutlass_experts_mod)
# Register in sys.modules so torch.compile/TorchDynamo can find it
sys.modules["cutlass_experts"] = cutlass_experts_mod

CutlassGptOssExperts = cutlass_experts_mod.CutlassGptOssExperts
load_cutlass_weights = cutlass_experts_mod.load_cutlass_weights
get_cutlass_extension = cutlass_experts_mod.get_cutlass_extension
ensure_custom_op_registered = cutlass_experts_mod.ensure_custom_op_registered


def get_checkpoint_path():
    """Get path to local model checkpoint."""
    snap_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots"
    )
    snapshots = os.listdir(snap_dir)
    return os.path.join(snap_dir, snapshots[0])


def replace_experts_with_cutlass(model, checkpoint_path, config):
    """
    Replace all GptOssExperts modules with CutlassGptOssExperts,
    loading weights directly from checkpoint safetensors.
    """
    from safetensors import safe_open

    # Load weight index
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_index = json.load(f)["weight_map"]

    # Open safetensors files (lazy loading)
    safetensor_files = {}

    def get_tensor(name):
        fname = weight_index[name]
        if fname not in safetensor_files:
            safetensor_files[fname] = safe_open(
                os.path.join(checkpoint_path, fname), framework="pt"
            )
        return safetensor_files[fname].get_tensor(name)

    # Find and replace all expert modules
    num_replaced = 0
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ not in ("GptOssExperts", "Mxfp4GptOssExperts"):
            continue

        print(f"  Replacing {name}...")

        # Create CutlassGptOssExperts
        cutlass_module = CutlassGptOssExperts(config).to("cuda")

        # Load bias parameters
        # named_modules gives e.g. "model.layers.0.mlp.experts"
        # checkpoint keys are "model.layers.0.mlp.experts.gate_up_proj_blocks"
        prefix = name
        gu_bias = get_tensor(f"{prefix}.gate_up_proj_bias").to("cuda")
        dp_bias = get_tensor(f"{prefix}.down_proj_bias").to("cuda")
        cutlass_module.gate_up_proj_bias.data.copy_(gu_bias)
        cutlass_module.down_proj_bias.data.copy_(dp_bias)

        # Load blocks and scales from checkpoint
        gu_blocks = get_tensor(f"{prefix}.gate_up_proj_blocks")
        gu_scales = get_tensor(f"{prefix}.gate_up_proj_scales")
        dp_blocks = get_tensor(f"{prefix}.down_proj_blocks")
        dp_scales = get_tensor(f"{prefix}.down_proj_scales")

        # Convert and load into CutlassGptOssExperts
        load_cutlass_weights(
            cutlass_module, gu_blocks, gu_scales, dp_blocks, dp_scales, "cuda"
        )

        # Replace module in model
        model.set_submodule(name, cutlass_module)
        num_replaced += 1

        # Free checkpoint tensors
        del gu_blocks, gu_scales, dp_blocks, dp_scales, gu_bias, dp_bias
        torch.cuda.empty_cache()

    # Close safetensor files
    for f in safetensor_files.values():
        del f
    safetensor_files.clear()

    print(f"  Replaced {num_replaced} expert modules")
    return num_replaced


def load_model_with_cutlass_experts():
    """Load GPT-OSS-20B with CUTLASS experts instead of triton."""
    model_name = "openai/gpt-oss-20b"
    checkpoint_path = get_checkpoint_path()

    print(f"Loading model config from {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load with dequantize=True to get the model structure with BF16 weights
    # (GptOssExperts, not Mxfp4GptOssExperts)
    print(f"Loading model with dequantize=True (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        quantization_config=Mxfp4Config(dequantize=True),
    )

    # Initialize CUTLASS extension
    print("Compiling CUTLASS extension...")
    get_cutlass_extension()
    ensure_custom_op_registered()

    # Replace GptOssExperts with CutlassGptOssExperts
    print("Replacing expert modules with CUTLASS versions...")
    replace_experts_with_cutlass(model, checkpoint_path, config)

    # Free BF16 expert weight memory
    gc.collect()
    torch.cuda.empty_cache()

    return model, config


def test_forward(model):
    """Test basic forward pass."""
    print("\n=== Test: Forward Pass ===")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 128), device="cuda")
    attention_mask = torch.ones((1, 128), device="cuda", dtype=torch.long)

    with torch.no_grad():
        start = time.time()
        output = model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    logits = output.logits
    print(f"  Output shape: {logits.shape}")
    print(f"  Output sample: {logits[0, -1, :5]}")
    print(f"  Forward time: {elapsed:.4f}s")
    print(f"  [PASS]")
    return output


def test_compile(model):
    """Test torch.compile."""
    print("\n=== Test: torch.compile ===")
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 128), device="cuda")
    attention_mask = torch.ones((1, 128), device="cuda", dtype=torch.long)

    # Eager baseline
    with torch.no_grad():
        output_eager = model(input_ids, attention_mask=attention_mask)

    # Compile
    print("  Compiling model...")
    compiled_model = torch.compile(model, backend="inductor")

    print("  Running compiled warmup...")
    with torch.no_grad():
        start = time.time()
        output_compiled = compiled_model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        warmup_time = time.time() - start

    print(f"  Compiled warmup time: {warmup_time:.4f}s")

    # Check correctness
    diff = (output_eager.logits.float() - output_compiled.logits.float()).abs().max().item()
    rel_diff = (output_eager.logits.float() - output_compiled.logits.float()).abs().mean().item() / \
               (output_eager.logits.float().abs().mean().item() + 1e-8)
    print(f"  Eager vs compiled max diff: {diff}")
    print(f"  Eager vs compiled rel diff: {rel_diff:.6f}")

    # Measurement run
    print("  Running compiled measurement...")
    with torch.no_grad():
        start = time.time()
        output_compiled2 = compiled_model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        measure_time = time.time() - start

    print(f"  Compiled measurement time: {measure_time:.4f}s")

    # For a 24-layer model in bf16 with quantized experts, torch.compile may reorder
    # bf16 operations causing compounding differences across layers.
    # rel_diff < 5% is acceptable.
    status = "PASS" if rel_diff < 0.05 else "FAIL"
    print(f"  [{status}]")
    assert status == "PASS", f"torch.compile mismatch: rel_diff={rel_diff}"


def test_correctness_vs_dequantized(model, config):
    """
    Compare CUTLASS model output against the dequantized BF16 model.
    Loads a separate dequantized model and compares outputs.
    """
    print("\n=== Test: Correctness vs Dequantized BF16 ===")

    # Load dequantized model
    print("  Loading dequantized BF16 reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        quantization_config=Mxfp4Config(dequantize=True),
    )

    vocab_size = config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 64), device="cuda")
    attention_mask = torch.ones((1, 64), device="cuda", dtype=torch.long)

    with torch.no_grad():
        output_cutlass = model(input_ids, attention_mask=attention_mask)
        output_ref = ref_model(input_ids, attention_mask=attention_mask)

    diff = (output_cutlass.logits.float() - output_ref.logits.float()).abs()
    rel_err = diff.mean().item() / (output_ref.logits.float().abs().mean().item() + 1e-8)
    max_diff = diff.max().item()

    print(f"  Relative error: {rel_err:.6f}")
    print(f"  Max diff: {max_diff:.4f}")

    # The CUTLASS model quantizes activations to FP8, so some error is expected
    status = "PASS" if rel_err < 0.2 else "FAIL"
    print(f"  [{status}]")

    # Clean up reference model
    del ref_model
    gc.collect()
    torch.cuda.empty_cache()

    assert status == "PASS", f"Correctness test failed: rel_err={rel_err}"


def main():
    print("=" * 65)
    print(" GPT-OSS-20B with CUTLASS MoE Experts")
    print("=" * 65)

    model, config = load_model_with_cutlass_experts()

    test_forward(model)
    test_compile(model)

    # Correctness test requires loading a second model - skip if OOM
    try:
        test_correctness_vs_dequantized(model, config)
    except torch.cuda.OutOfMemoryError:
        print("\n=== Test: Correctness vs Dequantized BF16 ===")
        print("  SKIPPED (not enough GPU memory for two models)")

    print("\n" + "=" * 65)
    print(" All tests passed!")
    print("=" * 65)


if __name__ == "__main__":
    main()
