"""
MXFP4 CUTLASS Kernel Tests: torch.compile integration and correctness validation.

Tests:
1. W4A4 torch.compile - verifies compiled output matches eager execution
2. W4A8 torch.compile - verifies compiled output matches eager execution
3. W4A8 correctness - validates CUTLASS output against triton hub dequant + torch.matmul reference
4. W4A8 correctness (compiled) - same as #3 but via torch.compile

Usage:
    PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/python3 test_mxfp4_matmul.py
"""

import torch
from torch.utils.cpp_extension import load
import os
import sys

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_cutlass_extension():
    """JIT compile and load the CUTLASS MXFP4 kernel extension."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    kernels_dir = os.path.join(project_root, "kernels")

    cutlass_include = os.path.join(project_root, "deps", "cutlass", "include")
    cutlass_tools_include = os.path.join(project_root, "deps", "cutlass", "tools", "util", "include")
    local_include = os.path.join(kernels_dir, "include")
    source_file = os.path.join(kernels_dir, "matmul_mxfp4_cutlass.cu")

    assert os.path.exists(cutlass_include), f"CUTLASS include not found at {cutlass_include}"
    assert os.path.exists(source_file), f"Kernel source not found at {source_file}"

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"

    ext = load(
        name="matmul_mxfp4_cutlass_compile_test",
        sources=[source_file],
        extra_include_paths=[local_include, cutlass_include, cutlass_tools_include],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "-DENABLE_NVFP4_SM120=1",
            "-DENABLE_CUTLASS_MOE_SM120=1",
            "-DCUTLASS_NVCC_ARCHS=120a",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return ext


def load_triton_reference():
    """Load triton hub quantization/dequantization functions as reference."""
    from kernels import get_kernel

    hub = get_kernel("kernels-community/triton_kernels")
    downcast = hub.numerics_details.mxfp.downcast_to_mxfp_torch
    upcast = hub.numerics_details.mxfp.upcast_from_mxfp_torch
    return hub, downcast, upcast


def register_custom_op(ext):
    """Register the CUTLASS kernel as a torch custom op for torch.compile."""

    @torch.library.custom_op("cutlass::fp4_group_mm", mutates_args=("output",))
    def fp4_group_mm(
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_blockscale: torch.Tensor,
        b_blockscales: torch.Tensor,
        alphas: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        sf_offsets: torch.Tensor,
    ) -> None:
        ext.cutlass_fp4_group_mm(
            output, a, b, a_blockscale, b_blockscales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )

    @fp4_group_mm.register_fake
    def _(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets):
        # In-place mutation of output; no new tensors created.
        pass

    return fp4_group_mm


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_w4a4_inputs(M, N, K, device):
    """Create random W4A4 inputs in the format expected by cutlass_fp4_group_mm."""
    group_size = 16
    k_groups = K // group_size

    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    a = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8)
    b = torch.randint(0, 255, (1, N, K // 2), device=device, dtype=torch.uint8)
    a_blockscale = torch.randn((M, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    b_blockscales = torch.randn((1, N, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    alphas = torch.ones((1,), device=device, dtype=torch.float32)
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    return output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets


def make_w4a8_inputs(M, N, K, device):
    """Create random W4A8 inputs in the format expected by cutlass_fp4_group_mm."""
    group_size = 32
    k_groups = K // group_size

    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    b = torch.randint(0, 255, (1, N, K // 2), device=device, dtype=torch.uint8)
    # UE8M0 scale range: avoid 0 (denorm) and extreme exponents to prevent overflow
    a_blockscale = torch.randint(110, 145, (M, k_groups), device=device, dtype=torch.uint8)
    b_blockscales = torch.randint(110, 145, (1, N, k_groups), device=device, dtype=torch.uint8)
    alphas = torch.ones((1,), device=device, dtype=torch.float32)
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    return output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets


def convert_scales_for_cutlass(ext, x_scale, w_scale, M, N, K):
    """Convert row-major scales to CUTLASS's blocked layout using C++ helpers.

    CUTLASS block-scaled MMA on SM120 expects scale factors in a specific
    tiled/blocked memory layout (defined by Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA/SFB).
    This is NOT simple row-major. The C++ helpers use the same ScaleConfig as the kernel
    to compute the correct layout mapping.
    """
    x_scale_flat = ext.convert_a_scales_for_w4a8(x_scale, M, N, K)
    w_scale_flat = ext.convert_b_scales_for_w4a8(w_scale, M, N, K)
    # Reshape back to expected tensor shapes for the kernel dispatcher
    x_scale_cutlass = x_scale_flat.view(x_scale.shape)
    w_scale_cutlass = w_scale_flat.view(w_scale.shape)
    return x_scale_cutlass, w_scale_cutlass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_w4a4_torch_compile(ext):
    """Test that W4A4 CUTLASS kernel works under torch.compile."""
    print("\n=== Test: W4A4 torch.compile ===")
    device = torch.device("cuda")
    M, N, K = 128, 256, 512
    inputs = make_w4a4_inputs(M, N, K, device)

    # Eager execution
    ext.cutlass_fp4_group_mm(*inputs)
    torch.cuda.synchronize()
    eager_output = inputs[0].clone()

    # Compiled execution via the registered custom op
    def fn(output, a, b, a_bs, b_bs, alphas, ps, eo, so):
        torch.ops.cutlass.fp4_group_mm(output, a, b, a_bs, b_bs, alphas, ps, eo, so)

    compiled_fn = torch.compile(fn, backend="inductor")
    output2 = torch.empty_like(eager_output)
    compiled_inputs = (output2,) + inputs[1:]
    compiled_fn(*compiled_inputs)
    torch.cuda.synchronize()

    diff = (output2 - eager_output).abs().max().item()
    print(f"  Compiled vs eager max diff: {diff}")
    assert diff == 0.0, f"W4A4 compiled output differs from eager by {diff}"
    print("  [PASS]")


def test_w4a8_torch_compile(ext):
    """Test that W4A8 CUTLASS kernel works under torch.compile."""
    print("\n=== Test: W4A8 torch.compile ===")
    device = torch.device("cuda")
    M, N, K = 128, 256, 512
    inputs = make_w4a8_inputs(M, N, K, device)

    # Eager execution
    ext.cutlass_fp4_group_mm(*inputs)
    torch.cuda.synchronize()
    eager_output = inputs[0].clone()

    # Compiled execution
    def fn(output, a, b, a_bs, b_bs, alphas, ps, eo, so):
        torch.ops.cutlass.fp4_group_mm(output, a, b, a_bs, b_bs, alphas, ps, eo, so)

    compiled_fn = torch.compile(fn, backend="inductor")
    output2 = torch.empty_like(eager_output)
    compiled_inputs = (output2,) + inputs[1:]
    compiled_fn(*compiled_inputs)
    torch.cuda.synchronize()

    diff = (output2 - eager_output).abs().max().item()
    print(f"  Compiled vs eager max diff: {diff}")
    assert diff == 0.0, f"W4A8 compiled output differs from eager by {diff}"
    print("  [PASS]")


def test_w4a8_correctness(ext, downcast, upcast):
    """
    Validate W4A8 CUTLASS kernel correctness against triton reference.

    Approach:
      1. Create bf16 activations X [M, K] and weights W [N, K].
      2. Quantize X to MXFP8 and W to MXFP4 using triton hub's downcast_to_mxfp_torch.
      3. Convert scale factors from row-major to CUTLASS's blocked layout.
      4. Reference: dequantize both back to bf16 using upcast_from_mxfp_torch,
         then compute torch.matmul(X_deq, W_deq^T) in float32.
      5. CUTLASS: feed quantized tensors + converted scales to cutlass_fp4_group_mm.
      6. Compare outputs.

    Scale format: both triton and CUTLASS use UE8M0 (uint8 biased exponent).
    Scale layout: CUTLASS expects a blocked layout, NOT row-major. We use
    convert_scales_for_cutlass() to handle the conversion.
    """
    print("\n=== Test: W4A8 Correctness vs Triton Reference ===")
    device = torch.device("cuda")

    for M, N, K in [(128, 256, 512), (256, 512, 1024), (128, 128, 256)]:
        torch.manual_seed(42)
        x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

        # Quantize using triton hub reference
        x_quant, x_scale = downcast(x_bf16, torch.float8_e4m3fn, axis=1)
        w_quant, w_scale = downcast(w_bf16, torch.uint8, axis=1)

        # Convert scales to CUTLASS blocked layout
        x_scale_c, w_scale_c = convert_scales_for_cutlass(ext, x_scale, w_scale, M, N, K)

        # Reference: dequantize + float32 matmul
        x_deq = upcast(x_quant, x_scale, torch.bfloat16, axis=1)
        w_deq = upcast(w_quant, w_scale, torch.bfloat16, axis=1)
        ref = (x_deq.float() @ w_deq.float().T).to(torch.bfloat16)

        # CUTLASS W4A8
        output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
        b = w_quant.unsqueeze(0).contiguous()
        b_scales = w_scale_c.unsqueeze(0).contiguous()
        alphas = torch.ones((1,), device=device, dtype=torch.float32)
        problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
        expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
        sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

        ext.cutlass_fp4_group_mm(
            output, x_quant, b, x_scale_c, b_scales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )
        torch.cuda.synchronize()

        # Compare
        abs_diff = (output.float() - ref.float()).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        ref_abs_mean = ref.float().abs().mean().item()
        rel_err = mean_diff / (ref_abs_mean + 1e-8)

        status = "PASS" if rel_err < 0.001 else ("WARN" if rel_err < 0.05 else "FAIL")
        print(f"  M={M} N={N} K={K}: rel_err={rel_err:.6f} max_diff={max_diff:.4f} [{status}]")
        assert status != "FAIL", f"W4A8 correctness failed for M={M} N={N} K={K} rel_err={rel_err}"

    print("  [PASS] All sizes correct")


def test_w4a8_correctness_compiled(ext, downcast, upcast):
    """Same as test_w4a8_correctness but runs the CUTLASS kernel via torch.compile."""
    print("\n=== Test: W4A8 Correctness (torch.compiled) vs Triton Reference ===")
    device = torch.device("cuda")
    M, N, K = 128, 256, 512

    torch.manual_seed(42)
    x_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    x_quant, x_scale = downcast(x_bf16, torch.float8_e4m3fn, axis=1)
    w_quant, w_scale = downcast(w_bf16, torch.uint8, axis=1)

    x_scale_c, w_scale_c = convert_scales_for_cutlass(ext, x_scale, w_scale, M, N, K)

    x_deq = upcast(x_quant, x_scale, torch.bfloat16, axis=1)
    w_deq = upcast(w_quant, w_scale, torch.bfloat16, axis=1)
    ref = (x_deq.float() @ w_deq.float().T).to(torch.bfloat16)

    # Run via torch.compile
    def fn(output, a, b, a_bs, b_bs, alphas, ps, eo, so):
        torch.ops.cutlass.fp4_group_mm(output, a, b, a_bs, b_bs, alphas, ps, eo, so)

    compiled_fn = torch.compile(fn, backend="inductor")

    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    b = w_quant.unsqueeze(0).contiguous()
    b_scales = w_scale_c.unsqueeze(0).contiguous()
    alphas = torch.ones((1,), device=device, dtype=torch.float32)
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    compiled_fn(output, x_quant, b, x_scale_c, b_scales, alphas, problem_sizes, expert_offsets, sf_offsets)
    torch.cuda.synchronize()

    abs_diff = (output.float() - ref.float()).abs()
    mean_diff = abs_diff.mean().item()
    ref_abs_mean = ref.float().abs().mean().item()
    rel_err = mean_diff / (ref_abs_mean + 1e-8)

    print(f"  Relative error: {rel_err:.6f} ({rel_err*100:.4f}%)")
    assert rel_err < 0.001, f"W4A8 compiled correctness failed: rel_err={rel_err}"
    print("  [PASS] Compiled CUTLASS output matches triton reference")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=" * 65)
    print(" MXFP4 CUTLASS Kernel: torch.compile & Correctness Tests")
    print("=" * 65)

    # 1. Load CUTLASS extension
    print("\n[1/4] Compiling CUTLASS extension (JIT)...")
    ext = load_cutlass_extension()
    print("  Done.")

    # 2. Register custom op for torch.compile
    print("\n[2/4] Registering torch.library custom op...")
    register_custom_op(ext)
    print("  Registered: torch.ops.cutlass.fp4_group_mm")

    # 3. Load triton reference
    print("\n[3/4] Loading triton/HF reference kernels...")
    hub, downcast, upcast = load_triton_reference()
    print("  Done.")

    # 4. Run tests
    print("\n[4/4] Running tests...")

    test_w4a4_torch_compile(ext)
    test_w4a8_torch_compile(ext)
    test_w4a8_correctness(ext, downcast, upcast)
    test_w4a8_correctness_compiled(ext, downcast, upcast)

    print("\n" + "=" * 65)
    print(" All tests completed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
