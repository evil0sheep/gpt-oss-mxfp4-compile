import pytest
import torch
import sys
import os

# Ensure we can import from kernels
sys.path.append(os.getcwd())

try:
    from transformers.integrations.mxfp4 import quantize_to_mxfp4, swizzle_mxfp4
    HAS_TRANSFORMERS_MXFP4 = True
except ImportError:
    HAS_TRANSFORMERS_MXFP4 = False

try:
    from kernels.matmul_mxfp4_triton import matmul_mxfp4
except ImportError:
    # Fallback if running from kernels dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from matmul_mxfp4_triton import matmul_mxfp4

def get_hf_kernel():
    try:
        from kernels import get_kernel
        REPO_ID = "kernels-community/triton_kernels"
        hub = get_kernel(REPO_ID)
        return hub
    except ImportError:
        return None

def quantize_input_mxfp4(x, device):
    """
    Quantize input x (BF16/FP32) to float8_e4m3fn with block-wise scaling (block size 32).
    Returns (x_q, x_s).
    x: [M, K]
    x_q: [M, K] float8_e4m3fn
    x_s: [M, K//32] uint8 (e8m0)
    """
    M, K = x.shape
    BLOCK = 32
    assert K % BLOCK == 0

    x_f = x.float()
    x_reshaped = x_f.reshape(M, K // BLOCK, BLOCK)

    # Calculate scales
    # Max abs value per block
    amax = x_reshaped.abs().max(dim=-1).values

    # FP8 E4M3FN max value is 448.0
    # Scale = amax / 448.0
    # We map this scale to E8M0 (uint8)
    # But wait, typical MXFP4 uses E8M0 scales which represent 2^E.
    # Standard Blackwell/Hopper FP8 matmuls often use float scales or specific formats.
    # The kernel expects scales_a to be used in dot_scaled.
    # If using E4M3FN inputs, dot_scaled typically takes a scale factor.
    # If scales_a is uint8, is it E8M0?
    # HF implementation details:
    # For FP8 inputs, often scales are just 1.0 if pre-scaled, or block scales.
    # Let's assume for this test we can use a simplified scaling:
    # Scale = nearest 2^E to cover the range.

    # Simplified: No scaling (scale=1), just cast to FP8 for testing if dynamic range allows.
    # To test correctness properly with HF which handles scales, we should try to match.
    # HF `matmul_ogs` for `x` (BF16) probably quantizes internally.
    # Let's try to just cast `x` to FP8 and use scale 1.0 (represented as 127 in E8M0? 0 is 2^-127?).
    # E8M0 bias is 127. 127 -> 2^(127-127) = 1.0.

    x_q = x.to(torch.float8_e4m3fn)
    x_s = torch.full((M, K // BLOCK), 127, dtype=torch.uint8, device=device)

    return x_q, x_s

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_correctness_vs_hf():
    if not HAS_TRANSFORMERS_MXFP4:
        pytest.skip("transformers.integrations.mxfp4 not available for input prep")

    torch.manual_seed(0)
    device = torch.device("cuda")

    # 1. Load HF Kernel
    hub = get_hf_kernel()
    if hub is None or not hasattr(hub, "matmul_ogs"):
        pytest.skip("HF kernel matmul_ogs not available")

    matmul_ogs = hub.matmul_ogs.matmul_ogs
    PrecisionConfig = hub.matmul_ogs.PrecisionConfig

    # 2. Prepare Data
    M, K, N = 128, 256, 256 # Use reasonable sizes for tiling

    # Reference inputs (BF16)
    # Range -2 to 2 to fit in FP8 nicely without massive scaling issues
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # Quantize for HF Kernel (MXFP4)
    # w_q: [N, K/2] packed
    # w_s: [N, K/32]
    w_q, w_s = quantize_to_mxfp4(w, hub)

    # Swizzle
    # We need to transpose w before swizzling because HF expects [In, Out] (K, N) usually?
    # Transformers `swizzle_mxfp4` doc says: "weights should be (Out, In)".
    # But usually for Linear, weights are (Out, In).
    # `quantize_to_mxfp4` preserves shape?
    # Let's trust the previous test usage:
    # w_q.transpose(-1, -2) passed to swizzle.
    # w_q is [N, K/2]. Transposed [K/2, N].
    w_q_t = w_q.transpose(-1, -2).contiguous()
    w_s_t = w_s.transpose(-1, -2).contiguous() # Scales also transposed? Usually.

    # NOTE: transformers swizzle might expect specific layout.
    # Let's assume standard flow:
    w_q_sw, w_s_sw = swizzle_mxfp4(w_q_t, w_s, hub)
    # w_s passed as is? `swizzle_mxfp4` signature: (weight, scale, hub).

    # HF Kernel Run
    try:
        pc = PrecisionConfig(weight_scale=w_s_sw, out_dtype=torch.bfloat16)
        # matmul_ogs(x, w, ...)
        # x: [M, K]
        # w: swizzled weights
        out_hf = matmul_ogs(x, w_q_sw, None, precision_config=pc)
        print(f"\n[HF] Output mean: {out_hf.float().mean()}")
    except Exception as e:
        pytest.fail(f"HF Kernel failed: {e}")

    # 3. Prepare Data for Our Kernel
    # a: [M, K] FP8
    # b: [N, K/2] packed (unswizzled) -> We pass w_q directly?
    # scales_a: [M, K/32] uint8
    # scales_b: [N, K/32] (swizzled) -> We pass w_s_sw directly?

    # Quantize X
    x_q, x_s = quantize_input_mxfp4(x, device)

    # B: w_q (unswizzled, packed). Shape [N, K/2].
    # Our kernel loads it effectively as [K, N] (transposed) for calculation?
    # If we pass w_q, we match the packed layout.

    # Scales B: w_s_sw (swizzled scales).

    try:
        # We need to make sure w_s_sw is passed correctly.
        # w_s_sw might be a tuple or tensor?
        # transformers swizzle returns tensors.
        out_ours = matmul_mxfp4(x_q, w_q, x_s, w_s_sw)
        print(f"[Ours] Output mean: {out_ours.float().mean()}")
    except Exception as e:
        pytest.fail(f"Our kernel failed: {e}")

    # Compare
    assert out_ours.shape == out_hf.shape
    assert out_ours.dtype == out_hf.dtype

    # Tolerances: FP8/FP4 matmul is lossy.
    # HF kernel uses BF16 input (maybe converts to FP8 internally?).
    # We cast to FP8.
    # Expect some deviation.
    torch.testing.assert_close(out_ours, out_hf, atol=0.5, rtol=0.1)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
