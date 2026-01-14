import pytest
import torch
import sys
import os

# Add deps/transformers/src to path if needed (assuming structure)
# or just rely on installed transformers
try:
    from transformers.integrations.mxfp4 import quantize_to_mxfp4, swizzle_mxfp4
    HAS_TRANSFORMERS_MXFP4 = True
except ImportError:
    HAS_TRANSFORMERS_MXFP4 = False

# Import our custom kernel
try:
    from kernels.matmul_mxfp4_triton import matmul_mxfp4
except ImportError:
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
    M, K, N = 1, 128, 128

    # Reference inputs (BF16)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # Quantize for HF Kernel (MXFP4)
    # quantize_to_mxfp4 returns (w_q, w_s)
    w_q, w_s = quantize_to_mxfp4(w, hub)

    # Swizzle
    # transformers does transpose(-1, -2) before swizzling for Linear layers weights W [Out, In] -> [In, Out]
    w_q_sw, w_s_sw = swizzle_mxfp4(w_q.transpose(-1, -2), w_s, hub)

    # HF Kernel Run
    try:
        pc = PrecisionConfig(weight_scale=w_s_sw, out_dtype=torch.bfloat16)
        out_hf = matmul_ogs(x, w_q_sw, None, precision_config=pc)
        print(f"\n[HF] Output mean: {out_hf.float().mean()}")
    except Exception as e:
        pytest.fail(f"HF Kernel failed: {e}")

    # 3. Prepare Data for Our Kernel
    # Our kernel matmul_mxfp4(a, b, scales_a, scales_b)
    # currently implements a fallback that expects:
    # A: FP8
    # B: uint8 (unpacked N, K)

    a_fp8 = x.to(torch.float8_e4m3fn)
    # Create fake B and scales for our kernel to ensure it runs
    # (Since our kernel is currently a placeholder fallback, strict input translation isn't implemented)
    b_ours = torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8)
    scales_a = torch.ones((M, K // 32), device=device, dtype=torch.uint8)
    scales_b = torch.ones((N, K // 32), device=device, dtype=torch.uint8)

    try:
        out_ours = matmul_mxfp4(a_fp8, b_ours, scales_a, scales_b)
        print(f"[Ours] Output mean: {out_ours.float().mean()}")
    except Exception as e:
        pytest.fail(f"Our kernel failed: {e}")

    # Compare (Expect failure for now on value, but check shape/type)
    assert out_ours.shape == out_hf.shape
    assert out_ours.dtype == out_hf.dtype
    torch.testing.assert_close(out_ours, out_hf, atol=1e-2, rtol=1e-2)
