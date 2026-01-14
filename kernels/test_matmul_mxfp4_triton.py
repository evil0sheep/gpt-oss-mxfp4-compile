import torch
import sys
import os

try:
    from kernels.matmul_mxfp4_triton import matmul_mxfp4
except ImportError:
    # Fallback for running directly inside kernels/
    try:
        from matmul_mxfp4_triton import matmul_mxfp4
    except ImportError:
        print("Could not import matmul_mxfp4_triton. Make sure you are in the correct directory.")
        sys.exit(1)

def test_matmul():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("--- Testing MXFP4 Triton Kernel ---")

    M, N, K = 128, 128, 128
    device = "cuda"

    # Prepare inputs
    # A: FP8
    a_f32 = torch.randn((M, K), device=device, dtype=torch.float32)
    a_fp8 = a_f32.to(torch.float8_e4m3fn)

    # B: FP4 (Simulated as unpacked uint8 for fallback)
    b_packed = torch.randint(0, 255, (N, K), device=device, dtype=torch.uint8)

    # Scales: E8M0 (uint8)
    # Block size 32
    scales_a = torch.randint(0, 255, (M, K // 32), device=device, dtype=torch.uint8)
    scales_b = torch.randint(0, 255, (N, K // 32), device=device, dtype=torch.uint8)

    print(f"A shape: {a_fp8.shape}")
    print(f"B packed shape: {b_packed.shape}")

    try:
        c = matmul_mxfp4(a_fp8, b_packed, scales_a, scales_b)
        print("Kernel execution successful!")
        print(f"Output shape: {c.shape}")
        print(f"Output sample: {c[0, :4]}")

        # We don't verify correctness rigorously yet, just that it runs and produces BF16
        if c.dtype == torch.bfloat16:
            print("Output dtype is correctly bfloat16")

    except Exception as e:
        print(f"Kernel failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_matmul()
