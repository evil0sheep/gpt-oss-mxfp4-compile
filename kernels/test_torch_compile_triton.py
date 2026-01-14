import torch
import torch.nn as nn
import sys
import os

# Ensure we can import the kernel
try:
    from kernels.matmul_mxfp4_triton import matmul_mxfp4
except ImportError:
    # Fallback for running directly inside kernels/
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from matmul_mxfp4_triton import matmul_mxfp4
    except ImportError:
        print("Could not import matmul_mxfp4_triton. Make sure you are in the correct directory.")
        sys.exit(1)

class MXFP4Matmul(nn.Module):
    def __init__(self):
        super().__init__()
        # In a real layer, weights and scales would likely be registered parameters/buffers
        pass

    def forward(self, a, b, scales_a, scales_b):
        return matmul_mxfp4(a, b, scales_a, scales_b)

def test_torch_compile():
    print("--- Testing torch.compile with MXFP4 Triton Kernel ---")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    device = "cuda"
    M, N, K = 128, 128, 128

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

    model = MXFP4Matmul().to(device)

    print("Compiling model...")
    # We use mode="reduce-overhead" to see if it handles the triton call efficiently
    # But default mode is also fine for functional test.
    compiled_model = torch.compile(model)

    print("Running warmup...")
    try:
        # Warmup
        _ = compiled_model(a_fp8, b_packed, scales_a, scales_b)
        print("Warmup successful.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Running benchmark...")
    # Run a few times
    for i in range(5):
        output = compiled_model(a_fp8, b_packed, scales_a, scales_b)

    torch.cuda.synchronize()
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Verify output is a tensor and correct type
    if isinstance(output, torch.Tensor) and output.dtype == torch.bfloat16:
        print("SUCCESS: torch.compile execution produced expected tensor.")
    else:
        print("FAILURE: Output check failed.")

if __name__ == "__main__":
    test_torch_compile()
