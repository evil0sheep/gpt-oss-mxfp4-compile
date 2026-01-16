import torch
import triton
import triton.language as tl
import sys
import os

# Add kernels to path
# This file is in kernels/triton_kernels/
# repo_root is ../../
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
kernels_path = os.path.join(repo_root, "kernels")
if kernels_path not in sys.path:
    sys.path.insert(0, kernels_path)

print(f"Added {kernels_path} to sys.path")

try:
    from triton_kernels.matmul_ogs_details._reduce_grouped import _reduce_grouped
    import triton_kernels.matmul_ogs_details._reduce_grouped as module_rg
    from triton_kernels.specialize import specialize
    print("Successfully imported _reduce_grouped")
except ImportError as e:
    print(f"Failed to import _reduce_grouped: {e}")
    sys.exit(1)

def test_compile():
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    device = torch.device("cuda")
    print("Testing compilation of _reduce_grouped...")

    # Dummy values matching the types in the kernel signature
    # X: *fp32
    X = torch.randn(1, 1, device=device, dtype=torch.float32)
    # Out: *bf16
    Out = torch.empty(1, 1, device=device, dtype=torch.bfloat16)

    # Grid size
    grid = (1,)

    # Arguments
    # def _reduce_grouped(X, stride_xb, stride_xm, stride_xn, XScale, Out, stride_om, stride_on, ...)

    # We pass None for Scales as they are constexpr in some configs or pointers in others.
    # Based on the previous log, XScale was constexpr=None.

    # Specialize the kernel
    constants = {
        "ACTIVATION_FN": None,
        "EPILOGUE_FN": None,
        "ACTIVATION_REDUCTION_N": 1,
        "HAS_IN_MX_SCALE": False,
        "HAS_OUT_MX_SCALE": False,
        "FLEXPOINT_SATURATE_INF": False,
        "K": 1,
        "BLOCK_N": 128
    }
    tuples = {
        "activation_fn_args": [],
        "epilogue_fn_args": []
    }

    specialized_kernel = specialize(_reduce_grouped, module_rg, constants, tuples)
    print("Kernel specialized successfully")


    def call_kernel(X, Out):
        specialized_kernel[grid](
            X, 1, 1, 1,         # X, strides
            None,               # XScale
            Out, 1, 1,          # Out, strides
            None, None, None,   # OutExpectedScale, OutActualScale, OutChecksumScale
            None,               # InIndx
            1, 1,               # B, N
            None, 0, 0,         # XMxScale, strides
            None, 0             # OutMxScale, strides
        )

    print("--- Testing Direct Execution ---")
    try:
        call_kernel(X, Out)
        print("Direct execution successful!")
    except Exception as e:
        print(f"Direct execution failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing torch.compile Execution ---")
    try:
        compiled_fn = torch.compile(call_kernel, backend="inductor", mode="reduce-overhead")
        # Warmup
        compiled_fn(X, Out)
        print("Compiled execution successful!")
    except Exception as e:
        print(f"Compiled execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compile()
