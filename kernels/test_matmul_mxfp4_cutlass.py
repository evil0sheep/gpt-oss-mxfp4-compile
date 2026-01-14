import torch
from torch.utils.cpp_extension import load
import os
import sys

def run_test_case(mxfp4_ext, mode, device):
    print(f"\n--- Running Test Case: {mode} ---")

    # Dimensions for a single expert GEMM
    if mode == "W4A8":
        M, N, K = 128, 256, 2048
    else:
        M, N, K = 128, 256, 512

    # Common Inputs
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    # Input B (Weights) - Always Compressed NVFP4 (stored as uint8, 2 elements per byte) -> [E, N, K/2]
    # We use 1 expert.
    b = torch.randint(0, 255, (1, N, K // 2), device=device, dtype=torch.uint8)

    # Block scales (FP8)
    if mode == "W4A8":
        group_size = 32
    else:
        group_size = 16
    k_groups = K // group_size

    # Shape: [M, k_groups]
    # a_blockscale initialized per mode

    # Shape: [E, N, k_groups] -> [1, N, k_groups]
    # b_blockscales initialized per mode

    # Alphas (one per expert)
    alphas = torch.ones((1,), device=device, dtype=torch.float32)

    # Grouped GEMM metadata
    # shape: [num_experts, 3] -> [[M, N, K]]
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    # Specific Inputs
    if mode == "W4A4":
        # Input A (Activations) - Packed FP4 (uint8, 2 elements per byte)
        # Shape: [M, K/2]
        a = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8)
        # Scales for FP4 are E4M3 (FP8)
        a_blockscale = torch.randn((M, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        b_blockscales = torch.randn((1, N, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    elif mode == "W4A8":
        # Input A (Activations) - FP8 (float8_e4m3fn, 1 element per byte)
        # Shape: [M, K]
        # Note: PyTorch random generation for float8 might need casting
        a = torch.randn((M, K), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
        # Scales for FP8 (E4M3) on SM120 MUST be UE8M0, represented as uint8
        a_blockscale = torch.randint(0, 255, (M, k_groups), device=device, dtype=torch.uint8)
        b_blockscales = torch.randint(0, 255, (1, N, k_groups), device=device, dtype=torch.uint8)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"[INFO] Invoking cutlass_fp4_group_mm ({mode})...")
    print(f"       A shape: {a.shape}, dtype: {a.dtype}")
    print(f"       B shape: {b.shape}, dtype: {b.dtype}")

    try:
        mxfp4_ext.cutlass_fp4_group_mm(
            output,
            a,
            b,
            a_blockscale,
            b_blockscales,
            alphas,
            problem_sizes,
            expert_offsets,
            sf_offsets
        )
        torch.cuda.synchronize()
        print(f"[SUCCESS] {mode} Kernel executed without error.")
    except RuntimeError as e:
        print(f"\n[ERROR] {mode} Caught RuntimeError during execution:")
        print(f"  {e}")

def main():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print("--- Testing Standalone CUTLASS MXFP4 Kernel ---")

    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    cutlass_include = os.path.join(project_root, "deps", "cutlass", "include")
    cutlass_tools_include = os.path.join(project_root, "deps", "cutlass", "tools", "util", "include")
    local_include = os.path.join(current_dir, "include")
    source_file = os.path.join(current_dir, "matmul_mxfp4_cutlass.cu")

    if not os.path.exists(cutlass_include):
        print(f"[ERROR] CUTLASS include not found at {cutlass_include}")
        return

    print(f"[INFO] Compiling extension...")
    print(f"       Source: {source_file}")
    print(f"       Includes: {cutlass_include}")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    # 2. JIT Compile the Extension
    try:
        mxfp4_ext = load(
            name="matmul_mxfp4_cutlass_ext",
            sources=[source_file],
            extra_include_paths=[local_include, cutlass_include, cutlass_tools_include],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-DENABLE_NVFP4_SM120=1",
                "-DENABLE_CUTLASS_MOE_SM120=1",
                "-DCUTLASS_NVCC_ARCHS=120a",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__"
            ],
            extra_ldflags=["-lcuda"],
            verbose=False # Set to True to debug compilation errors
        )
        print("[INFO] Extension compiled and loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Compilation failed: {e}")
        return

    # 3. Check CUDA
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, skipping execution test.")
        return

    device = torch.device("cuda")

    # 4. Run Tests
    run_test_case(mxfp4_ext, "W4A4", device)
    run_test_case(mxfp4_ext, "W4A8", device)

if __name__ == "__main__":
    main()
