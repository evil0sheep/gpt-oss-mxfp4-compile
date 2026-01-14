import torch
from torch.utils.cpp_extension import load
import os
import sys

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
            verbose=True
        )
        print("[INFO] Extension compiled and loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Compilation failed: {e}")
        return

    # 3. Prepare Dummy Inputs
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, skipping execution test.")
        return

    print("\n--- Preparing Inputs ---")
    device = torch.device("cuda")

    # Dimensions for a single expert GEMM
    M, N, K = 16, 128, 128

    # Shapes based on kernel expectations (approximate for test plumbing)
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    # Input A (Activations) - Packed FP4 (uint8, 2 elements per byte)
    # Shape: [M, K/2]
    a = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8)

    # Input B (Weights) - Compressed NVFP4 (stored as uint8, 2 elements per byte) -> [E, N, K/2]
    # We use 1 expert.
    b = torch.randint(0, 255, (1, N, K // 2), device=device, dtype=torch.uint8)

    # Block scales (FP8)
    # Group size is 16. K=128 => K/16 = 8 blocks per row.
    group_size = 16
    k_groups = K // group_size

    # Shape: [M, k_groups]
    a_blockscale = torch.randn((M, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    # Shape: [E, N, k_groups] -> [1, N, k_groups]
    b_blockscales = torch.randn((1, N, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    # Alphas (one per expert)
    alphas = torch.ones((1,), device=device, dtype=torch.float32)

    # Grouped GEMM metadata
    # shape: [num_experts, 3] -> [[M, N, K]]
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)

    # Offsets for 1 expert
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    # 4. Invoke Kernel
    print("[INFO] Invoking cutlass_fp4_group_mm...")

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
        print("[SUCCESS] Kernel executed without error.")
    except RuntimeError as e:
        print(f"\n[ERROR] Caught RuntimeError during execution:")
        print(f"  {e}")

if __name__ == "__main__":
    main()
