import torch
from torch.utils.cpp_extension import load
import os
import time
import statistics

def benchmark_kernel(extension, inputs_tuple, num_iters=100, warmup=10):
    # Unpack only the tensor arguments required by the kernel
    # Signature: (output, a, b, a_scales, b_scales, alphas, problem_sizes, expert_offsets, sf_offsets)
    tensors = inputs_tuple[:9]

    # Warmup
    for _ in range(warmup):
        extension.cutlass_fp4_group_mm(*tensors)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        start = time.time()
        extension.cutlass_fp4_group_mm(*tensors)
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000) # Convert to ms

    avg_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    return avg_ms, std_ms

def main():
    print("--- Benchmarking CUTLASS MXFP4 Kernel (Compiler Flags Impact) ---")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        return

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

    # 2. Prepare Inputs (W4A4)
    device = torch.device("cuda")
    # Using a small problem size matching the test script to verify functionality
    M, N, K = 256, 4096, 4096

    print(f"[INFO] Problem Size: M={M}, N={N}, K={K}")

    # Output: BF16 [M, N]
    output = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    # Input A (Activations): Packed FP4 (uint8, 2 elems/byte) -> [M, K/2]
    a = torch.randint(0, 255, (M, K // 2), device=device, dtype=torch.uint8)

    # Input B (Weights): Packed FP4 (uint8, 2 elems/byte) -> [E=1, N, K/2]
    b = torch.randint(0, 255, (1, N, K // 2), device=device, dtype=torch.uint8)

    # Scales: FP8
    group_size = 16
    k_groups = K // group_size

    # Shape: [M, k_groups]
    a_scales = torch.randn((M, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    # Shape: [E=1, N, k_groups]
    b_scales = torch.randn((1, N, k_groups), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)

    # Alphas (one per expert)
    alphas = torch.ones((1,), device=device, dtype=torch.float32)

    # Metadata
    problem_sizes = torch.tensor([[M, N, K]], device=device, dtype=torch.int32)
    expert_offsets = torch.tensor([0], device=device, dtype=torch.int32)
    sf_offsets = torch.tensor([0], device=device, dtype=torch.int32)

    inputs = (output, a, b, a_scales, b_scales, alphas, problem_sizes, expert_offsets, sf_offsets, M, N, K)


    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    # 3. Define Configurations
    base_flags = [
        "-O3",
        "-std=c++17",
        "-DENABLE_NVFP4_SM120=1",
        "-DENABLE_CUTLASS_MOE_SM120=1",
        "-DCUTLASS_NVCC_ARCHS=120a",
        "--expt-relaxed-constexpr",
    ]

    # Flags to force enablement of half operators in CUDA headers
    extra_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__"
    ]

    configs = [
        ("Without_Half_Flags", base_flags),
        ("With_Half_Flags", base_flags + extra_flags)
    ]

    results = {}

    # 4. Run Benchmarks
    for name, flags in configs:
        print(f"\n[INFO] Compiling extension for config: '{name}'...")
        try:
            # We use a unique name to ensure fresh compilation
            ext = load(
                name=f"matmul_mxfp4_bench_{name}",
                sources=[source_file],
                extra_include_paths=[local_include, cutlass_include, cutlass_tools_include],
                extra_cuda_cflags=flags,
                verbose=False
            )
            print(f"[INFO] Benchmarking '{name}'...")
            avg, std = benchmark_kernel(ext, inputs)
            results[name] = (avg, std)
            print(f"       Avg: {avg:.4f} ms | Std: {std:.4f} ms")
        except Exception as e:
            print(f"[ERROR] Failed to compile or run '{name}': {e}")
            results[name] = None

    # 5. Report
    print("\n" + "="*40)
    print("       RESULTS SUMMARY")
    print("="*40)
    for name, res in results.items():
        if res:
            print(f"{name:<20}: {res[0]:.4f} ms")
        else:
            print(f"{name:<20}: FAILED")
    print("="*40)

if __name__ == "__main__":
    main()
