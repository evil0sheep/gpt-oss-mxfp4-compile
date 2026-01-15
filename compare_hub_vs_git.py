import torch
import os
import sys

def main():
    print("--- Comparing Hub vs Local Git Kernels ---")

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        return
    device = torch.device("cuda")

    # 1. Setup Input Data
    M, K, N = 1, 4096, 4096
    print(f"[INFO] Problem Size: M={M}, N={N}, K={K}")
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # Reference BF16
    ref_out = torch.matmul(x, w.t())
    print(f"[INFO] Reference Output (BF16) - Mean: {ref_out.abs().mean():.4f}, Max: {ref_out.abs().max():.4f}")

    # 2. Run Hub Implementation
    print("\n[INFO] --- Testing Hub Implementation ---")
    try:
        from kernels import get_kernel
        from transformers.integrations.mxfp4 import quantize_to_mxfp4, swizzle_mxfp4

        hub_repo = "kernels-community/triton_kernels"
        hub_mod = get_kernel(hub_repo)
        print(f"[INFO] Hub loaded from: {hub_mod.__file__}")

        # Quantize using Hub
        w_q_hub, w_s_hub = quantize_to_mxfp4(w, hub_mod)
        w_q_sw_hub, w_s_sw_hub = swizzle_mxfp4(w_q_hub.transpose(-1, -2), w_s_hub, hub_mod)

        # Run Hub Matmul
        matmul_ogs_hub = hub_mod.matmul_ogs.matmul_ogs
        PrecisionConfig_hub = hub_mod.matmul_ogs.PrecisionConfig

        pc_hub = PrecisionConfig_hub(weight_scale=w_s_sw_hub, out_dtype=torch.bfloat16)
        out_hub = matmul_ogs_hub(x, w_q_sw_hub, None, precision_config=pc_hub)
        print("[SUCCESS] Hub execution finished.")

    except Exception as e:
        print(f"[ERROR] Hub execution failed: {e}")
        import traceback
        traceback.print_exc()
        out_hub = None

    # 3. Run Local Git Implementation
    print("\n[INFO] --- Testing Local Git Implementation ---")
    try:
        # Setup path
        repo_root = os.path.dirname(os.path.abspath(__file__))
        local_kernels_path = os.path.join(repo_root, "deps", "triton", "python", "triton_kernels")

        if os.path.exists(local_kernels_path):
            if local_kernels_path not in sys.path:
                sys.path.insert(0, local_kernels_path)
                print(f"[INFO] Added {local_kernels_path} to sys.path")
        else:
            raise FileNotFoundError(f"Local kernels path not found: {local_kernels_path}")

        import triton_kernels
        # Explicit import of submodules just to be safe
        import triton_kernels.matmul_ogs

        print(f"[INFO] Local kernels imported from: {triton_kernels.__file__}")

        # Quantize using Local
        w_q_git, w_s_git = quantize_to_mxfp4(w, triton_kernels)
        w_q_sw_git, w_s_sw_git = swizzle_mxfp4(w_q_git.transpose(-1, -2), w_s_git, triton_kernels)

        # Run Local Matmul
        matmul_ogs_git = triton_kernels.matmul_ogs.matmul_ogs
        PrecisionConfig_git = triton_kernels.matmul_ogs.PrecisionConfig

        pc_git = PrecisionConfig_git(weight_scale=w_s_sw_git, out_dtype=torch.bfloat16)
        out_git = matmul_ogs_git(x, w_q_sw_git, None, precision_config=pc_git)
        print("[SUCCESS] Local Git execution finished.")

    except Exception as e:
        print(f"[ERROR] Local Git execution failed: {e}")
        import traceback
        traceback.print_exc()
        out_git = None

    # 4. Comparison
    print("\n[INFO] --- Comparison Results ---")

    if out_hub is not None and out_git is not None:
        diff = (out_hub - out_git).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Hub vs Git Max Diff: {max_diff:.6f}")
        print(f"Hub vs Git Mean Diff: {mean_diff:.6f}")

        if max_diff == 0:
            print(">>> MATCH: Local Git implementation matches Hub implementation exactly.")
        else:
            print(">>> MISMATCH: Implementations differ.")

    if out_hub is not None:
        diff_ref = (out_hub - ref_out).abs()
        print(f"Hub vs Ref Max Diff: {diff_ref.max().item():.6f}")

    if out_git is not None:
        diff_ref = (out_git - ref_out).abs()
        print(f"Git vs Ref Max Diff: {diff_ref.max().item():.6f}")

if __name__ == "__main__":
    main()
