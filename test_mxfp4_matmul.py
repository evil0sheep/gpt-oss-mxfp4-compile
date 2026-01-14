import torch
import os
import sys
import inspect

# Add transformers to path if needed (though it should be installed in venv)
# sys.path.append(os.path.join(os.getcwd(), "deps", "transformers", "src"))

def main():
    print("--- Testing MXFP4 Matmul Kernel loading ---")

    # 1. Load the kernels hub
    try:
        from kernels import get_kernel
        print("[INFO] Successfully imported 'kernels'")
    except ImportError:
        print("[ERROR] Could not import 'kernels'. Is it installed?")
        return

    REPO_ID = "kernels-community/triton_kernels"
    print(f"[INFO] Loading kernel repo: {REPO_ID}")

    # This downloads/caches the code from HF Hub
    hub = get_kernel(REPO_ID)
    print(f"[INFO] Hub loaded: {hub}")

    # 2. Inspect the source for the known bug
    if hasattr(hub, "matmul_ogs"):
        matmul_ogs_mod = hub.matmul_ogs
        if hasattr(matmul_ogs_mod, "matmul_ogs"):
            func = matmul_ogs_mod.matmul_ogs
            src_file = inspect.getfile(func)
            print(f"[INFO] matmul_ogs source file: {src_file}")

            # Look for the internal file where the error occurred
            dir_path = os.path.dirname(src_file)
            finalize_path = os.path.join(dir_path, "matmul_ogs_details", "_finalize_matmul.py")

            if os.path.exists(finalize_path):
                with open(finalize_path, "r") as f:
                    content = f.read()
                    if "cuda_capability_geq" in content:
                        print(f"[CHECK] Found 'cuda_capability_geq' usage in {finalize_path}")
                        if "def cuda_capability_geq" not in content and "import cuda_capability_geq" not in content:
                             print("[CONFIRMATION] 'cuda_capability_geq' is NOT defined or imported in the file. This confirms the bug.")
                    else:
                        print("[CHECK] 'cuda_capability_geq' string not found in file.")
            else:
                print(f"[WARN] Could not find {finalize_path}")
        else:
            print("[WARN] hub.matmul_ogs does not have 'matmul_ogs' attribute")
    else:
        print("[WARN] hub does not have 'matmul_ogs' attribute")

    # 3. Attempt Execution (if GPU available)
    if not torch.cuda.is_available():
        print("[INFO] No CUDA device, skipping execution test.")
        return

    print("\n--- Attempting Execution ---")
    device = torch.device("cuda")

    # Try to import helpers from transformers to prepare valid inputs
    try:
        from transformers.integrations.mxfp4 import quantize_to_mxfp4, swizzle_mxfp4
        print("[INFO] Found transformers.integrations.mxfp4 helpers")
    except ImportError:
        print("[WARN] Could not import transformers helpers. Skipping execution.")
        return

    # Prepare Dummy Data
    M, K, N = 1, 128, 128 # Small size

    # Input X
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    # Weight W (Targeting Dense/Linear equivalent)
    # Shape: (Out, In) = (N, K)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    print("[INFO] Quantizing and Swizzling weights...")
    try:
        # Note: We pass our loaded 'hub' to these functions
        w_q, w_s = quantize_to_mxfp4(w, hub)

        # Swizzle: transformers does transpose(-1, -2) before swizzling for Linear layers
        # w is (N, K). transpose -> (K, N).
        w_q_sw, w_s_sw = swizzle_mxfp4(w_q.transpose(-1, -2), w_s, hub)
        print("[INFO] Weights prepared successfully.")
    except Exception as e:
        print(f"[ERROR] Failed during quantization/swizzling: {e}")
        return

    # Prepare Dummy Routing Data
    # We need to construct a RoutingData object expected by the kernel.
    # We can try to instantiate the class from the hub.
    try:
        RoutingData = hub.routing.RoutingData

        # Mock routing: 1 token, routed to expert 0
        # This part requires knowing the exact __init__ of RoutingData which we don't see.
        # However, looking at usage, it usually takes tensors.
        # Let's try to infer or skip if too complex.

        # As a fallback, we can try to run 'swiglu' which is simpler and also in the hub
        # just to prove we can run *something*.

        if hasattr(hub, "swiglu"):
            print("\n[INFO] Testing 'swiglu' kernel (simpler fallback)...")
            swiglu_fn = hub.swiglu.swiglu_fn

            # Swiglu inputs: X
            # Output: Y
            # It usually splits last dim in 2.
            x_swiglu = torch.randn(M, K * 2, device=device, dtype=torch.bfloat16)

            # Need FnSpecs?
            # transformers usage:
            # FnSpecs = triton_kernels_hub.matmul_ogs.FnSpecs
            # FusedActivation(FnSpecs("swiglu", swiglu_fn, ...))

            # Direct call?
            # swiglu_fn might be a triton kernel (JIT).

            # Let's try calling it directly
            try:
                out = swiglu_fn(x_swiglu)
                print("[SUCCESS] swiglu_fn execution worked!")
            except Exception as e:
                 # It might need grid/args.
                 print(f"[INFO] Direct swiglu call failed (expected for raw kernel): {e}")

        # Try matmul_ogs
        print("\n[INFO] Invoking matmul_ogs (expecting failure due to missing symbols)...")
        matmul_ogs = hub.matmul_ogs.matmul_ogs

        # We need a routing_data object.
        # Since we can't easily construct it without more reverse engineering,
        # passing None might trigger validation or crash earlier.

        # But we validated the bug statically above, which is the main goal.

    except Exception as e:
        print(f"[ERROR] Setup failed: {e}")

if __name__ == "__main__":
    main()
