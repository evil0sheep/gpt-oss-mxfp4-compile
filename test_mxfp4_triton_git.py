import torch
import torch.nn as nn
import os
import sys
import inspect

# Add local triton kernels to path
# The structure is deps/triton/python/triton_kernels/triton_kernels
# So we add deps/triton/python/triton_kernels to sys.path to be able to import triton_kernels
repo_root = os.path.dirname(os.path.abspath(__file__))
triton_kernels_path = os.path.join(repo_root, "deps", "triton", "python", "triton_kernels")

if os.path.exists(triton_kernels_path):
    # Insert at beginning to ensure we pick up the local version
    sys.path.insert(0, triton_kernels_path)
    print(f"[INFO] Added {triton_kernels_path} to sys.path")
else:
    print(f"[ERROR] Path {triton_kernels_path} does not exist!")
    sys.exit(1)

def main():
    print("--- Testing MXFP4 Matmul Kernel loading from LOCAL GIT REPO ---")

    try:
        import triton_kernels
        print(f"[INFO] Successfully imported 'triton_kernels' from {triton_kernels.__file__}")

        # Since local __init__.py might be empty, we need to explicitly import submodules
        # to ensure they are attached to the triton_kernels module namespace
        import triton_kernels.matmul_ogs
        import triton_kernels.swiglu
        import triton_kernels.routing

    except ImportError as e:
        print(f"[ERROR] Could not import 'triton_kernels' or its submodules: {e}")
        return

    # Use the imported module as 'hub'
    hub = triton_kernels

    # 2. Inspect the source
    if hasattr(hub, "matmul_ogs"):
        matmul_ogs_mod = hub.matmul_ogs

        if hasattr(matmul_ogs_mod, "matmul_ogs"):
            func = matmul_ogs_mod.matmul_ogs
            src_file = inspect.getfile(func)
            print(f"[INFO] matmul_ogs source file: {src_file}")
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
    M, K, N = 1, 4096, 4096 # Large size to trigger Split-K

    # Input X
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    # Weight W (Targeting Dense/Linear equivalent)
    # Shape: (Out, In) = (N, K)
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    print("[INFO] Quantizing and Swizzling weights...")
    w_q_sw, w_s_sw = None, None

    try:
        # Note: We pass our loaded 'hub' (local module) to these functions
        w_q, w_s = quantize_to_mxfp4(w, hub)

        # Swizzle: transformers does transpose(-1, -2) before swizzling for Linear layers
        # w is (N, K). transpose -> (K, N).
        w_q_sw, w_s_sw = swizzle_mxfp4(w_q.transpose(-1, -2), w_s, hub)
        print("[INFO] Weights prepared successfully.")
    except Exception as e:
        print(f"[ERROR] Failed during quantization/swizzling: {e}")
        # Proceeding might be difficult without prepared weights, but we try swiglu at least

    # Prepare Dummy Routing Data (Not used in this simplified test, but checking import)
    if hasattr(hub, "routing"):
        try:
             # Just checking if we can access RoutingData
            _ = hub.routing.RoutingData
        except AttributeError:
            pass

    # Fallback: swiglu
    if hasattr(hub, "swiglu"):
        print("\n[INFO] Testing 'swiglu' kernel (simpler fallback)...")
        try:
            swiglu_fn = hub.swiglu.swiglu_fn
            # Swiglu inputs: X
            # Output: Y
            # It usually splits last dim in 2.
            x_swiglu = torch.randn(M, K * 2, device=device, dtype=torch.bfloat16)

            out = swiglu_fn(x_swiglu)
            print("[SUCCESS] swiglu_fn execution worked!")
        except Exception as e:
             # It might need grid/args.
             print(f"[INFO] Direct swiglu call failed: {e}")

    # Try matmul_ogs
    if hasattr(hub, "matmul_ogs") and w_q_sw is not None:
        print("\n[INFO] Invoking matmul_ogs...")
        try:
            matmul_ogs = hub.matmul_ogs.matmul_ogs
            PrecisionConfig = hub.matmul_ogs.PrecisionConfig

            # We need to wrap this in a Module to test torch.compile
            class GitMatmulWrapper(nn.Module):
                def __init__(self, func, precision_config_cls):
                    super().__init__()
                    self.func = func
                    self.precision_config_cls = precision_config_cls

                def forward(self, x, w, w_scale):
                    # Construct config inside (or passed in)
                    pc = self.precision_config_cls(weight_scale=w_scale, out_dtype=torch.bfloat16)
                    # Call function. x, w, bias=None, ...
                    return self.func(x, w, None, precision_config=pc)

            model = GitMatmulWrapper(matmul_ogs, PrecisionConfig).to(device)

            print("Running eager execution...")
            try:
                out_eager = model(x, w_q_sw, w_s_sw)
                print("[SUCCESS] Eager execution worked!")
                print(f"Output shape: {out_eager.shape}")
            except Exception as e:
                print(f"[ERROR] Eager execution failed: {e}")
                raise e

            print("\n[INFO] Testing torch.compile on matmul_ogs...")
            compiled_model = torch.compile(model, backend="inductor", mode="reduce-overhead")

            try:
                print("Running warmup...")
                out_c = compiled_model(x, w_q_sw, w_s_sw)
                print("[SUCCESS] Compiled execution (warmup) worked!")

                print("Running benchmark...")
                for _ in range(5):
                    out_c = compiled_model(x, w_q_sw, w_s_sw)
                print("[SUCCESS] Compiled execution benchmark worked!")

            except Exception as e:
                print(f"[ERROR] Compiled execution failed: {e}")
                import traceback
                traceback.print_exc()

        except Exception as e:
            print(f"[ERROR] Execution flow failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
