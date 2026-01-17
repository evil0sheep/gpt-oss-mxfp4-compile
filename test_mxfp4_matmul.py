import torch
import torch.nn as nn
import os
import sys
import inspect

# Add transformers to path if needed (though it should be installed in venv)
# sys.path.append(os.path.join(os.getcwd(), "deps", "transformers", "src"))

def main():
    print("--- Testing MXFP4 Matmul Kernel loading ---")

    # 1. Load the kernels hub from installed package
    try:
        import triton_kernels
        import triton_kernels.matmul_ogs
        import triton_kernels.swiglu
        import triton_kernels.routing
        import triton_kernels.tensor
        import triton_kernels.tensor_details
        import triton_kernels.numerics_details
        try:
            import triton_kernels.numerics_details.mxfp
        except ImportError:
            pass

        print("[INFO] Successfully imported 'triton_kernels' and submodules")
    except ImportError as e:
        print(f"[ERROR] Could not import 'triton_kernels'. Is it installed? {e}")
        return

    # Use the installed package as the hub
    hub = triton_kernels
    print(f"[INFO] Hub loaded from installed package: {hub}")

    # 2. Inspect the source for the known bug
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
        # Mock routing: 1 token, routed to expert 0
        # This part requires knowing the exact __init__ of RoutingData which we don't see.
        # However, looking at usage, it usually takes tensors.
        # Let's try to infer or skip if too complex.

        # As a fallback, we can try to run 'swiglu' which is simpler and also in the hub
        # just to prove we can run *something*.

        if hasattr(hub, "swiglu"):
            print("\n[INFO] Testing 'swiglu' kernel (simpler fallback)...")
            try:
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
            except Exception as e:
                 print(f"[WARN] Accessing swiglu_fn failed: {e}")

        # Try matmul_ogs
        print("\n[INFO] Invoking matmul_ogs...")
        matmul_ogs = hub.matmul_ogs.matmul_ogs
        PrecisionConfig = hub.matmul_ogs.PrecisionConfig

        # We need to wrap this in a Module to test torch.compile
        class HFMatmulWrapper(nn.Module):
            def __init__(self, func, precision_config_cls):
                super().__init__()
                self.func = func
                self.precision_config_cls = precision_config_cls

            def forward(self, x, w, w_scale):
                # Construct config inside (or passed in)
                pc = self.precision_config_cls(weight_scale=w_scale, out_dtype=torch.bfloat16)
                # Call function. x, w, bias=None, ...
                return self.func(x, w, None, precision_config=pc)

        model = HFMatmulWrapper(matmul_ogs, PrecisionConfig).to(device)

        print("Running eager execution...")
        try:
            out_eager = model(x, w_q_sw, w_s_sw)
            print("[SUCCESS] Eager execution worked!")
            print(f"Output shape: {out_eager.shape}")
        except Exception as e:
            print(f"[ERROR] Eager execution failed: {e}")

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
        print(f"[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
