import argparse
import os
import random
import sys
import time
import torch

# Disable parallel compilation in Inductor to test subprocess hypothesis
# We must import config before compilation starts
try:
    import torch._inductor.config
    torch._inductor.config.compile_threads = 1
    print("[INFO] Set torch._inductor.config.compile_threads = 1 to force in-process compilation")
except ImportError:
    print("[WARN] Could not import torch._inductor.config")

from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("--- Testing MXFP4 Single Layer Compilation (SERIAL) ---")
    set_seed(0)

    model_name = "openai/gpt-oss-20b"
    print(f"Loading model {model_name} with MXFP4 quantization...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            quantization_config=Mxfp4Config()
        )
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return

    print("Model loaded.")

    target_layer = None
    target_name = ""

    # Traverse to find an GptOssMLP module (which wraps experts and handles routing)
    for name, module in model.named_modules():
        if "GptOssMLP" in module.__class__.__name__:
            target_layer = module
            target_name = name
            break

    if target_layer is None:
        print("[ERROR] No GptOssMLP layer found in the model.")
        return

    print(f"Targeting layer: {target_name} ({target_layer.__class__.__name__})")

    hidden_size = model.config.hidden_size
    batch_size = 1
    seq_len = 128

    print(f"Preparing input (Batch: {batch_size}, Seq: {seq_len}, Hidden: {hidden_size})")
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

    # Eager Execution
    print("\nRunning Eager Execution...")
    try:
        with torch.no_grad():
            output_eager = target_layer(hidden_states)
        print("[SUCCESS] Eager execution worked.")
        if isinstance(output_eager, tuple):
            output_eager = output_eager[0]
        print(f"Output shape: {output_eager.shape}")
    except Exception as e:
        print(f"[ERROR] Eager execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compilation
    print("\nCompiling layer with torch.compile(backend='inductor', mode='reduce-overhead')...")
    compiled_layer = torch.compile(target_layer, backend="inductor", mode="reduce-overhead")

    # Compiled Execution
    print("Running Compiled Execution (Warmup)...")
    try:
        with torch.no_grad():
            # Warmup
            start = time.time()
            output_compiled = compiled_layer(hidden_states)
            torch.cuda.synchronize()
            print(f"Warmup took: {time.time() - start:.4f}s")

        print("Running Compiled Execution (Benchmark)...")
        with torch.no_grad():
            start = time.time()
            output_compiled = compiled_layer(hidden_states)
            torch.cuda.synchronize()
            print(f"Inference took: {time.time() - start:.4f}s")

        print("[SUCCESS] Compiled execution worked!")

    except Exception as e:
        print(f"[ERROR] Compiled execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
