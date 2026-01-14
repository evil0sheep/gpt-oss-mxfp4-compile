import argparse
import os
import random
import sys
import time

import numpy as np
import torch

# Add deps/transformers/src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_path = os.path.join(current_dir, "deps", "transformers", "src")
sys.path.append(transformers_path)

print(f"Added {transformers_path} to sys.path")

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mxfp4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MXFP4 quantization",
    )
    args = parser.parse_args()

    print("Initializing...")
    set_seed(0)

    model_name = "openai/gpt-oss-20b"

    if args.mxfp4:
        print(f"Loading model {model_name} with MXFP4 quantization...")
        kwargs = {"quantization_config": Mxfp4Config()}
    else:
        print(f"Loading model {model_name} with bfloat16 and forced dequantization...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Increase position embeddings for large context tests
        config.max_position_embeddings = 131072 * 2

        # Disable MXFP4 kernels
        if hasattr(config, "quantization_config"):
            qc = config.quantization_config
            if isinstance(qc, dict):
                qc["dequantize"] = True
            else:
                qc.dequantize = True

        kwargs = {"config": config}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        print(f"Error loading model via AutoModel: {e}")
        import traceback

        traceback.print_exc()
        return

    print("Model loaded successfully.")

    # Prepare dummy inputs
    # Shape: [batch_size, seq_len]
    batch_size = 1
    seq_len = 128
    vocab_size = model.config.vocab_size

    print("Preparing input tensors...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones((batch_size, seq_len), device="cuda", dtype=torch.long)

    # Run eager forward pass
    print("Running eager forward pass...")
    with torch.no_grad():
        start = time.time()
        output_eager = model(input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        print(f"Eager pass took: {time.time() - start:.4f}s")

    print(
        "Compiling model with torch.compile(backend='inductor', mode='reduce-overhead')..."
    )

    # We compile the forward method of the model (or the model itself)
    optimized_model = torch.compile(
        model,
        backend="inductor",
        mode="reduce-overhead",
    )

    print("Running compiled forward pass (Warmup)...")
    try:
        with torch.no_grad():
            start = time.time()
            output_compiled = optimized_model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            print(f"Compiled warmup took: {time.time() - start:.4f}s")

        print("Running compiled forward pass (Measurement)...")
        with torch.no_grad():
            start = time.time()
            output_compiled = optimized_model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            print(f"Compiled inference took: {time.time() - start:.4f}s")

        print("Success! Torch.compile executed without error.")

    except Exception as e:
        print(f"\nFAILURE: torch.compile execution failed.")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
