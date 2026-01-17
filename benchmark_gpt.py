import argparse
import time
import torch
import torch._inductor.config

# Force in-process compilation to support monkeypatched kernels
# and prevent issues with subprocess worker isolation
torch._inductor.config.compile_threads = 1

from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

def benchmark_forward(model, input_ids, num_steps=100):
    """
    Benchmarks a simple forward pass without caching.
    """
    # Warmup
    print("  Warmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
    torch.cuda.synchronize()

    # Benchmark
    print(f"  Running {num_steps} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_steps):
        # Mark step for CUDA Graphs (essential for reduce-overhead stability)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_steps
    return avg_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to benchmark")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(0)

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
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Prepare input: Single token context (Batch=1, Seq=1)
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 1), device="cuda")

    print("\n=== Benchmarking Eager Execution ===")
    try:
        eager_avg = benchmark_forward(model, input_ids, args.steps)
        print(f"Eager Average Latency: {eager_avg*1000:.4f} ms/pass")
    except Exception as e:
        print(f"[ERROR] Eager benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        eager_avg = None

    print("\n=== Benchmarking Compiled Execution ===")
    print("Compiling model with torch.compile(backend='inductor', mode='reduce-overhead')...")

    # Compile the model
    # Since inputs are static (1, 1) and no cache, this should work well with reduce-overhead
    optimized_model = torch.compile(model, backend="inductor", mode="reduce-overhead")

    try:
        compiled_avg = benchmark_forward(optimized_model, input_ids, args.steps)
        print(f"Compiled Average Latency: {compiled_avg*1000:.4f} ms/pass")

        if eager_avg:
            speedup = eager_avg / compiled_avg
            print(f"Speedup: {speedup:.2f}x")

    except Exception as e:
        print(f"[ERROR] Compiled benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
