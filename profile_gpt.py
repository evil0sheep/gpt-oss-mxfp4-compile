import argparse
import os
import torch
import torch._inductor.config
from torch.profiler import profile, record_function, ProfilerActivity

# Force in-process compilation to support monkeypatched kernels
# and prevent issues with subprocess worker isolation
torch._inductor.config.compile_threads = 1

from transformers import AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

def run_profile(model, input_ids, trace_path):
    """
    Runs a single step under profiler after warmup and saves the trace.
    """
    print(f"  Warmup (5 steps)...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids, use_cache=False)
    torch.cuda.synchronize()

    print(f"  Profiling (1 step)...")

    # Configure profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_step"):
            # Mark step for CUDA Graphs (essential for reduce-overhead stability)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad():
                _ = model(input_ids, use_cache=False)
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"  Saved trace to {trace_path}")

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Create traces directory if it doesn't exist
    os.makedirs("traces", exist_ok=True)

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

    # Profile Eager
    print("\n=== Profiling Eager Execution ===")
    try:
        run_profile(model, input_ids, "traces/eager_trace.json")
    except Exception as e:
        print(f"[ERROR] Eager profiling failed: {e}")
        import traceback
        traceback.print_exc()

    # Profile Compiled
    print("\n=== Profiling Compiled Execution ===")
    print("Compiling model with torch.compile(backend='inductor', mode='reduce-overhead')...")

    # Compile the model
    # Note: compilation will happen lazily upon first execution in the benchmark function
    # but we do a dummy run first to ensure compilation happens before profiling loop if possible,
    # though the profiler loop handles warmup too.
    optimized_model = torch.compile(model, backend="inductor", mode="reduce-overhead")

    # Force compilation before profiling to avoid capturing compilation in the trace
    print("  Triggering compilation...")
    try:
        with torch.no_grad():
             _ = optimized_model(input_ids, use_cache=False)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[ERROR] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        run_profile(optimized_model, input_ids, "traces/compiled_trace.json")
    except Exception as e:
        print(f"[ERROR] Compiled profiling failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
