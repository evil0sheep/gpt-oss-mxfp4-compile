import os
import random
import sys

import numpy as np
import torch

# Set environment variable for MXFP4 support
os.environ["VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"] = "1"

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.workspace import init_workspace_manager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Initializing...")
    set_seed(0)

    model_name = "openai/gpt-oss-20b"
    print(f"Initializing configs for {model_name}...")

    model_config = ModelConfig(
        model=model_name,
        tokenizer=model_name,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        quantization="mxfp4",
    )

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        worker_use_ray=False,
    )

    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=8192,
        is_encoder_decoder=False,
    )

    device_config = DeviceConfig(device=torch.device("cuda"))

    load_config = LoadConfig(
        load_format="auto",
        model_loader_extra_config={},
    )

    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.8,
        swap_space=0,
        cache_dtype="fp8",
        model_config=model_config,
    )

    compilation_config = CompilationConfig()

    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=load_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
    )

    print("Initializing distributed environment...")
    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29500",
            backend="nccl",
        )

    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    print("Initializing GPUModelRunner...")
    init_workspace_manager(torch.device("cuda"))
    runner = GPUModelRunner(vllm_config, torch.device("cuda"))

    print("Loading model...")
    runner.load_model()

    print("Running profile_run (executes dummy forward pass)...")
    runner.profile_run()

    print("Initializing KV cache...")
    kv_cache_specs = [runner.get_kv_cache_spec()]
    # Mock available memory (e.g. 40GB)
    available_gpu_memory = [40 * 1024 * 1024 * 1024]
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )
    runner.initialize_kv_cache(kv_cache_configs[0])

    print("Capturing model...")
    runner.capture_model()

    print(
        "Success! GPUModelRunner executed profile_run, init_kv_cache, and capture_model."
    )


if __name__ == "__main__":
    main()
