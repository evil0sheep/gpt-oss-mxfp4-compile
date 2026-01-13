# GPT-OSS vLLM Setup and Usage Guide

This document captures the commands and configuration used to set up, run, and benchmark `openai/gpt-oss-20b` on NVIDIA Blackwell hardware using vLLM.

## 1. Environment Setup & Installation

Create a virtual environment and install the latest version of vLLM (v0.13.0+ is recommended for Blackwell support).

```bash
# Create virtual environment with Python 3.12
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install vLLM
uv pip install -U vllm
```

## 2. Configuration

Create a configuration file named `GPT-OSS_Blackwell.yaml` to optimize settings for the hardware (FP8 KV cache, async scheduling, etc.).

```yaml
kv-cache-dtype: fp8
compilation-config: '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true}}'
async-scheduling: true
no-enable-prefix-caching: true
max-num-batched-tokens: 8192
```

## 3. Running the Server

Start the vLLM server. We adjust `max-num-seqs` and `gpu-memory-utilization` to ensure stability and avoid OOM errors during initialization.

```bash
nohup vllm serve openai/gpt-oss-20b \
  --config GPT-OSS_Blackwell.yaml \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.8 \
  > vllm.log 2>&1 &
```

You can monitor the logs with:
```bash
tail -f vllm.log
```

## 4. Benchmarking

Run the serving benchmark to measure throughput and latency.

```bash
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model openai/gpt-oss-20b \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos \
  --max-concurrency 256 \
  --num-prompts 1280 \
  --save-result --result-filename vllm_benchmark_serving_results.json
```

## 5. Testing with Python

Use the following script `test_vllm.py` to verify connectivity and inference.

```python
from openai import OpenAI
import sys

def main():
    print("Initializing OpenAI client pointing to vLLM...")
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    model_name = "openai/gpt-oss-20b"
    print(f"Sending request to model: {model_name}")

    try:
        result = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you tell me what model you are?"}
            ]
        )

        print("\nResponse received:")
        print("-" * 50)
        print(result.choices[0].message.content)
        print("-" * 50)

    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run it with:
```bash
python test_vllm.py
```
