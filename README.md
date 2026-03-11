# GPT-OSS-20B CUTLASS MXFP4 Integration

CUTLASS-based W4A8 (FP8 activations, FP4 weights) MoE expert implementation for GPT-OSS-20B on NVIDIA Blackwell (SM120), with full `torch.compile` compatibility.

Replaces the triton-based `Mxfp4GptOssExperts` with a CUTLASS grouped GEMM backend that includes GPU-optimized scale conversion and fused O(n) routing.

## Repository Structure

```
gpt-oss-test/
├── kernels/                          # CUTLASS kernel source and tests
│   ├── matmul_mxfp4_cutlass.cu       # CUDA kernel: W4A4/W4A8 grouped GEMM,
│   │                                 #   GPU scale conversion, fused MoE routing
│   ├── cutlass_experts.py            # CutlassGptOssExperts module (PyTorch)
│   ├── include/common.hpp            # Shared CUTLASS header utilities
│   ├── setup.py                      # setuptools build configuration
│   ├── test_cutlass_experts.py       # Unit tests (12 tests + profiling)
│   ├── test_matmul_mxfp4_cutlass.py  # Basic W4A4/W4A8 execution tests
│   ├── benchmark_matmul_mxfp4_cutlass.py  # Kernel-level benchmarks
│   └── cutlass_matmul_investigations.md   # Technical investigation notes
├── test_cutlass_model.py             # Full GPT-OSS-20B model integration test
├── test_mxfp4_matmul.py              # torch.compile + correctness vs triton
├── benchmark_optimizations.py        # E2E model benchmark (optimization impact)
├── GPT-OSS_Blackwell.yaml            # Blackwell model configuration
├── deps/                             # External dependencies (git submodules)
│   ├── cutlass/                      # CUTLASS (branch: fix/sm120-alignment)
│   ├── transformers/                 # HuggingFace transformers fork
│   ├── triton/                       # Triton compiler fork
│   ├── vllm/                         # vLLM inference framework fork
│   ├── kernels/                      # Reference triton kernels
│   └── marlin/                       # Marlin quantization library
└── .venv/                            # Python 3.13 virtual environment
```

## Prerequisites

### Hardware
- NVIDIA Blackwell GPU (SM120 / compute capability 12.0)

### Software
- Python 3.13+
- CUDA 12.x with SM120 support
- PyTorch with CUDA 12.x and float8 support

### Dependencies

The project uses a Python virtual environment with key packages:
- `torch` (with CUDA)
- `transformers` (editable install from `deps/transformers/src`)
- `safetensors`, `huggingface_hub`
- `ninja`, `cmake` (for JIT CUDA compilation)

The CUTLASS extension is JIT-compiled on first use via `torch.utils.cpp_extension.load()`. No pre-build step is needed.

### CUTLASS
The project depends on a patched CUTLASS at `deps/cutlass` (branch `fix/sm120-alignment` from `https://github.com/m96-chan/cutlass.git`). This branch contains SM120 block-scaled alignment fixes required for the W4A8 kernel.

### Model Checkpoint
Tests that run the full GPT-OSS-20B model require the checkpoint cached locally:
```
~/.cache/huggingface/hub/models--openai--gpt-oss-20b/
```

## How It Works

### Architecture
GPT-OSS-20B uses a Mixture-of-Experts (MoE) architecture with 32 experts per layer, top-4 routing, and MXFP4-quantized weights. The standard inference path uses triton kernels for W4A4 (FP4 activations, FP4 weights) computation.

This project replaces that with CUTLASS W4A8 (FP8 activations, FP4 weights), gaining activation precision while maintaining the same FP4 weight format from the checkpoint.

### Forward Pass Pipeline
1. **Fused routing** -- CUDA counting sort kernel (O(n)) assigns tokens to experts
2. **FP8 quantization** -- BF16 activations quantized to float8_e4m3fn with UE8M0 block scales
3. **GPU scale conversion** -- Activation scales permuted to CUTLASS blocked layout on GPU
4. **CUTLASS grouped GEMM** -- gate_up_proj (FP8 x FP4 -> BF16)
5. **SwiGLU activation** -- gate * sigmoid(alpha * gate) * (up + 1)
6. **FP8 re-quantization** -- Intermediate activations re-quantized
7. **CUTLASS grouped GEMM** -- down_proj (FP8 x FP4 -> BF16)
8. **Weighted scatter** -- Results scaled by routing weights and scattered back

### Key Technical Details
- **Scale layout**: CUTLASS SM120 block-scaled MMA requires scales in a specific CuTe tiled layout, not row-major. See `cutlass_matmul_investigations.md` section 2.2.
- **K-padding**: K dimensions must be multiples of 128 (MMA tile size). GPT-OSS K=2880 is padded to 2944 at load time.
- **torch.compile**: All CUDA kernels registered as `torch.library.custom_op` with `register_fake` implementations for TorchDynamo tracing.

## Running Tests

Activate the virtual environment first:
```bash
export PATH=$PWD/.venv/bin:$PATH
```

### Unit Tests (no model checkpoint needed)
Tests CUTLASS kernel building blocks, expert module, edge cases, scale conversion, and fused routing:
```bash
python3 kernels/test_cutlass_experts.py
```
Runs 12 tests + profiling. First run includes CUDA JIT compilation (~2-3 minutes).

### Kernel-Level Tests
Basic W4A4 and W4A8 execution tests:
```bash
python3 kernels/test_matmul_mxfp4_cutlass.py
```

Correctness validation against triton reference + torch.compile:
```bash
python3 test_mxfp4_matmul.py
```

### Full Model Tests (requires GPT-OSS-20B checkpoint)
Loads the full GPT-OSS-20B model, replaces experts with CUTLASS, tests forward pass and torch.compile:
```bash
python3 test_cutlass_model.py
```

## Running Benchmarks

### Kernel-Level Benchmark
```bash
python3 kernels/benchmark_matmul_mxfp4_cutlass.py
```

### Optimization Impact Benchmark (requires GPT-OSS-20B checkpoint)
Measures the impact of GPU scale conversion and fused routing on end-to-end compiled model forward time:
```bash
python3 benchmark_optimizations.py
```
Compares three configurations:
- **A**: GPU scale conversion + fused CUDA routing (fully optimized)
- **B**: CPU batched scale conversion + fused CUDA routing
- **C**: GPU scale conversion + Python argsort routing

### Results (Blackwell B200, seq_len=128)

| Configuration | Eager (ms) | Compiled (ms) |
|---|---|---|
| Fully optimized | 34.55 | 30.67 |
| CPU scales | 40.93 | 41.49 |
| Python routing | 35.55 | 39.28 |

GPU scale conversion saves 10.82ms (26%) and fused routing saves 8.61ms (22%) in the compiled path.

## Test Coverage Summary

| Test | What it validates |
|---|---|
| Test 1a/1b | FP8 quantization correctness and torch.compile |
| Test 2a/2b | SwiGLU activation correctness and torch.compile |
| Test 3 | Single-expert CUTLASS matmul vs triton reference (0.0% error) |
| Test 4 | Multi-expert grouped GEMM (0.0% error) |
| Test 5 | Full CutlassGptOssExperts end-to-end (rel_err=0.044 vs BF16) |
| Test 6 | CutlassGptOssExperts torch.compile (0.0 max diff) |
| Test 7 | Edge cases: single token, all-to-one-expert, large batch |
| Test 8 | 3D input [batch, seq, hidden] vs 2D equivalence |
| Test 9 | K-padding (K not multiple of 128) + torch.compile |
| Test 10 | Batched vs per-expert scale conversion (exact match) |
| Test 11 | GPU vs CPU scale conversion (exact match) |
| Test 12 | Fused counting-sort routing vs argsort reference |

## Further Reading

- [CUTLASS Investigation Notes](kernels/cutlass_matmul_investigations.md) -- Detailed technical findings, scale layout analysis, and architecture details.
