# CUTLASS Mixed Precision Matmul Investigations (SM120)

This document summarizes the investigation, implementation, and findings regarding MXFP4/MXFP8 block-scaled matmul on NVIDIA Blackwell (SM120) using CUTLASS kernels, including `torch.compile` integration and correctness validation.

## 1. Current Status Summary

| Mode | Execution | torch.compile | Correctness vs Triton | Notes |
|------|-----------|---------------|----------------------|-------|
| **W4A4** (NVFP4×NVFP4) | Working | Working | Not tested (scale format mismatch with triton) | UE4M3 scales, group_size=16 |
| **W4A8** (MXFP8×MXFP4) | Working | Working | **Exact match** (0.0% error) | UE8M0 scales, group_size=32 |
| **W4A16** (FP4×BF16) | Not implemented | — | — | Requires different kernel architecture |

## 2. Key Findings

### 2.1 torch.compile Integration
Both W4A4 and W4A8 CUTLASS kernels work under `torch.compile` with zero difference vs eager execution. The integration requires:

1. **Register as custom op** via `torch.library.custom_op` with `mutates_args=("output",)`:
   ```python
   @torch.library.custom_op("cutlass::fp4_group_mm", mutates_args=("output",))
   def fp4_group_mm(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets):
       ext.cutlass_fp4_group_mm(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets)

   @fp4_group_mm.register_fake
   def _(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets):
       pass  # in-place mutation, no new tensors
   ```
2. **Call via `torch.ops.cutlass.fp4_group_mm`** inside compiled functions.
3. This allows TorchDynamo to trace through the custom op without graph breaks.

### 2.2 Scale Factor Layout (CRITICAL FINDING)

**CUTLASS block-scaled MMA on SM120 expects scale factors in a specific blocked/tiled memory layout, NOT simple row-major.** This is defined by `Sm1xxBlockScaledConfig::tile_atom_to_shape_SFA/SFB` (CuTe layouts). Passing row-major scales produces ~34% relative error.

- The layout is hierarchical: a `blocked_product` of MMA-level scale access patterns tiled across the (M, K) or (N, K) dimensions.
- **Solution**: Added C++ helper functions `convert_a_scales_for_w4a8()` and `convert_b_scales_for_w4a8()` that use the same `ScaleConfig` as the kernel to rearrange row-major scales to the correct layout.
- With converted scales, the CUTLASS output matches the triton reference with **0.0% relative error** (bit-exact after bf16 rounding).

### 2.3 W4A8 Correctness Validation

Validated against triton hub's `downcast_to_mxfp_torch` (quantization) + `upcast_from_mxfp_torch` (dequantization) + `torch.matmul`:

| Size | Relative Error | Max Diff |
|------|---------------|----------|
| M=128 N=256 K=512 | 0.000000 | 0.0000 |
| M=256 N=512 K=1024 | 0.000000 | 0.0000 |
| M=128 N=128 K=256 | 0.000000 | 0.0000 |

### 2.4 Known Issues / TODO

1. ~~**`__get_group_gemm_starts_w4a8` hardcodes `group_size=16`**~~ **FIXED.** Changed to `group_size = 32` (matching `ScaleConfig::SFVecSize`). Additionally, scale pointer calculations were overhauled: A scale offsets now use raw element offsets (not `padded_M * group_k`), and B scale offsets use `ldb_blockscale` (stride per expert in the blocked layout buffer). Validated correct with multi-expert grouped GEMM and full GPT-OSS-20B model.

2. ~~**Scale conversion is CPU-side**~~ **OPTIMIZED.** Weight scales (B) are converted once at load time. Activation scales (A) now use `batch_convert_a_scales_for_w4a8_gpu()`: layout cosizes computed on CPU (just 32 ints of template math), actual scale data permuted by a GPU kernel (`convert_a_scales_gpu_kernel`) — no scale data transfer to/from CPU. CuTe layout functions are `CUTE_HOST_DEVICE` so `tile_atom_to_shape_SFA` works in `__global__` kernels.

3. **W4A4 correctness not validated against triton.** The W4A4 path uses UE4M3 (float8_e4m3fn) scale factors with group_size=16, which doesn't match triton's UE8M0/group_size=32 format. A separate reference implementation or format conversion is needed. The W4A4 scale layout likely has the same blocked layout requirement.

4. **W4A16 (FP4 weights, BF16 activations)** requires a fundamentally different kernel architecture using standard `OpClassTensorOp` with on-the-fly dequantization. Cannot reuse the `OpClassBlockScaledTensorOp` pipeline.

## 3. Architecture & Configuration Details

### W4A4 (Pure NVFP4)
*   **Types:** `cutlass::nv_float4_t<cutlass::float_e2m1_t>` (A & B)
*   **Scales:** `cutlass::float_ue4m3_t` (float8_e4m3fn in PyTorch), group_size=16
*   **Schedule:** `KernelPtrArrayTmaWarpSpecializedCooperative`
*   **Tile:** 128×128×128, Cluster 1×1×1

### W4A8 (Mixed Precision)
*   **Types:** `cutlass::mx_float8_t<cutlass::float_e4m3_t>` (A), `cutlass::mx_float4_t<cutlass::float_e2m1_t>` (B)
*   **Scales:** `cutlass::float_ue8m0_t` (uint8 in PyTorch), group_size=32 (SfVectorSize=32)
*   **Schedule:** `KernelPtrArrayTmaWarpSpecializedCooperative`
*   **Tile:** 128×128×128, Cluster 1×1×1
*   **Input Shapes:**
    *   A (activations): `[M, K]` float8_e4m3fn
    *   B (weights): `[E, N, K/2]` uint8 (packed FP4, ColumnMajor)
    *   a_blockscale: `[M, K/32]` uint8 (must be in CUTLASS blocked layout)
    *   b_blockscales: `[E, N, K/32]` uint8 (must be in CUTLASS blocked layout)
*   **Matmul semantics:** C[M,N] = A[M,K] × B^T[K,N] (standard linear layer Y=XW^T)

### Scale Factor Layout Details
*   Defined by `Sm1xxBlockScaledConfig` in `cutlass/detail/sm100_blockscaled_layout.hpp`
*   Uses `tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{})` for SFA
*   `SfAtom` is a `blocked_product` of MMA-level scale layout with tile layout
*   Basic block: `mnBasicBlockShape=(32,4)`, `mnBasicBlockStride=(16,4)`
*   Scale data is organized for TMA (Tensor Memory Accelerator) coalesced reads

### Triton/HuggingFace Quantization Compatibility
*   `downcast_to_mxfp_torch(tensor, torch.uint8, axis=1)` → FP4 packed uint8 + UE8M0 uint8 scales
*   `downcast_to_mxfp_torch(tensor, torch.float8_e4m3fn, axis=1)` → FP8 + UE8M0 uint8 scales
*   Group size: always 32 elements per scale
*   Scale format: UE8M0 = biased exponent, `value = 2^(byte - 127)`
*   FP4 packing: low nibble = even element, high nibble = odd element (matches CUTLASS)
*   **Key difference:** triton stores scales in row-major; CUTLASS needs blocked layout (see §2.2)

## 4. Reproduction Steps

### Prerequisites
1.  **Environment:** NVIDIA Blackwell (SM120) GPU
2.  **CUTLASS Source:** `deps/cutlass` on branch `fix/sm120-alignment` from `https://github.com/m96-chan/cutlass.git`

### Commands

```bash
cd gpt-oss-test
export PATH=$PWD/.venv/bin:$PATH

# Basic execution tests (W4A4 + W4A8, no correctness check)
time python3 kernels/test_matmul_mxfp4_cutlass.py

# Full test: torch.compile + correctness vs triton reference
time python3 test_mxfp4_matmul.py

# Benchmark
time python3 kernels/benchmark_matmul_mxfp4_cutlass.py
```

## 5. Key Code References

*   **Kernel Source:** `kernels/matmul_mxfp4_cutlass.cu`
    *   `run_w4a4_original()` — W4A4 grouped GEMM
    *   `run_w4a8_group_mm_sm120()` — W4A8 grouped GEMM
    *   `convert_a_scales_for_w4a8()` — Scale layout conversion (SFA)
    *   `convert_b_scales_for_w4a8()` — Scale layout conversion (SFB)
    *   `cutlass_fp4_group_mm()` — Main dispatcher (auto-detects W4A4 vs W4A8 from dtype)
*   **Tests:**
    *   `kernels/test_matmul_mxfp4_cutlass.py` — Basic execution test
    *   `test_mxfp4_matmul.py` — torch.compile + correctness test
    *   `kernels/benchmark_matmul_mxfp4_cutlass.py` — Performance benchmark
*   **CUTLASS Layout Definitions:**
    *   `deps/cutlass/include/cutlass/detail/sm100_blockscaled_layout.hpp` — `Sm1xxBlockScaledConfig`, `tile_atom_to_shape_SFA/SFB`
    *   `deps/cutlass/include/cute/atom/mma_traits_sm120.hpp` — MMA trait scale layouts
    *   `deps/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` — TMA setup
*   **Reference Example:** `deps/cutlass/examples/92_blackwell_moe_gemm/`

## 6. Full Model Integration (CutlassGptOssExperts)

### 6.1 Implementation

`CutlassGptOssExperts` (`kernels/cutlass_experts.py`) is a drop-in replacement for `Mxfp4GptOssExperts` that:
- Uses CUTLASS W4A8 grouped GEMM instead of triton's `matmul_ogs`
- Fused CUDA routing kernel (counting sort O(n) vs argsort O(n log n))
- GPU-side activation scale layout conversion (no CPU round-trip for scale data)
- Quantizes activations to FP8 per-inference using standard PyTorch ops
- SwiGLU activation in pure PyTorch
- Fully compatible with `torch.compile`

### 6.2 Additional Findings (Model Integration)

#### K-Dimension Alignment (CRITICAL)
The CUTLASS SM120 block-scaled MMA tile is 128×128×128. **K must be a multiple of 128** (enforced by TMA). GPT-OSS-20B has hidden_size=intermediate_size=2880, which is NOT a multiple of 128 (2880/128=22.5).

**Solution**: Pad K to 2944 (=23×128) at weight-load time for weights/scales, and at runtime for activations. Zero-padding doesn't affect correctness since padded elements multiply to zero. This adds 2.2% overhead in K dimension.

Attempted alternative: Changing MmaTileShape to 128×128×64 fails with `"TMA requires CTA_Tile and SLayout top-level size equivalence"`. The SM120 block-scaled TMA pipeline requires K=128 in the tile.

#### Scale Layout Buffer Sizes (CRITICAL)
The CUTLASS blocked scale layout (`tile_atom_to_shape_SFA/SFB`) produces output buffers whose size may **NOT** be a multiple of K_groups. For example, with N=2880, K=2880 (K_groups=90), the SFB layout produces 270,848 elements (vs 259,200 = N×K_groups). This ratio varies with N and K.

**Impact**: The kernel cannot assume `expert_id * N * group_k` spacing between experts in the B scale buffer. Similarly, A scale offsets cannot use `sf_offset * group_k` indexing.

**Solution**:
- B scales: Added `ldb_blockscale` parameter to the CUDA kernel, computed as `b_blockscales.numel() / num_experts`. Each expert's scales are spaced by `ldb_blockscale` elements.
- A scales: Changed `sf_offsets` to use raw element offsets instead of `padded_M` units.

#### num_local_experts
GPT-OSS-20B has `num_local_experts=32` (not 128 as initially assumed), `num_experts_per_tok=4`.

### 6.3 Validation Results

| Test | Result | Notes |
|------|--------|-------|
| Building blocks (quantize, swiglu, single-expert, multi-expert) | 0.0% error | Exact match vs triton reference |
| CutlassGptOssExperts module | rel_err=0.044 | vs dequantized BF16 reference |
| CutlassGptOssExperts torch.compile | 0.0 max diff | Exact match eager vs compiled |
| Full GPT-OSS-20B forward | 34.55ms eager | Output shape [1, 128, 201088] |
| Full GPT-OSS-20B torch.compile | 30.67ms inference | Warmup ~7s, rel_diff=1.9% vs eager |
| GPU vs CPU scale conversion | 0.0% error | Exact match (Test 11) |
| Fused routing vs argsort | rel_err=0.001 | Exact expert counts/offsets; bf16 order diff only |

### 6.4 Performance Optimization Results

Benchmarked on GPT-OSS-20B full model forward (seq_len=128, 24 layers, 32 experts):

| Configuration | Eager (ms) | Compiled (ms) |
|---|---|---|
| **GPU scales + fused routing (optimized)** | **34.55** | **30.67** |
| CPU batched scales + fused routing | 40.93 | 41.49 |
| GPU scales + Python argsort routing | 35.55 | 39.28 |

**Impact of individual optimizations (compiled):**

| Optimization | Savings | Relative Improvement |
|---|---|---|
| GPU scale conversion (vs CPU batched) | 10.82ms | 26.1% faster |
| Fused routing kernel (vs Python argsort) | 8.61ms | 21.9% faster |
| Combined | ~19ms | ~39% faster |

**Key observations:**
- GPU scale conversion eliminates CPU round-trips for scale data. Only 32 ints (expert counts) go to CPU for CuTe cosize computation.
- Fused routing has outsized compiled impact (8.6ms compiled vs 1ms eager) because `torch.compile` cannot efficiently fuse the Python argsort + scatter_add pattern.
- `torch.compile` provides 11% speedup on the optimized path but *hurts* configs with CPU sync points or Python routing ops (graph breaks prevent optimization).

### 6.5 Key Code References (Updated)

*   **CutlassGptOssExperts:** `kernels/cutlass_experts.py`
    *   `quantize_activations_to_fp8()` — BF16→FP8 quantization
    *   `swiglu()` — SwiGLU activation
    *   `CutlassGptOssExperts` — Full module
    *   `load_cutlass_weights()` — Checkpoint loading + scale conversion + K-padding
*   **CUDA kernels:** `kernels/matmul_mxfp4_cutlass.cu`
    *   `convert_a_scales_gpu_kernel` — GPU-side activation scale layout permutation
    *   `batch_convert_a_scales_for_w4a8_gpu()` — Host launcher for GPU scale conversion
    *   `moe_histogram_kernel` + `moe_sort_kernel` — Fused O(n) counting sort routing
    *   `fused_moe_routing()` — Host function combining histogram + sort
*   **Full model test:** `test_cutlass_model.py`
    *   Loads GPT-OSS-20B with dequantized weights, replaces experts from checkpoint
    *   Tests: forward, torch.compile, correctness vs triton MXFP4 (W4A4)
*   **Unit tests:** `kernels/test_cutlass_experts.py`
    *   Tests 1-6: building blocks, single/multi-expert, end-to-end, torch.compile
    *   Tests 7-9: edge cases (0-token experts, single token, large batch), 3D input, K-padding
    *   Test 10: batched vs per-expert scale conversion consistency
    *   Test 11: GPU vs CPU scale conversion consistency
    *   Test 12: fused routing correctness (counting sort vs argsort)
    *   Profile: forward pass timing breakdown
*   **Optimization benchmark:** `benchmark_optimizations.py`
    *   A/B/C comparison of GPU scale conversion and fused routing impact
    *   Tests eager and compiled model forward times

## 7. Deferred Items

1.  **W4A4 mode** — Not implemented in CutlassGptOssExperts. Only W4A8 is supported. Can be added later.
2.  **Multi-node support** — Only single-node tested. Expert parallelism not implemented.
3.  **K-padding overhead** — 2.2% extra compute from padding K=2880→2944. Could eliminate with a custom CUTLASS tile shape, but SM120 TMA constraints prevent this.
4.  **Proper HF quantizer integration** — Currently uses a standalone script that loads dequantized model then swaps. Should create a proper `CutlassMxfp4HfQuantizer` for cleaner integration with `from_pretrained`.
