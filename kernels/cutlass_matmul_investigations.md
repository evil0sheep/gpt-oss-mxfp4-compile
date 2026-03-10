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

1. **`__get_group_gemm_starts_w4a8` hardcodes `group_size=16`** (line 383 of `matmul_mxfp4_cutlass.cu`) but the actual MXFP group size (SfVectorSize) is 32. This is benign for single-expert at offset 0 but will cause **incorrect scale pointer offsets for multi-expert MoE workloads**. Must fix before MoE deployment.
   - **Impact**: When expert_offsets > 0, the scale pointer calculation `offset * (K / group_size)` will compute wrong offsets because it divides by 16 instead of 32, producing half the correct number of scale groups per row. This means each expert beyond the first will read scales from the wrong memory location.
   - **Fix**: Change `group_size = 16` to `group_size = 32` in `__get_group_gemm_starts_w4a8`, or better, derive it from `ScaleConfig::SFVecSize`.

2. **Scale conversion is CPU-side and not performance-optimized.** The `convert_{a,b}_scales_for_w4a8()` functions transfer to CPU, reorder, and transfer back. For production:
   - **Weight scales (B)**: Convert once at weight-load time (offline). This is the recommended approach — the CUTLASS blocked layout is deterministic for a given (N, K) shape, so it can be precomputed and cached.
   - **Activation scales (A)**: These change every inference call (depend on input data). Options:
     - (a) Write a simple CUDA kernel that does the row-major → blocked layout permutation on GPU
     - (b) Investigate whether the CUTLASS blocked layout for SFA is actually just row-major for typical M values (it may be, since the scale layout is tiled by MMA tile shape and M is the "dynamic" dimension)
     - (c) Accept the CPU round-trip cost if activation quantization is already CPU-bound
   - **Priority**: High for activation scales (per-inference), low for weight scales (one-time)

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

## 6. Next Steps (Recommended Priority)

1.  **Fix multi-expert scale offset bug** — Change `group_size=16` to `group_size=32` (or derive from ScaleConfig) in `__get_group_gemm_starts_w4a8`.
2.  **Integrate into model inference** — Wire the CUTLASS W4A8 kernel + scale conversion into the GPT-OSS-20B model's expert matmul path (replacing the failing triton kernel path).
3.  **GPU-side scale conversion** — Move `convert_{a,b}_scales_for_w4a8` to a CUDA kernel for production use. Or pre-convert scales at model load time.
4.  **Validate W4A4 correctness** — Need format conversion between UE4M3/group16 and UE8M0/group32 to compare against triton, or write a standalone reference.
5.  **End-to-end torch.compile test** — Run `repro_compile_hftf.py` with CUTLASS backend instead of triton to validate full model compilation.
