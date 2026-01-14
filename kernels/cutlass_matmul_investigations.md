# CUTLASS Mixed Precision Matmul Investigations (SM120)

This document summarizes the investigation, implementation attempts, and findings regarding adding W4A16 (FP4 weights, BF16 activations) and W4A8 (FP4 weights, FP8 activations) support to the CUTLASS kernels on NVIDIA Blackwell (SM120) architecture.

## 1. Feasibility Analysis

### W4A16 (FP4 Weights, BF16 Activations)
*   **Feasibility:** **Feasible but requires structural changes.**
*   **Analysis:** The current kernel targets `OpClassBlockScaledTensorOp` (MXFP4), which mandates block-scaled formats for *both* operands. W4A16 typically involves dequantizing weights to BF16 in registers before computing the dot product using standard Tensor Ops (`OpClassTensorOp`).
*   **Implementation Path:** Requires implementing a new kernel path using standard BF16 Tensor Ops with a custom Mainloop iterator that performs on-the-fly dequantization of FP4 weights to BF16. This cannot use the existing `BlockScaledTensorOp` pipeline.
*   **Contrast:** Triton kernels achieve this by loading FP4 and dequantizing in software before the dot product.

### W4A8 (FP4 Weights, FP8 Activations)
*   **Feasibility:** **High (Hardware Supported).**
*   **Analysis:** SM120 hardware natively supports mixed precision block-scaled operations (e.g., E4M3 x E2M1).
*   **Requirements:**
    *   Use specific wrapper types: `cutlass::mx_float8_t` (for A) and `cutlass::mx_float4_t` (for B).
    *   Scale factors for mixed precision (MXFP) must be **UE8M0** (`uint8`), unlike pure NVFP4 which uses **UE4M3**.
    *   Strict alignment requirements: Scale tensors must be 128-byte aligned, necessitating specific group sizes and strides (e.g., `SfVectorSize=32` requires `K` to be a multiple of 32 and scale tensors to have compatible strides).

## 2. Implementation & Findings

### W4A4 (Pure NVFP4)
*   **Status:** **Fully Functional.**
*   **Configuration:**
    *   ArchTag: `cutlass::arch::Sm120`
    *   Types: `cutlass::nv_float4_t` (A & B)
    *   Scales: `cutlass::float_ue4m3_t`
    *   Schedule: `KernelScheduleAuto`
*   **Fixes Applied:**
    *   Applied alignment fixes from `m96-chan/fix/sm120-alignment` branch to resolve `misaligned address` crashes on SM120.
    *   Linked `-lcuda` to resolve `cuDriverGetVersion` symbol errors.

### W4A8 (Mixed Precision)
*   **Status:** **Blocked / Experimental.**
*   **Configuration Attempts:**
    *   ArchTag: Tried `Sm120` and `Sm100`.
    *   Types: `cutlass::mx_float8_t` (A), `cutlass::mx_float4_t` (B).
    *   Scales: `cutlass::float_ue8m0_t` (verified correct type).
    *   Schedule: Tried `KernelScheduleAuto` and explicit `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100`.
*   **Errors Encountered:**
    *   **Runtime (`Sm120` + `Auto`):** `status=7` (Invalid Problem) or `cudaFuncSetAttribute: invalid resource handle`. This indicates the generic SM90 kernel wrapper selected by `Auto` likely does not support the specific instructions or operand layout for mixed-precision block scaling.
    *   **Compilation (`Sm100` + Explicit Schedule):** `CollectiveBuilder` fails to build the Epilogue (`static assertion failed`), suggesting incompatibility between the `Sm100` mixed-precision mainloop schedule and the requested Epilogue configuration.
*   **Conclusion:** W4A8 requires a specific, validated combination of Schedule, Tile Shape, and Epilogue policy (likely mirroring Example 92's MoE setup) that is not currently fully exposed via the generic `GemmUniversal` builder path used in this project.

## 3. Reproduction Steps

### Prerequisites
1.  **Environment:** NVIDIA Blackwell (SM120) GPU (e.g., RTX 5090).
2.  **CUTLASS Source:** Ensure `deps/cutlass` is on branch `fix/sm120-alignment` from `https://github.com/m96-chan/cutlass.git` to fix alignment crashes.

### Commands

**1. Setup Environment**
```bash
# Assuming you are in the project root
cd gpt-oss-test
export PATH=$PWD/.venv/bin:$PATH
```

**2. Run Tests**
The test script `kernels/test_matmul_mxfp4_cutlass.py` attempts to run both W4A4 and W4A8.

```bash
# This will verify W4A4 passes. W4A8 is currently disabled/fails.
time python3 kernels/test_matmul_mxfp4_cutlass.py
```

**3. Inspecting W4A8 Failure (If enabled)**
To see the detailed initialization failure for W4A8:
1.  Edit `kernels/matmul_mxfp4_cutlass.cu`: Uncomment/enable the W4A8 instantiation in `cutlass_fp4_group_mm` (currently guarded or disabled).
2.  Edit `kernels/matmul_mxfp4_cutlass.cu`: Ensure `#define CUTLASS_DEBUG_TRACE_LEVEL 1` is present at the top.
3.  Re-run the test command. You will see output like `Gemm::initialize failed: Error Internal` or `invalid resource handle`.

## 4. Key Code References

*   **Kernel Source:** `kernels/matmul_mxfp4_cutlass.cu` - Contains the `GemmUniversal` adapter and builder logic.
*   **Test Script:** `kernels/test_matmul_mxfp4_cutlass.py` - Python test harness.
*   **CUTLASS Fixes:** `deps/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` (Alignment fixes).
*   **Reference Example:** `deps/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` (Reference for mixed precision on Blackwell).