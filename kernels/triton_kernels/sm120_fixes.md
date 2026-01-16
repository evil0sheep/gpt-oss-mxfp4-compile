# SM120 Fixes (Blackwell)

## Disabled Swizzling Layouts

In `kernels/triton_kernels/tensor_details/layout.py`, we temporarily disabled the automatic selection of `BlackwellMXValueLayout` and `BlackwellMXScaleLayout` for Compute Capability >= 10.0 (SM120).

Instead, we force `StridedLayout` (no swizzling) and print a warning: `WARNING: Swizzling is disabled for Blackwell`.

**Reason:** The kernel implementation (`matmul_ogs_details/_matmul_ogs.py`) currently contains static assertions that only allow `HOPPER_VALUE` or `None` for `SWIZZLE_MX_VALUE`. Passing `BLACKWELL_VALUE` caused a compile-time assertion failure.

## `_reduce_grouped` Compilation Fix (Inductor Scope Issue)

We encountered a `NameError: 'load_scale' is not defined` (and potentially other scope issues) when compiling specialized versions of `_reduce_grouped` using `torch.compile` (Inductor). This was due to Inductor/Triton failing to propagate the captured globals of dynamically created functions to the Triton compiler's `CodeGenerator` during the final compilation pass.

**Fix:** We applied a workaround in `kernels/triton_kernels/specialize.py`. The `define_kernel` function now monkeypatches `triton.compiler.code_generator.CodeGenerator.builtin_namespace` to include any `JITFunction` objects found in the specialized kernel's globals. This ensures the compiler can resolve helper functions like `load_scale` even if the standard scope lookup fails.