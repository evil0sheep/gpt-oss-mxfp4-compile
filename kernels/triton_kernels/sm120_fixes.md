# SM120 Fixes (Blackwell)

## Disabled Swizzling Layouts

In `kernels/triton_kernels/tensor_details/layout.py`, we temporarily disabled the automatic selection of `BlackwellMXValueLayout` and `BlackwellMXScaleLayout` for Compute Capability >= 10.0 (SM120).

Instead, we force `StridedLayout` (no swizzling) and print a warning: `WARNING: Swizzling is disabled for Blackwell`.

**Reason:** The kernel implementation (`matmul_ogs_details/_matmul_ogs.py`) currently contains static assertions that only allow `HOPPER_VALUE` or `None` for `SWIZZLE_MX_VALUE`. Passing `BLACKWELL_VALUE` caused a compile-time assertion failure.

## Known Issues

- `_reduce_grouped` compilation failure during `torch.compile` runs. Suspected scope issue where `load_scale` (and possibly other helpers) are not visible to the JIT compiler when invoked from a different module context.