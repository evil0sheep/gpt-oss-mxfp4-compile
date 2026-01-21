import torch
import triton
import triton.language as tl
from kernels.include import unswizzle_mx_scale_bw

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_mxfp4_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    a_scales_ptr, b_scales_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Offsets
    # -----------------------------------------------------------
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # -----------------------------------------------------------
    # B Scales Offsets (Blackwell)
    # -----------------------------------------------------------
    MX_PACK_DIVISOR: tl.constexpr = 32
    MX_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // MX_PACK_DIVISOR
    PACKED_MX_BLOCK: tl.constexpr = (MX_SCALE_BLOCK_K // 4) * 32 * 4 * 4
    SCALE_BLOCK_N: tl.constexpr = BLOCK_N // 128

    # Offset for N-blocks of scales
    offs_n_scale = (pid_n * SCALE_BLOCK_N + tl.arange(0, SCALE_BLOCK_N))

    # Pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am)
    a_scale_ptrs = a_scales_ptr + (offs_m[:, None] * stride_asm)

    # Base pointer for B scales (swizzled)
    # Stride bsn should jump between N-blocks
    b_scale_ptrs_base = b_scales_ptr + (offs_n_scale[:, None] * stride_bsn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_idx * BLOCK_K

        # --- Load A [BLOCK_M, BLOCK_K] ---
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptr_k = a_ptrs + (offs_k[None, :] * stride_ak)
        a = tl.load(a_ptr_k, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # --- Load B [BLOCK_K/2, BLOCK_N] ---
        # B is [N, K/2]. We load as [BLOCK_N, BLOCK_K/2] then transpose logic implicitly by pointer mapping?
        # We want `b` tensor to be [BLOCK_K/2, BLOCK_N] for dot_scaled.
        # Maps to: A [M, K] * B_scaled [K, N]
        # B packed is [BLOCK_K/2, BLOCK_N].
        # In memory B is [N, K/2].
        # Ptr = base + offs_n * stride_bn + offs_k_packed * stride_bk
        offs_k_packed = (k_start // 2) + tl.arange(0, BLOCK_K // 2)
        b_ptr_k = b_ptr + (offs_n[None, :] * stride_bn) + (offs_k_packed[:, None] * stride_bk)
        b = tl.load(b_ptr_k, mask=(offs_n[None, :] < N) & (offs_k_packed[:, None] < K // 2), other=0.0)

        # --- Load A Scales [BLOCK_M, BLOCK_K/32] ---
        offs_k_scale_a = (k_start // 32) + tl.arange(0, BLOCK_K // 32)
        a_scale_ptr_k = a_scale_ptrs + (offs_k_scale_a[None, :] * stride_ask)
        a_scales = tl.load(a_scale_ptr_k, mask=(offs_m[:, None] < M) & (offs_k_scale_a[None, :] < K // 32), other=1)

        # --- Load B Scales (Blackwell Swizzled) ---
        # Stride bsk assumed 1
        offs_k_scale_b = PACKED_MX_BLOCK * k_idx + tl.arange(0, PACKED_MX_BLOCK)
        b_scale_ptrs_now = b_scale_ptrs_base + (offs_k_scale_b[None, :] * stride_bsk)
        b_scales_packed = tl.load(b_scale_ptrs_now)
        b_scales = unswizzle_mx_scale_bw(b_scales_packed)

        # --- Dot Scaled ---
        acc = tl.dot_scaled(
            a, a_scales, "e4m3",
            b, b_scales, "e2m1",
            acc=acc,
        )

    # Store
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)

def matmul_mxfp4(a, b, scales_a, scales_b):
    """
    a: [M, K] float8_e4m3fn
    b: [N, K/2] uint8 (packed float4_e2m1)
    scales_a: [M, K//32] uint8
    scales_b: [N_blocks, ...] Swizzled B scales
    """
    M, K = a.shape
    N = b.shape[0]

    # Handle strides for swizzled scales_b
    if scales_b.ndim == 5:
        # Assumed shape (1, N_blocks, K_chunks, 2, 256)
        stride_bsn = scales_b.stride(1)
        stride_bsk = 1
    elif scales_b.ndim == 2:
        # Fallback if passed as 2D
        stride_bsn = scales_b.stride(0)
        stride_bsk = scales_b.stride(1)
    else:
        # Try to guess or default (e.g. if flattened or 1D)
        # Assuming last dim is contiguous
        stride_bsk = 1
        # If we can't determine N stride, we might error, but let's try stride(0) if available
        stride_bsn = scales_b.stride(0) if scales_b.ndim > 0 else 1

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    matmul_mxfp4_kernel[grid](
        a, b, c,
        scales_a, scales_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scales_a.stride(0), scales_a.stride(1),
        stride_bsn, stride_bsk,
    )
    return c
