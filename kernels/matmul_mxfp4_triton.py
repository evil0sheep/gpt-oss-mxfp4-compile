import torch
import triton
import triton.language as tl
import sys

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
    """
    Kernel for computing Matmul with MXFP4 weights and FP8 inputs.
    A: [M, K]  (FP8 e4m3fn)
    B: [N, K]  (Unpacked FP4 -> stored as uint8 [N, K])
    C: [M, N]  (BF16)

    Scales are per block of 32 elements in K dimension.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # We iterate over K
    # BLOCK_K must be multiple of 32 for scaling blocks
    # BLOCK_K must be multiple of 2 for packing

    # Pointers for A
    # A shape [M, K]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am)

    # Pointers for B
    # B: [N, K] (unpacked for fallback)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn)

    # Scales
    # A scales: [M, K/32]
    a_scale_ptrs = a_scales_ptr + (offs_am[:, None] * stride_asm)
    # B scales: [N, K/32]
    b_scale_ptrs = b_scales_ptr + (offs_bn[:, None] * stride_bsn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Calculate offsets for this K block
        k_start = k * BLOCK_K

        # --- Load A ---
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Check boundaries if K not multiple of BLOCK_K? assume aligned for now.
        a = tl.load(a_ptrs + offs_k[None, :] * stride_ak)

        # --- Load B ---
        # B is unpacked [N, K] for this fallback
        b = tl.load(b_ptrs + offs_k[None, :] * stride_bk)

        # --- Load Scales ---
        # Scale block size is 32.
        # We broadcast scales to match [BLOCK_M/N, BLOCK_K]
        offs_k_scale_repeated = ((k_start + tl.arange(0, BLOCK_K)) // 32)

        a_scale = tl.load(a_scale_ptrs + offs_k_scale_repeated[None, :] * stride_ask)
        b_scale = tl.load(b_scale_ptrs + offs_k_scale_repeated[None, :] * stride_bsk)

        # --- Dot Product ---
        # Fallback to standard dot with cast

        # Dequantize A (simplistic)
        a_bf16 = a.to(tl.bfloat16) * a_scale.to(tl.bfloat16)

        # Dequantize B (simplistic)
        # Treating uint8 as if it was the value for now (or cast to bf16)
        b_bf16 = b.to(tl.bfloat16) * b_scale.to(tl.bfloat16)

        acc = tl.dot(a_bf16, tl.trans(b_bf16), acc=acc)

    # Store Output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)

def matmul_mxfp4(a, b, scales_a, scales_b):
    """
    a: [M, K] float8_e4m3fn
    b: [N, K] uint8 (unpacked float4_e2m1 for fallback)
    scales_a: [M, K//32] float8_e8m0 (uint8)
    scales_b: [N, K//32] float8_e8m0 (uint8)
    """
    M, K = a.shape
    N, K_unpacked = b.shape
    # assert K_packed * 2 == K, "B last dim must be K/2"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # Grid
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    # Let autotune handle it usually, but we need to provide grid function if not using default
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
        scales_b.stride(0), scales_b.stride(1),
    )
    return c
