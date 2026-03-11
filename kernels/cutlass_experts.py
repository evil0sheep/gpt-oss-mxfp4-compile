"""
CUTLASS-based MoE expert implementation for GPT-OSS with torch.compile support.

Replaces the triton-based Mxfp4GptOssExperts with a CUTLASS W4A8 grouped GEMM
backend that is fully compatible with torch.compile.

Building blocks:
  - quantize_activations_to_fp8: BF16 -> FP8 (e4m3fn) + UE8M0 scales
  - swiglu: SwiGLU activation (gate * sigmoid(alpha*gate) * (up + 1))
  - CutlassGptOssExperts: Full expert module using CUTLASS grouped GEMM
"""

import os
import sys
import torch
from torch import nn
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MXFP_BLOCK_SIZE = 32  # elements per scale factor group
MMA_TILE_K = 128  # CUTLASS MMA tile K dimension — K must be a multiple of this


def pad_k_to_tile(K):
    """Return K padded to the next multiple of MMA_TILE_K."""
    return ((K + MMA_TILE_K - 1) // MMA_TILE_K) * MMA_TILE_K


def pad_fp4_weights(data, scales, K_orig):
    """
    Pad FP4 packed weights and their scales along K to the next MMA tile boundary.

    Args:
        data: [N, K_orig//2] uint8 (packed FP4)
        scales: [N, K_orig//32] uint8 (UE8M0)
        K_orig: original K dimension

    Returns:
        data_padded: [N, K_padded//2] uint8
        scales_padded: [N, K_padded//32] uint8
        K_padded: padded K
    """
    K_padded = pad_k_to_tile(K_orig)
    if K_padded == K_orig:
        return data, scales, K_orig

    N = data.shape[0]
    # Pad packed data: K/2 -> K_padded/2 (zeros for padding)
    pad_data = (K_padded - K_orig) // 2
    data_padded = torch.nn.functional.pad(data, (0, pad_data), value=0)

    # Pad scales: K/32 -> K_padded/32 (127 = 2^0 = 1.0 in UE8M0, but 0 works
    # since padded data is 0 anyway)
    pad_scales = (K_padded - K_orig) // MXFP_BLOCK_SIZE
    scales_padded = torch.nn.functional.pad(scales, (0, pad_scales), value=0)

    return data_padded, scales_padded, K_padded


def pad_fp8_activations(fp8_data, scales, K_orig):
    """
    Pad FP8 activations and their scales along K to the next MMA tile boundary.

    Args:
        fp8_data: [M, K_orig] float8_e4m3fn
        scales: [M, K_orig//32] uint8 (UE8M0)
        K_orig: original K dimension

    Returns:
        data_padded: [M, K_padded] float8_e4m3fn
        scales_padded: [M, K_padded//32] uint8
        K_padded: padded K
    """
    K_padded = pad_k_to_tile(K_orig)
    if K_padded == K_orig:
        return fp8_data, scales, K_orig

    pad_data = K_padded - K_orig
    # FP8 zero padding
    data_padded = torch.nn.functional.pad(fp8_data, (0, pad_data), value=0)

    pad_scales = (K_padded - K_orig) // MXFP_BLOCK_SIZE
    scales_padded = torch.nn.functional.pad(scales, (0, pad_scales), value=0)

    return data_padded, scales_padded, K_padded


# ---------------------------------------------------------------------------
# CUTLASS Extension Loading
# ---------------------------------------------------------------------------

_cutlass_ext = None
_custom_op_registered = False


def get_cutlass_extension():
    """JIT compile and cache the CUTLASS MXFP4 kernel extension."""
    global _cutlass_ext
    if _cutlass_ext is not None:
        return _cutlass_ext

    kernels_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(kernels_dir, ".."))

    cutlass_include = os.path.join(project_root, "deps", "cutlass", "include")
    cutlass_tools_include = os.path.join(project_root, "deps", "cutlass", "tools", "util", "include")
    local_include = os.path.join(kernels_dir, "include")
    source_file = os.path.join(kernels_dir, "matmul_mxfp4_cutlass.cu")

    assert os.path.exists(cutlass_include), f"CUTLASS include not found at {cutlass_include}"
    assert os.path.exists(source_file), f"Kernel source not found at {source_file}"

    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"

    _cutlass_ext = load(
        name="matmul_mxfp4_cutlass_experts",
        sources=[source_file],
        extra_include_paths=[local_include, cutlass_include, cutlass_tools_include],
        extra_cuda_cflags=[
            "-O3", "-std=c++17",
            "-DENABLE_NVFP4_SM120=1",
            "-DENABLE_CUTLASS_MOE_SM120=1",
            "-DCUTLASS_NVCC_ARCHS=120a",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return _cutlass_ext


def ensure_custom_op_registered():
    """Register the CUTLASS kernel as a torch custom op for torch.compile."""
    global _custom_op_registered
    if _custom_op_registered:
        return

    ext = get_cutlass_extension()

    @torch.library.custom_op("cutlass::fp4_group_mm", mutates_args=("output",))
    def fp4_group_mm(
        output: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_blockscale: torch.Tensor,
        b_blockscales: torch.Tensor,
        alphas: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        sf_offsets: torch.Tensor,
    ) -> None:
        ext.cutlass_fp4_group_mm(
            output, a, b, a_blockscale, b_blockscales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )

    @fp4_group_mm.register_fake
    def _(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets):
        pass  # in-place mutation, no new tensors

    _custom_op_registered = True


# ---------------------------------------------------------------------------
# Activation Quantization: BF16 -> FP8 + UE8M0 scales
# ---------------------------------------------------------------------------

def quantize_activations_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize BF16/FP32 activations to float8_e4m3fn with UE8M0 block scales.

    This reimplements triton hub's downcast_to_mxfp_torch for FP8 output,
    using only standard PyTorch ops for torch.compile compatibility.

    Args:
        x: [M, K] tensor in bfloat16 or float32

    Returns:
        x_fp8: [M, K] float8_e4m3fn quantized activations
        x_scale: [M, K // 32] uint8 UE8M0 scale factors
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, K = x.shape
    assert K % MXFP_BLOCK_SIZE == 0, f"K={K} must be divisible by {MXFP_BLOCK_SIZE}"

    device = x.device
    x_f32 = x.float()

    # Compute per-group max absolute value
    # Reshape to [M, K//32, 32] and take max along the group dim
    abs_groups = x_f32.abs().view(M, K // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE)
    max_val = abs_groups.max(dim=-1, keepdim=True).values  # [M, K//32, 1]

    # FP8 e4m3fn max representable value
    max_fp8 = 448.0

    # Compute dequantization scale = max_val / max_fp8
    # Then round UP to nearest power of 2 (UE8M0 format)
    dequant_scale = max_val / max_fp8
    # Bit manipulation: round up to power of 2
    ds_int = dequant_scale.view(torch.int32)
    ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    dequant_scale_rounded = ds_int_rounded.view(torch.float32)

    # Handle zero/subnormal: ensure scale is at least 2^-127 (smallest UE8M0)
    dequant_scale_rounded = torch.where(
        dequant_scale_rounded == 0,
        torch.tensor(1.1754944e-38, device=device, dtype=torch.float32),  # 2^-126
        dequant_scale_rounded,
    )

    # Quantize: x_scaled = x / scale, then cast to FP8
    x_scaled = x_f32.view(M, K // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE) / dequant_scale_rounded
    x_scaled = x_scaled.view(M, K)
    x_fp8 = x_scaled.clamp(-max_fp8, max_fp8).to(torch.float8_e4m3fn)

    # Extract UE8M0 scale: biased exponent from the rounded power-of-2 float
    # UE8M0 = (float32_bits >> 23) & 0xFF
    scale_uint8 = ((ds_int_rounded.view(M, K // MXFP_BLOCK_SIZE, 1) >> 23) & 0xFF).to(torch.uint8)
    scale_uint8 = scale_uint8.squeeze(-1)  # [M, K//32]

    return x_fp8, scale_uint8


# ---------------------------------------------------------------------------
# SwiGLU Activation
# ---------------------------------------------------------------------------

def swiglu(gate_up: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """
    SwiGLU activation: gate * sigmoid(alpha * gate) * (up + 1)

    Input is interleaved: gate = even indices, up = odd indices.

    Args:
        gate_up: [..., 2*N] tensor (gate and up interleaved)
        alpha: sigmoid scaling factor (default 1.702 for GPT-OSS)
        limit: clamping limit for numerical stability

    Returns:
        [..., N] tensor after SwiGLU activation
    """
    gate = gate_up[..., ::2]   # even indices
    up = gate_up[..., 1::2]    # odd indices
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)


# ---------------------------------------------------------------------------
# CutlassGptOssExperts Module
# ---------------------------------------------------------------------------

class CutlassGptOssExperts(nn.Module):
    """
    MXFP4-quantized expert layer using CUTLASS W4A8 grouped GEMM.

    Replaces Mxfp4GptOssExperts with a torch.compile-compatible implementation.
    Weights are stored in CUTLASS-native format (packed FP4 uint8 + blocked-layout scales).

    Forward path:
      1. Route tokens to experts (pure PyTorch)
      2. Quantize activations BF16 -> FP8 + UE8M0 scales
      3. Convert activation scales to CUTLASS blocked layout
      4. CUTLASS grouped GEMM for gate_up_proj
      5. Add bias + SwiGLU activation
      6. Quantize intermediate -> FP8
      7. CUTLASS grouped GEMM for down_proj
      8. Add bias, scale by routing weights, scatter back
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)

        # Biases (kept in FP32)
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )

        # Weights will be set by load_cutlass_weights() after checkpoint loading
        # gate_up_proj_data: [E, 2*intermediate, hidden//2] uint8 (packed FP4)
        # gate_up_proj_scales: [E, 2*intermediate, hidden//32] uint8 (CUTLASS blocked layout)
        # down_proj_data: [E, hidden, intermediate//2] uint8 (packed FP4)
        # down_proj_scales: [E, hidden, intermediate//32] uint8 (CUTLASS blocked layout)
        self.register_buffer("gate_up_proj_data", None)
        self.register_buffer("gate_up_proj_scales", None)
        self.register_buffer("down_proj_data", None)
        self.register_buffer("down_proj_scales", None)

    def _run_cutlass_gemm(
        self,
        activations: torch.Tensor,          # [total_tokens, K] float8_e4m3fn
        act_scales_rowmajor: torch.Tensor,   # [total_tokens, K//32] uint8 (row-major)
        weights: torch.Tensor,               # [E, N, K_padded//2] uint8
        weight_scales: torch.Tensor,         # [E, converted_size] uint8 (CUTLASS blocked layout, flat per expert)
        expert_token_counts: torch.Tensor,   # [E] int32 - number of tokens per expert
        expert_offsets: torch.Tensor,        # [E] int32 - start offset in activations for each expert
        N: int,
        K: int,
        K_padded: int,
    ) -> torch.Tensor:
        """Run CUTLASS W4A8 grouped GEMM across all active experts."""
        ext = get_cutlass_extension()
        total_tokens = activations.shape[0]
        num_experts = expert_token_counts.shape[0]
        device = activations.device

        output = torch.zeros((total_tokens, N), device=device, dtype=torch.bfloat16)

        # Pad activations to K_padded if needed
        if K_padded > K:
            activations, act_scales_rowmajor, _ = pad_fp8_activations(
                activations, act_scales_rowmajor, K
            )

        # Batched scale conversion: single GPU<->CPU round-trip for all experts
        # (replaces per-expert loop that did 2*num_experts GPU<->CPU transfers)
        act_scales_blocked, sf_offsets = ext.batch_convert_a_scales_for_w4a8(
            act_scales_rowmajor, expert_token_counts, expert_offsets, N, K_padded
        )

        # Build problem_sizes: [num_experts, 3] -> [[m_i, N, K_padded], ...]
        problem_sizes = torch.zeros((num_experts, 3), device=device, dtype=torch.int32)
        problem_sizes[:, 0] = expert_token_counts
        problem_sizes[:, 1] = N
        problem_sizes[:, 2] = K_padded

        alphas = torch.ones((num_experts,), device=device, dtype=torch.float32)

        torch.ops.cutlass.fp4_group_mm(
            output, activations, weights, act_scales_blocked, weight_scales,
            alphas, problem_sizes, expert_offsets, sf_offsets,
        )

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_indices: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through CUTLASS-backed MoE experts.

        Args:
            hidden_states: [batch, seq, hidden_size] or [num_tokens, hidden_size] in bfloat16
            router_indices: [num_tokens, top_k] int64 - selected expert indices
            routing_weights: [num_tokens, top_k] float - routing weights (softmax scores)

        Returns:
            Same shape as hidden_states in bfloat16
        """
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        top_k = router_indices.shape[1]

        # --- Step 1: Route tokens to experts ---
        token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, top_k)
        flat_token_idx = token_indices.reshape(-1)
        flat_expert_idx = router_indices.reshape(-1)
        flat_weights = routing_weights.reshape(-1)

        # Sort by expert index to group tokens per expert
        sorted_order = torch.argsort(flat_expert_idx, stable=True)
        sorted_expert_idx = flat_expert_idx[sorted_order]
        sorted_token_idx = flat_token_idx[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Compute per-expert token counts and offsets
        expert_token_counts = torch.zeros(self.num_experts, device=device, dtype=torch.int32)
        expert_token_counts.scatter_add_(
            0, sorted_expert_idx.to(torch.int64),
            torch.ones_like(sorted_expert_idx, dtype=torch.int32),
        )
        expert_offsets = torch.zeros(self.num_experts, device=device, dtype=torch.int32)
        torch.cumsum(expert_token_counts[:-1], dim=0, out=expert_offsets[1:])

        # Gather tokens in expert-sorted order
        gathered_hidden = hidden_states[sorted_token_idx]  # [T*top_k, hidden_size]

        # --- Step 2: Quantize activations BF16 -> FP8 ---
        act_fp8, act_scales = quantize_activations_to_fp8(gathered_hidden)

        # --- Step 3+4: CUTLASS grouped GEMM for gate_up_proj ---
        N_gate_up = 2 * self.intermediate_size
        K_hidden = self.hidden_size

        gate_up_output = self._run_cutlass_gemm(
            act_fp8, act_scales,
            self.gate_up_proj_data, self.gate_up_proj_scales,
            expert_token_counts, expert_offsets,
            N=N_gate_up, K=K_hidden, K_padded=self.K_hidden_padded,
        )

        # --- Step 5: Add bias + SwiGLU ---
        expert_bias_gate_up = self.gate_up_proj_bias[sorted_expert_idx]
        gate_up_output = gate_up_output + expert_bias_gate_up
        intermediate = swiglu(gate_up_output, self.alpha, self.limit)

        # --- Step 6: Quantize intermediate -> FP8 ---
        inter_fp8, inter_scales = quantize_activations_to_fp8(intermediate)

        # --- Step 7: CUTLASS grouped GEMM for down_proj ---
        N_hidden = self.hidden_size
        K_inter = self.intermediate_size

        down_output = self._run_cutlass_gemm(
            inter_fp8, inter_scales,
            self.down_proj_data, self.down_proj_scales,
            expert_token_counts, expert_offsets,
            N=N_hidden, K=K_inter, K_padded=self.K_inter_padded,
        )

        # --- Step 8: Add bias, scale by routing weights, scatter back ---
        expert_bias_down = self.down_proj_bias[sorted_expert_idx]
        down_output = down_output + expert_bias_down
        down_output = down_output * sorted_weights.unsqueeze(-1)

        result = torch.zeros((num_tokens, self.hidden_size), device=device, dtype=hidden_states.dtype)
        result.scatter_add_(
            0,
            sorted_token_idx.unsqueeze(-1).expand(-1, self.hidden_size),
            down_output.to(hidden_states.dtype),
        )

        return result.view(input_shape)


def load_cutlass_weights(
    module: CutlassGptOssExperts,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
    device: torch.device,
):
    """
    Load checkpoint weights into CutlassGptOssExperts, converting scales
    to CUTLASS blocked layout.

    Args:
        module: The CutlassGptOssExperts module to populate
        gate_up_blocks: [E, 2*intermediate, hidden//32, 16] uint8 from checkpoint
        gate_up_scales: [E, 2*intermediate, hidden//32] uint8 from checkpoint
        down_blocks: [E, hidden, intermediate//32, 16] uint8 from checkpoint
        down_scales: [E, hidden, intermediate//32] uint8 from checkpoint
        device: target device
    """
    ext = get_cutlass_extension()
    E = module.num_experts
    hidden = module.hidden_size
    intermediate = module.intermediate_size

    # Compute padded K dimensions (must be multiples of MMA_TILE_K=128)
    K_hidden_padded = pad_k_to_tile(hidden)
    K_inter_padded = pad_k_to_tile(intermediate)
    module.K_hidden_padded = K_hidden_padded
    module.K_inter_padded = K_inter_padded

    # Reshape blocks to [E, N, K//2] then pad K to tile boundary
    gate_up_data_raw = gate_up_blocks.reshape(E, 2 * intermediate, hidden // 2).to(device).contiguous()
    down_data_raw = down_blocks.reshape(E, hidden, intermediate // 2).to(device).contiguous()
    gate_up_scales_raw = gate_up_scales.reshape(E, 2 * intermediate, hidden // 32).to(device).contiguous()
    down_scales_raw = down_scales.reshape(E, hidden, intermediate // 32).to(device).contiguous()

    # Pad weight data and scales along K if needed
    if K_hidden_padded > hidden:
        pad_data = (K_hidden_padded - hidden) // 2
        pad_scales = (K_hidden_padded - hidden) // MXFP_BLOCK_SIZE
        gate_up_data_raw = torch.nn.functional.pad(gate_up_data_raw, (0, pad_data), value=0)
        gate_up_scales_raw = torch.nn.functional.pad(gate_up_scales_raw, (0, pad_scales), value=0)

    if K_inter_padded > intermediate:
        pad_data = (K_inter_padded - intermediate) // 2
        pad_scales = (K_inter_padded - intermediate) // MXFP_BLOCK_SIZE
        down_data_raw = torch.nn.functional.pad(down_data_raw, (0, pad_data), value=0)
        down_scales_raw = torch.nn.functional.pad(down_scales_raw, (0, pad_scales), value=0)

    # Convert scales from row-major to CUTLASS blocked layout
    # For weight scales (SFB), M is irrelevant (we use a dummy M=128)
    dummy_M = 128

    # gate_up_proj: N = 2*intermediate, K = K_hidden_padded
    gate_up_converted_list = []
    for e in range(E):
        converted = ext.convert_b_scales_for_w4a8(
            gate_up_scales_raw[e], dummy_M, 2 * intermediate, K_hidden_padded
        )
        gate_up_converted_list.append(converted)
    gate_up_scales_converted = torch.stack(gate_up_converted_list, dim=0)

    # down_proj: N = hidden, K = K_inter_padded
    down_converted_list = []
    for e in range(E):
        converted = ext.convert_b_scales_for_w4a8(
            down_scales_raw[e], dummy_M, hidden, K_inter_padded
        )
        down_converted_list.append(converted)
    down_scales_converted = torch.stack(down_converted_list, dim=0)

    module.gate_up_proj_data = gate_up_data_raw
    module.gate_up_proj_scales = gate_up_scales_converted
    module.down_proj_data = down_data_raw
    module.down_proj_scales = down_scales_converted
