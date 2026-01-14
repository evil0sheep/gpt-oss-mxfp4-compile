/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CUTLASS_DEBUG_TRACE_LEVEL 1
#include <torch/extension.h>
#include <cutlass/arch/arch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "common.hpp"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include <cassert>

using namespace cute;

// =================================================================================================
// W4A4 IMPLEMENTATION (Original Restored)
// =================================================================================================

template <typename ElementAB, typename ElementC, typename ElementSF,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts_w4a4(
    ElementAB** a_offsets, ElementAB** b_offsets, ElementC** out_offsets,
    ElementSF** a_scales_offsets, ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int, ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int, const int32_t* expert_offsets,
    const int32_t* sf_offsets, const int32_t* problem_sizes_as_shapes,
    const int K, const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  // Originally int32_t but upcasting to int64_t to avoid overflow
  // during offset calculations
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  // size for block in block scale.
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  assert((m >= 0 && n == N && k == K && k % 2 == 0) &&
         "unexpected problem sizes");

  int64_t half_k = static_cast<int64_t>(k / 2);
  int64_t group_k = static_cast<int64_t>(k / group_size);
  // Shape of A as uint8/byte = [M, K // 2]
  // Shape of B as uint8/byte = [E, N, K // 2]
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;

  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  // Shape of C = [M, N]
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  // Shape of a_scale = [sum(sf_sizes), K // group_size]
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;

  assert((reinterpret_cast<uintptr_t>(a_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");

  // Shape of B scale = [E, N, K // group_size]
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  assert((reinterpret_cast<uintptr_t>(b_scales_offsets[expert_id]) % 128) ==
             0 &&
         "TMA requires 128-byte alignment");
  // Shape of alpha = [E]
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

#define __CALL_GET_STARTS_KERNEL_BLOCKSCALE_W4A4(ELEMENT_AB_TYPE, SF_TYPE,    \
                                            TENSOR_C_TYPE, C_TYPE, LayoutSFA, \
                                            LayoutSFB, ScaleConfig)           \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                            \
    __get_group_gemm_starts_w4a4<ELEMENT_AB_TYPE, C_TYPE, SF_TYPE, float,          \
                            LayoutSFA, LayoutSFB, ScaleConfig>                \
        <<<1, num_experts, 0, stream>>>(                                      \
            static_cast<ELEMENT_AB_TYPE**>(a_starts.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE**>(b_starts.data_ptr()),              \
            static_cast<C_TYPE**>(out_starts.data_ptr()),                     \
            static_cast<SF_TYPE**>(a_scales_starts.data_ptr()),               \
            static_cast<SF_TYPE**>(b_scales_starts.data_ptr()),               \
            static_cast<float**>(alpha_starts.data_ptr()),                    \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),              \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE*>(a_tensors.data_ptr()),              \
            static_cast<ELEMENT_AB_TYPE*>(b_tensors.data_ptr()),              \
            static_cast<C_TYPE*>(out_tensors.data_ptr()),                     \
            static_cast<SF_TYPE*>(a_scales.data_ptr()),                       \
            static_cast<SF_TYPE*>(b_scales.data_ptr()),                       \
            static_cast<float*>(alphas.data_ptr()),                           \
            static_cast<int32_t*>(expert_offsets.data_ptr()),                 \
            static_cast<int32_t*>(sf_offsets.data_ptr()),                     \
            static_cast<int32_t*>(problem_sizes.data_ptr()), K, N);           \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts_w4a4(
    const torch::Tensor& a_starts, const torch::Tensor& b_starts,
    const torch::Tensor& out_starts, const torch::Tensor& a_scales_starts,
    const torch::Tensor& b_scales_starts, const torch::Tensor& alpha_starts,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    /*these are used for their base addresses*/
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor const& out_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& alphas,
    torch::Tensor const& expert_offsets, torch::Tensor const& sf_offsets,
    torch::Tensor const& problem_sizes, int M, int N, int K) {
  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  TORCH_CHECK(out_tensors.size(1) == N,
              "Output tensor shape doesn't match expected shape");
  TORCH_CHECK(K / 2 == b_tensors.size(2),
              "b_tensors(dim = 2) and a_tensors(dim = 1) trailing"
              " dimension must match");
  if (false) {
  }
  //(ELEMENT_AB_TYPE, BS_TYPE, TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB,
  // ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE_W4A4(
      cutlass::float_e2m1_t, cutlass::float_ue4m3_t, torch::kBFloat16,
      cutlass::bfloat16_t, LayoutSFA, LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL_BLOCKSCALE_W4A4(cutlass::float_e2m1_t,
                                      cutlass::float_ue4m3_t, torch::kFloat16,
                                      half, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

void run_w4a4_original(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int M,
    int N, int K) {
  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  using ElementType = cutlass::float_e2m1_t;
  using ElementSFType = cutlass::float_ue4m3_t;
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

  // Hardcode output type to bfloat16 for SM120 logic
  using ElementC = cutlass::bfloat16_t;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  // Layout definitions
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Architecture definitions
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;

  using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutD*, AlignmentD,
          cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB,
          LayoutB*, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int);
  torch::Tensor c_strides1 =
      torch::full({num_experts}, output.stride(0), options_int);
  torch::Tensor a_strides1 =
      torch::full({num_experts}, a.stride(0) * 2, options_int);
  torch::Tensor b_strides1 =
      torch::full({num_experts}, b.stride(1) * 2, options_int);

  run_get_group_gemm_starts_w4a4<LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, alpha_ptrs,
      layout_sfa, layout_sfb, a, b, output, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes, M, N, K);

  // Create an instance of the GEMM
  Gemm gemm_op;

  // Initialize problem_sizes_as_shapes correctly
  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Set the Scheduler info
  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.get_device();
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);
  }
  hw_info.sm_count = min(cached_sm_counts[hw_info.device_id], INT_MAX);

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementType**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides1.data_ptr()),
      static_cast<const ElementType**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides1.data_ptr()),
      static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides1.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides1.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = 0.0f;

  // Gemm Arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  if (can_implement_status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::can_implement failed: " << cutlass::cutlassGetStatusString(can_implement_status) << std::endl;
  }
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM: status=", (int)can_implement_status);

  // Run the GEMM
  auto status = gemm_op.initialize(args, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    std::cout << "Workspace size: " << workspace_size << std::endl;
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to initialize GEMM: status=", (int)status,
              " workspace_size=", workspace_size, " num_experts=", num_experts,
              " M=", M, " N=", N, " K=", K);

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM SM120: ", cutlass::cutlassGetStatusString(status));
}

// =================================================================================================
// W4A8 IMPLEMENTATION (New)
// =================================================================================================

template <typename ScalarA, typename ScalarB, typename ElementC, 
          typename ScalarSFA, typename ScalarSFB,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts_w4a8(
    ScalarA** a_offsets, ScalarB** b_offsets, ElementC** out_offsets,
    ScalarSFA** a_scales_offsets, ScalarSFB** b_scales_offsets,
    ElementAccumulator** alpha_offsets, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, ScalarA* a_base,
    ScalarB* b_base, ElementC* out_base,
    ScalarSFA* a_scales_base, ScalarSFB* b_scales_base,
    ElementAccumulator* alphas_base, const int32_t* expert_offsets,
    const int32_t* sf_offsets, const int32_t* problem_sizes_as_shapes,
    const int K, const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  // size for block in block scale.
  int64_t group_size = 16;
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);
  
  int64_t group_k = static_cast<int64_t>(k / group_size);

  constexpr bool IsA4Bit = std::is_same_v<ScalarA, cutlass::float_e2m1_t>;
  constexpr bool IsB4Bit = std::is_same_v<ScalarB, cutlass::float_e2m1_t>;

  int64_t offset_a = IsA4Bit ? (k / 2) : k;
  int64_t offset_b = IsB4Bit ? (k / 2) : k;
  
  a_offsets[expert_id] = a_base + expert_offset * offset_a;
  b_offsets[expert_id] = b_base + expert_id * n * offset_b;

  out_offsets[expert_id] = out_base + expert_offset * n;
  
  // Scales are assumed to be 1 byte (ue4m3 or ue8m0)
  a_scales_offsets[expert_id] = a_scales_base + sf_offset * group_k;
  b_scales_offsets[expert_id] = b_scales_base + expert_id * n * group_k;
  alpha_offsets[expert_id] = alphas_base + expert_id;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
  *layout_sfb_ptr = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1));
}

#define CALL_GET_STARTS_KERNEL(ScalarA, ScalarB, SFA, SFB, CType, OutDtype, LayoutSFA, LayoutSFB, ScaleConfig) \
  if (out_tensors.dtype() == OutDtype) { \
    __get_group_gemm_starts_w4a8<ScalarA, ScalarB, CType, SFA, SFB, float, \
                            LayoutSFA, LayoutSFB, ScaleConfig> \
        <<<1, num_experts, 0, stream>>>( \
            static_cast<ScalarA**>(a_starts.data_ptr()), \
            static_cast<ScalarB**>(b_starts.data_ptr()), \
            static_cast<CType**>(out_starts.data_ptr()), \
            static_cast<SFA**>(a_scales_starts.data_ptr()), \
            static_cast<SFB**>(b_scales_starts.data_ptr()), \
            static_cast<float**>(alpha_starts.data_ptr()), \
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()), \
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()), \
            static_cast<ScalarA*>(a_tensors.data_ptr()), \
            static_cast<ScalarB*>(b_tensors.data_ptr()), \
            static_cast<CType*>(out_tensors.data_ptr()), \
            static_cast<SFA*>(a_scales.data_ptr()), \
            static_cast<SFB*>(b_scales.data_ptr()), \
            static_cast<float*>(alphas.data_ptr()), \
            static_cast<int32_t*>(expert_offsets.data_ptr()), \
            static_cast<int32_t*>(sf_offsets.data_ptr()), \
            static_cast<int32_t*>(problem_sizes.data_ptr()), K, N); \
  }

template <typename ElementA, typename ElementB, typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_group_gemm_starts_w4a8(
    const torch::Tensor& a_starts, const torch::Tensor& b_starts,
    const torch::Tensor& out_starts, const torch::Tensor& a_scales_starts,
    const torch::Tensor& b_scales_starts, const torch::Tensor& alpha_starts,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor const& out_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& alphas,
    torch::Tensor const& expert_offsets, torch::Tensor const& sf_offsets,
    torch::Tensor const& problem_sizes, int M, int N, int K) {
  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  using ScalarA = typename ElementA::DataType;
  using ScalarB = typename ElementB::DataType;
  using ScalarSFA = typename ElementA::ScaleFactorType;
  using ScalarSFB = typename ElementB::ScaleFactorType;

  // Dispatch based on output type (BF16 or FP16)
  CALL_GET_STARTS_KERNEL(ScalarA, ScalarB, ScalarSFA, ScalarSFB, cutlass::bfloat16_t, torch::kBFloat16, LayoutSFA, LayoutSFB, ScaleConfig)
  else CALL_GET_STARTS_KERNEL(ScalarA, ScalarB, ScalarSFA, ScalarSFB, cutlass::half_t, torch::kFloat16, LayoutSFA, LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }
}

template <typename ElementA, typename ElementB, typename ElementC>
void run_w4a8_group_mm_sm120(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets, int M,
    int N, int K) {
  
  using ScalarA = typename ElementA::DataType;
  using ScalarB = typename ElementB::DataType;
  using ElementSFA = typename ElementA::ScaleFactorType;
  using ElementSFB = typename ElementB::ScaleFactorType;

  constexpr bool IsA4Bit = std::is_same_v<ScalarA, cutlass::float_e2m1_t>;
  constexpr bool IsB4Bit = std::is_same_v<ScalarB, cutlass::float_e2m1_t>;

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
  
  using ElementD = ElementC;
  using ElementAccumulator = float;
  
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;

  // Alignment constraints
  static constexpr int AlignmentA = IsA4Bit ? 32 : 16;
  static constexpr int AlignmentB = IsB4Bit ? 32 : 16;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;

  using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementAccumulator, ElementC, ElementAccumulator>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutD*, AlignmentD,
          cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
          FusionOperation>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB,
          LayoutB*, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using ScaleConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = static_cast<int>(expert_offsets.size(0));
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor layout_sfa = torch::empty({num_experts, 5}, options_int);
  torch::Tensor layout_sfb = torch::empty({num_experts, 5}, options_int);
  torch::Tensor c_strides1 =
      torch::full({num_experts}, output.stride(0), options_int);
  
  // Stride calculations
  int64_t stride_a_multiplier = IsA4Bit ? 2 : 1;
  int64_t stride_b_multiplier = IsB4Bit ? 2 : 1;

  torch::Tensor a_strides1 =
      torch::full({num_experts}, a.stride(0) * stride_a_multiplier, options_int);
  torch::Tensor b_strides1 =
      torch::full({num_experts}, b.stride(1) * stride_b_multiplier, options_int);

  run_get_group_gemm_starts_w4a8<ElementA, ElementB, LayoutSFA, LayoutSFB, ScaleConfig>(
      a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, alpha_ptrs,
      layout_sfa, layout_sfb, a, b, output, a_blockscale, b_blockscales, alphas,
      expert_offsets, sf_offsets, problem_sizes, M, N, K);

  Gemm gemm_op;
  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  cutlass::KernelHardwareInfo hw_info;
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;
  hw_info.device_id = a.get_device();
  static std::unordered_map<int, int> cached_sm_counts;
  if (cached_sm_counts.find(hw_info.device_id) == cached_sm_counts.end()) {
    cached_sm_counts[hw_info.device_id] =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);
  }
  hw_info.sm_count = min(cached_sm_counts[hw_info.device_id], INT_MAX);

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ScalarA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides1.data_ptr()),
      static_cast<const ScalarB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides1.data_ptr()),
      static_cast<const ElementSFA**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
      static_cast<const ElementSFB**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())};

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(c_strides1.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides1.data_ptr())};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha_ptr_array =
      reinterpret_cast<float**>(alpha_ptrs.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  fusion_args.beta = 0.0f;

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info,
      scheduler};

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  if (can_implement_status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::can_implement failed: " << cutlass::cutlassGetStatusString(can_implement_status) << std::endl;
  }
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM: status=", (int)can_implement_status);

  auto status = gemm_op.initialize(args, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    std::cout << "Gemm::initialize failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
    std::cout << "Workspace size: " << workspace_size << std::endl;
  }
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "Failed to initialize GEMM: status=", (int)status);

  status = gemm_op.run(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM SM120: ", cutlass::cutlassGetStatusString(status));
}

// =================================================================================================
// MAIN DISPATCHER
// =================================================================================================

#if (defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100) || \
    (defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120)
constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;
#endif

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous.")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

void cutlass_fp4_group_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
#if (defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100) || \
    (defined ENABLE_NVFP4_SM120 && ENABLE_NVFP4_SM120)
  
  // Input validation
  CHECK_TH_CUDA(a, "a");
  CHECK_CONTIGUOUS(a, "a");
  TORCH_CHECK(a.scalar_type() == FLOAT4_E2M1X2 || a.scalar_type() == at::ScalarType::Float8_e4m3fn,
    "a must be Byte (packed FP4) or Float8_e4m3fn");
  
  CHECK_INPUT(b, FLOAT4_E2M1X2, "b");
  
  CHECK_TH_CUDA(a_blockscale, "a_blockscale");
  CHECK_CONTIGUOUS(a_blockscale, "a_blockscale");
  TORCH_CHECK(a_blockscale.scalar_type() == SF_DTYPE || a_blockscale.scalar_type() == at::ScalarType::Byte,
    "a_blockscale must be Float8_e4m3fn or Byte (UE8M0)");

  CHECK_TH_CUDA(b_blockscales, "b_blockscales");
  CHECK_CONTIGUOUS(b_blockscales, "b_blockscales");
  TORCH_CHECK(b_blockscales.scalar_type() == SF_DTYPE || b_blockscales.scalar_type() == at::ScalarType::Byte,
    "b_blockscales must be Float8_e4m3fn or Byte (UE8M0)");
  CHECK_INPUT(alphas, at::ScalarType::Float, "alphas");

  TORCH_CHECK(a_blockscale.dim() == 2, "Invalid a_blockscale dim");
  TORCH_CHECK(b_blockscales.dim() == 3, "Invalid b_blockscales dim");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D");
  
  int M = static_cast<int>(a.size(0));
  int N = static_cast<int>(b.size(1));
  int K = static_cast<int>(2 * b.size(2)); // B is always packed FP4, so K = dim2 * 2

  // Check Architecture
  int32_t version_num = get_sm_version_num();
  TORCH_CHECK(version_num >= 120, "Required SM120 for NVFP4/MXFP8");

  if (a.scalar_type() == FLOAT4_E2M1X2) {
      // W4A4 Mode (Pure NVFP4) -> Use Original Restored Function
      run_w4a4_original(
            output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
            expert_offsets, sf_offsets, M, N, K);
  } else {
      // W4A8 Mode (MXFP8 x MXFP4) -> Use New Function
      using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
      using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;

      if (output.scalar_type() == torch::kBFloat16) {
          run_w4a8_group_mm_sm120<ElementA, ElementB, cutlass::bfloat16_t>(
            output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
            expert_offsets, sf_offsets, M, N, K);
      } else {
          run_w4a8_group_mm_sm120<ElementA, ElementB, cutlass::half_t>(
            output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes,
            expert_offsets, sf_offsets, M, N, K);
      }
  }
#else
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled cutlass_fp4_group_mm kernel.");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_fp4_group_mm", &cutlass_fp4_group_mm,
        "CUTLASS FP4 Grouped GEMM");
}