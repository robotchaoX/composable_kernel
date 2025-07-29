// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bias_bnorm_fwd.hpp.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for
 * \link DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the
 * computing of pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename ComputePtrOffset,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename DsGridDesc_M_N,
          typename EGridDesc_M_N,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_fwd_xdl_cshuffle_v3(typename GridwiseGemm::Argument karg,
                                                const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
                                                const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
                                                const DsGridDesc_M_N ds_grid_desc_m_n,
                                                const EGridDesc_M_N c_grid_desc_m_n,
                                                const ComputePtrOffset compute_ptr_offset_of_groups,
                                                const ComputePtrOffset compute_ptr_offset_of_n)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // offset base pointer for each work-group
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);

    const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);

    static constexpr index_t NumDTensor = GridwiseGemm::NumDTensor;
    using DsGridPointer                 = typename GridwiseGemm::DsGridPointer;
    DsGridPointer p_ds_grid_grp{};

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_group_offset[i]; });

    const long_index_t a_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
    const long_index_t b_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
    const long_index_t e_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    using Block2CTileMap         = typename GridwiseGemm::Block2CTileMapDefault;
    const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

    GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        karg.p_a_grid + a_group_offset + a_n_offset,
        karg.p_b_grid + b_group_offset,
        p_ds_grid_grp,
        karg.p_c_grid + e_group_offset + e_n_offset,
        p_shared,
        karg,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op,
        block_2_ctile_map,
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        ds_grid_desc_m_n,
        c_grid_desc_m_n);
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_m_n;
    ignore = c_grid_desc_m_n;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          typename ComputePtrOffset,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename DsGridDesc_M_N,
          typename EGridDesc_M_N,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds(
            typename GridwiseGemm::Argument karg,
            const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
            const DsGridDesc_M_N ds_grid_desc_m_n,
            const EGridDesc_M_N c_grid_desc_m_n,
            const ComputePtrOffset compute_ptr_offset_of_groups,
            const ComputePtrOffset compute_ptr_offset_of_n)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // offset base pointer for each work-group
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);

    const auto& ds_group_offset = compute_ptr_offset_of_groups.GetDsPtrOffset(g_idx);

    static constexpr index_t NumDTensor = GridwiseGemm::NumDTensor;
    using DsGridPointer                 = typename GridwiseGemm::DsGridPointer;
    DsGridPointer p_ds_grid_grp{};

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = karg.p_ds_grid[i] + ds_group_offset[i]; });

    const long_index_t a_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetAPtrOffset(g_idx));
    const long_index_t b_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetBPtrOffset(g_idx));
    const long_index_t e_group_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_groups.GetEPtrOffset(g_idx));

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    // Pass two lds pointer is the key to tell compiler that ds_read/write
    // operate on different lds chunk at same time without order dependecy
    __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte()];
    __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    using Block2CTileMap         = typename GridwiseGemm::Block2CTileMapDefault;
    const auto block_2_ctile_map = Block2CTileMap{karg.M, karg.N, 4};

    GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
        karg.p_a_grid + a_group_offset + a_n_offset,
        karg.p_b_grid + b_group_offset,
        p_ds_grid_grp,
        karg.p_c_grid + e_group_offset + e_n_offset,
        p_shared_0,
        p_shared_1,
        karg,
        karg.a_element_op,
        karg.b_element_op,
        karg.c_element_op,
        block_2_ctile_map,
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        ds_grid_desc_m_n,
        c_grid_desc_m_n);
#else
    ignore = karg;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_m_n;
    ignore = c_grid_desc_m_n;
    ignore = compute_ptr_offset_of_groups;
    ignore = compute_ptr_offset_of_n;
#endif // end of if (defined(__gfx9__))
}

} // namespace

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Forward convolution with up to 3 spatial dimentions
//  @li         Input tensor in GNWC data format
//  @li         Weight tensor in GKXC data format
//  @li         Output tensor in GNWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename BiasDataType,
          typename EDataType,
          typename BnScaleDataType,
          typename BnBiasDataType,
          typename BnMeanVarDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          bool ConvBias,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1>
struct DeviceGroupedConvBiasBnormFwd_Xdl_CShuffle_V3
    : public DeviceGroupedConvBiasBnormFwd<NDimSpatial,
          ALayout,
          BLayout,
          ELayout,
          ADataType,
          BDataType,
          BiasDataType,
          EDataType,
          BnScaleDataType,
          BnBiasDataType,
          BnMeanVarDataType,
          AElementwiseOperation,
          BElementwiseOperation,
          CElementwiseOperation,
          ConvBias>
{
    using DeviceOp = DeviceGroupedConvBiasBnormFwd_Xdl_CShuffle_V3;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // Generate vector size for C & Ds
    using CDEBlockTransferScalarPerVectors =
        typename uniform_sequence_gen<NumDTensor + 1,
                                      CDEBlockTransferScalarPerVector_NPerBlock>::type;

    using ConvToGemmFwdTransformer = TransformConvFwdToGemm<NDimSpatial,
                                                            ConvForwardSpecialization,
                                                            false /*SplitN*/,
                                                            ADataType,
                                                            EDataType>;

    using ComputePtrOffset = ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    template <typename ALay>
    static auto
    MakeAGridDescriptor_AK0_M_AK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
            is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
            ctc::NHWGC,
            std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
                               ctc::NDHWGC,
                               ALay>>;

        const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<Layout>();

        const auto in_gemmm_gemmk_desc =
            matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_desc);

        const auto M = in_gemmm_gemmk_desc.GetLength(I0);
        const auto K = in_gemmm_gemmk_desc.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(in_gemmm_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename BLay>
    static auto
    MakeBGridDescriptor_BK0_N_BK1(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
            is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
            ctc::GKYXC,
            std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
                               ctc::GKZYXC,
                               BLay>>;

        const auto wei_gemmnraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeBDescriptor_N_K<Layout>();

        const auto wei_gemmn_gemmk_desc =
            matrix_padder.PadBDescriptor_N_K(wei_gemmnraw_gemmkraw_desc);

        const auto N = wei_gemmn_gemmk_desc.GetLength(I0);
        const auto K = wei_gemmn_gemmk_desc.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(wei_gemmn_gemmk_desc,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    template <typename ELay>
    static auto MakeEGridDescriptor_M_N(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)

    {
        namespace ctc = tensor_layout::convolution;
        using Layout  = std::conditional_t<
            is_NGCHW_GKCYX_NGKHW<ALayout, BLayout, ELayout>(),
            ctc::NHWGK,
            std::conditional_t<is_NGCDHW_GKCZYX_NGKDHW<ALayout, BLayout, ELayout>(),
                               ctc::NDHWGK,
                               ELay>>;

        const auto out_gemmmraw_gemmnraw_desc =
            conv_to_gemm_transformer.template MakeCDescriptor_M_N<Layout>();

        const auto out_gemmm_gemmn_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmnraw_desc);

        return out_gemmm_gemmn_desc;
    }

    // Shape of Ds and E must be aligned. Strides can be different.
    // Pass e_g_n_k_wos_lengths for logical broadcast.
    static auto MakeDsGridDescriptor_M_N(const ConvToGemmFwdTransformer& conv_to_gemm_transformer)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer);
            },
            Number<NumDTensor>{});
    }

    // desc for problem definition
    constexpr static ConvToGemmFwdTransformer dummy_conv_to_gemm_transformer;
    using EGridDesc_M_N =
        remove_cvref_t<decltype(MakeEGridDescriptor_M_N<ELayout>(dummy_conv_to_gemm_transformer))>;
    using DsGridDesc_M_N =
        remove_cvref_t<decltype(MakeDsGridDescriptor_M_N(dummy_conv_to_gemm_transformer))>;

    // Use appropriate gridwise gemm
    using GridwiseGemm = GridwiseGemmMultiD_xdl_cshuffle_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        DsLayout,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        AComputeDataType,
        BComputeDataType,
        ADataType,
        BDataType,
        DoElementwiseBeforeCShuffle>;

    // desc for blockwise copy
    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(MakeAGridDescriptor_AK0_M_AK1<ALayout>(
        dummy_conv_to_gemm_transformer))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(MakeBGridDescriptor_BK0_N_BK1<BLayout>(
        dummy_conv_to_gemm_transformer))>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,
                const void* p_b,
                const void* p_bias,
                void* p_e,
                const void* p_bnScale,
                const void* p_bnBias,
                void* p_resultSaveMean,
                void* p_resultSaveInvVariance,
                void* p_resultRunningMean,
                void* p_resultRunningVariance
                double epsilon,
                double exponentialAverageFactor,
                const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
                const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
                const std::array<index_t, NDimSpatial + 3>& conv_bias_g_n_k_wos_strides,
                const std::array<index_t, NDimSpatial + 3>& bn_scale_g_n_k_wos_strides,
                const std::array<index_t, NDimSpatial + 3>& bn_bias_g_n_k_wos_strides,
                const std::array<index_t, NDimSpatial + 3>& bn_mean_var_g_n_k_wos_strides,
                const std::array<index_t, NDimSpatial>& conv_filter_strides,
                const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                const std::array<index_t, NDimSpatial>& input_left_pads,
                const std::array<index_t, NDimSpatial>& input_right_pads,
                const AElementwiseOperation& a_element_op,
                const BElementwiseOperation& b_element_op,
                const CElementwiseOperation& c_element_op)
            : p_a_grid_{},
              p_b_grid_{},
              p_ds_grid_{p_ds},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              a_g_n_c_wis_lengths_{a_g_n_c_wis_lengths},
              a_g_n_c_wis_strides_{conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                  a_g_n_c_wis_lengths, a_g_n_c_wis_strides)},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{conv_ngchw_to_nhwgc_transformer.TransposeWeiStrides(
                  b_g_k_c_xs_lengths, b_g_k_c_xs_strides)},
              ds_g_n_k_wos_lengths_{ds_g_n_k_wos_lengths},
              ds_g_n_k_wos_strides_{ds_g_n_k_wos_strides},
              e_g_n_k_wos_lengths_{e_g_n_k_wos_lengths},
              e_g_n_k_wos_strides_{conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(
                  e_g_n_k_wos_lengths, e_g_n_k_wos_strides)},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              num_group_{a_g_n_c_wis_lengths_[0]},
              conv_to_gemm_transformer_{a_g_n_c_wis_lengths_,
                                        a_g_n_c_wis_strides_,
                                        b_g_k_c_xs_lengths_,
                                        b_g_k_c_xs_strides_,
                                        e_g_n_k_wos_lengths_,
                                        e_g_n_k_wos_strides_,
                                        conv_filter_strides_,
                                        conv_filter_dilations_,
                                        input_left_pads_,
                                        input_right_pads_},
              ds_grid_desc_m_n_{},
              e_grid_desc_m_n_{
                  DeviceOp::MakeEGridDescriptor_M_N<ELayout>(conv_to_gemm_transformer_)},
              a_grid_desc_ak0_m_ak1_{
                  MakeAGridDescriptor_AK0_M_AK1<ALayout>(conv_to_gemm_transformer_)},
              b_grid_desc_bk0_n_bk1_{
                  MakeBGridDescriptor_BK0_N_BK1<BLayout>(conv_to_gemm_transformer_)},
              compute_ptr_offset_of_groups_{},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            // A/B/E Batch/N Stride
            compute_ptr_offset_of_groups_.BatchStrideA_ = a_g_n_c_wis_strides_[0];
            compute_ptr_offset_of_groups_.BatchStrideB_ = b_g_k_c_xs_strides_[0];

            // p_as and p_bs are pointers
            p_a_grid_ = static_cast<const ADataType*>(p_as);
            p_b_grid_ = static_cast<const BDataType*>(p_bs);

            // populate pointer, batch stride, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                // D batch stride
                compute_ptr_offset_of_groups_.BatchStrideDs_(i) = ds_g_n_k_wos_strides_[i][0];

                ConvToGemmFwdTransformer conv_to_gemm_transformer_d{a_g_n_c_wis_lengths_,
                                                                    a_g_n_c_wis_strides_,
                                                                    b_g_k_c_xs_lengths_,
                                                                    b_g_k_c_xs_strides_,
                                                                    e_g_n_k_wos_lengths_,
                                                                    ds_g_n_k_wos_strides_[i],
                                                                    conv_filter_strides_,
                                                                    conv_filter_dilations_,
                                                                    input_left_pads_,
                                                                    input_right_pads_};

                // D desc
                ds_grid_desc_m_n_(i) =
                    DeviceOp::MakeEGridDescriptor_M_N<DLayout>(conv_to_gemm_transformer_d);
            });

            compute_ptr_offset_of_groups_.BatchStrideE_ = e_g_n_k_wos_strides_[0];
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return 0;
        }

        void Print() const
        {
            std::cout << "A[AK0, M, AK1]: " << a_grid_desc_ak0_m_ak1_ << std::endl;
            std::cout << "B[BK0, N, BK1]: " << b_grid_desc_bk0_n_bk1_ << std::endl;
            static_for<0, NumDTensor, 1>{}(
                [&](auto i) { std::cout << "Ds[M, N]: " << ds_grid_desc_m_n_[i] << std::endl; });
            std::cout << "E[M, N]: " << e_grid_desc_m_n_ << std::endl;
        }

        //  private:
        // pointers (tuple if multi AB, pointer if no)
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        const BiasDataType* p_bias_;
        const BnScaleDataType* p_bnScale_;
        const BnBiasDataType* p_bnBias_;
        BnMeanVarDataType* p_resultSaveMean_;
        BnMeanVarDataType* p_resultSaveInvVariance_;
        BnMeanVarDataType* p_resultRunningMean_;
        BnMeanVarDataType* p_resultRunningVariance_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;

        index_t num_group_;

        ConvToGemmFwdTransformer conv_to_gemm_transformer_;

        // tensor descriptors for block/thread-wise copy
        DsGridDesc_M_N ds_grid_desc_m_n_;
        EGridDesc_M_N e_grid_desc_m_n_;

        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;

        // for computing batch offset
        ComputePtrOffset compute_ptr_offset_of_groups_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            float ave_time = 0;

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
            const index_t GemmK =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            index_t gdx, gdy, gdz;
            // TODO: Do we want to support kbatch ??
            std::tie(gdx, gdy, gdz) =
                GridwiseGemm::CalculateGridSize(GemmM, GemmN, I1 /*arg.KBatch*/);

            gdy = arg.num_group_;
            gdz = 1;

            index_t K_split                  = (GemmK + KPerBlock - 1) / KPerBlock * KPerBlock;
            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            typename GridwiseGemm::Argument gemm_arg{
                arg.p_a_grid_,
                arg.p_b_grid_,
                arg.p_ds_grid_,
                arg.p_e_grid_,
                GemmM,
                GemmN,
                GemmK,
                // No need to set strides, we pass descs to kernel
                I0,
                I0,
                {},
                I0,
                I1, // kbatch
                arg.a_element_op_,
                arg.b_element_op_,
                arg.c_element_op_};

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemm::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemm::Argument> rotating_mem(
                        gemm_arg_,
                        stream_config.rotating_count,
                        gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                        gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                    };

                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_ak0_m_ak1_,
                        arg.b_grid_desc_bk0_n_bk1_,
                        arg.ds_grid_desc_m_n_,
                        arg.e_grid_desc_m_n_,
                        arg.compute_ptr_offset_of_groups_);
                }
                else
                {
                    ave_time += launch_and_time_kernel(stream_config,
                                                       kernel,
                                                       dim3(gdx, gdy, gdz),
                                                       dim3(BlockSize),
                                                       0,
                                                       gemm_arg,
                                                       arg.a_grid_desc_ak0_m_ak1_,
                                                       arg.b_grid_desc_bk0_n_bk1_,
                                                       arg.ds_grid_desc_m_n_,
                                                       arg.e_grid_desc_m_n_,
                                                       arg.compute_ptr_offset_of_groups_);
                }
            };

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    const auto kernel =
                        kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                ComputePtrOffset,
                                                                DeviceOp::AGridDesc_AK0_M_AK1,
                                                                DeviceOp::BGridDesc_BK0_N_BK1,
                                                                DeviceOp::DsGridDesc_M_N,
                                                                DeviceOp::EGridDesc_M_N,
                                                                true,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                    Run(kernel);
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::One)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::One>;
                        Run(kernel);
                    }
                    else if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Full)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Full>;
                        Run(kernel);
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 2)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Two)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Two>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 3)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Three)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Three>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 4)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Four)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Four>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 5)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Five)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Five>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 6)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Six)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Six>;
                            Run(kernel);
                        }
                    }

                    if constexpr(GridwiseGemm::BlockwiseGemmPipe::PrefetchStages > 7)
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Seven)
                        {
                            const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3<
                                GridwiseGemm,
                                ComputePtrOffset,
                                DeviceOp::AGridDesc_AK0_M_AK1,
                                DeviceOp::BGridDesc_BK0_N_BK1,
                                DeviceOp::DsGridDesc_M_N,
                                DeviceOp::EGridDesc_M_N,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Seven>;
                            Run(kernel);
                        }
                    }
                }
                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            ComputePtrOffset,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::DsGridDesc_M_N,
                            DeviceOp::EGridDesc_M_N,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_fwd_xdl_cshuffle_v3_2lds<
                            GridwiseGemm,
                            ComputePtrOffset,
                            DeviceOp::AGridDesc_AK0_M_AK1,
                            DeviceOp::BGridDesc_BK0_N_BK1,
                            DeviceOp::DsGridDesc_M_N,
                            DeviceOp::EGridDesc_M_N,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy,
                            TailNumber::Even>;
                        Run(kernel);
                    }
                }
                else
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Odd>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel =
                            kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                    ComputePtrOffset,
                                                                    DeviceOp::AGridDesc_AK0_M_AK1,
                                                                    DeviceOp::BGridDesc_BK0_N_BK1,
                                                                    DeviceOp::DsGridDesc_M_N,
                                                                    DeviceOp::EGridDesc_M_N,
                                                                    true,
                                                                    InMemoryDataOperationEnum::Set,
                                                                    minimum_occupancy,
                                                                    TailNumber::Even>;
                        Run(kernel);
                    }
                }
            }
            // has_main_k_block_loop
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    const auto kernel =
                        kernel_grouped_conv_fwd_xdl_cshuffle_v3<GridwiseGemm,
                                                                ComputePtrOffset,
                                                                DeviceOp::AGridDesc_AK0_M_AK1,
                                                                DeviceOp::BGridDesc_BK0_N_BK1,
                                                                DeviceOp::DsGridDesc_M_N,
                                                                DeviceOp::EGridDesc_M_N,
                                                                false,
                                                                InMemoryDataOperationEnum::Set,
                                                                minimum_occupancy>;
                    Run(kernel);
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.f;

            avg_time += RunGemm(arg, stream_config);

            return avg_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        namespace ctc = tensor_layout::convolution;

        const index_t G = arg.b_g_k_c_xs_lengths_[I0];
        const index_t K = arg.b_g_k_c_xs_lengths_[I1];
        const index_t C = arg.b_g_k_c_xs_lengths_[I2];

        // check device
        if(get_device_name() == "gfx908")
        {
            // FIXME: re-enable fp64 when SWDEV-335738 is fixed
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, int32_t>))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout
                        << "On gfx908 the accumulation data type must be one of fp32 or int32!"
                        << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                        << std::endl;
                }
                return false;
            }
        }

        if(!ck::is_xdl_supported())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Current device does not support xdl instructions!"
                          << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // check ConvolutionForwardSpecialization
        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t ConvStride = arg.conv_filter_strides_[i];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(SpatialDim == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "The input paramters do not align with specialization "
                                     "Filter1x1Stride1Pad0!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = arg.b_g_k_c_xs_lengths_[i + 3];
                const index_t LeftPad    = arg.input_left_pads_[i];
                const index_t RightPad   = arg.input_right_pads_[i];

                if(!(SpatialDim == 1 && LeftPad == 0 && RightPad == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout
                            << "The input paramters do not align with specialization Filter1x1Pad0!"
                            << " In " << __FILE__ << ":" << __LINE__
                            << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
            }
        }

        // check vector access of A
        // FIXME: layout
        if constexpr(is_same_v<ALayout, ctc::NWGC> || is_same_v<ALayout, ctc::NHWGC> ||
                     is_same_v<ALayout, ctc::NDHWGC>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && C % ABlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[A Layout] The number of input channels is not a multiple of "
                                 "ABlockTransferSrcScalarPerVector!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported A Layout!"
                          << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(is_same_v<BLayout, ctc::GKXC> ||
                     is_same_v<BLayout, ctc::GKYXC> || is_same_v<BLayout, ctc::GKZYXC>)

        {
            if(!(BBlockTransferSrcVectorDim == 2 && C % BBlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[B Layout] The number of input channels is not a multiple of "
                                 "BBlockTransferSrcScalarPerVector!"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported A Layout!"
                          << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // check vector access of E
        if constexpr(is_same_v<ELayout, ctc::NWGK> || is_same_v<ELayout, ctc::NHWGK> ||
                     is_same_v<ELayout, ctc::NDHWGK>)
        {
            if(!(K % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "[E Layout] The K is not a multiple of "
                                 "CDEBlockTransferScalarPerVector_NPerBlock"
                              << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported E Layout!"
                          << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // Gridwise gemm v3 doesn't verify descriptors size
        if(!arg.conv_to_gemm_transformer_.AreDescriptorsSmallerThan2GB())
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout
                    << "[conv_to_gemm_transformer_] One of the descriptors is bigger than 2GB!"
                    << " In " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                    << std::endl;
            }
            return false;
        }

        // check Gridwise GEMM
        const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_.GetLength(I1);
        const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_.GetLength(I1);
        const index_t GemmK =
            arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

        typename GridwiseGemm::Argument gemm_arg{nullptr,
                                                 nullptr,
                                                 {},
                                                 nullptr,
                                                 GemmM,
                                                 GemmN,
                                                 GemmK,
                                                 I0,
                                                 I0,
                                                 {},
                                                 I0,
                                                 I1 /*KBatch*/,
                                                 arg.a_element_op_,
                                                 arg.b_element_op_,
                                                 arg.c_element_op_};

        return GridwiseGemm::CheckValidity(gemm_arg);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const void* p_a,
        const void* p_b,
        const void* p_bias,
        void* p_e,
        const void* p_bnScale,
        const void* p_bnBias,
        void* p_resultSaveMean,
        void* p_resultSaveInvVariance,
        void* p_resultRunningMean,
        void* p_resultRunningVariance
        double epsilon,
        double exponentialAverageFactor,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& conv_bias_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_scale_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_bias_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_mean_var_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op)
    {
        return Argument{p_a,
        p_b,
        p_bias,
        p_e,
        p_bnScale,
        p_bnBias,
        p_resultSaveMean,
        p_resultSaveInvVariance,
        p_resultRunningMean,
        p_resultRunningVariance
         epsilon,
         exponentialAverageFactor,
         a_g_n_c_wis_lengths,
         a_g_n_c_wis_strides,
         b_g_k_c_xs_lengths,
         b_g_k_c_xs_strides,
         e_g_n_k_wos_lengths,
         e_g_n_k_wos_strides,
         conv_bias_g_n_k_wos_strides,
         bn_scale_g_n_k_wos_strides,
         bn_bias_g_n_k_wos_strides,
         bn_mean_var_g_n_k_wos_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        a_element_op,
        b_element_op,
        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,
        const void* p_b,
        const void* p_bias,
        void* p_e,
        const void* p_bnScale,
        const void* p_bnBias,
        void* p_resultSaveMean,
        void* p_resultSaveInvVariance,
        void* p_resultRunningMean,
        void* p_resultRunningVariance
        double epsilon,
        double exponentialAverageFactor,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& e_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& conv_bias_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_scale_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_bias_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial + 3>& bn_mean_var_g_n_k_wos_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op) override
    {
        return std::make_unique<Argument>(p_a,
        p_b,
        p_bias,
        p_e,
        p_bnScale,
        p_bnBias,
        p_resultSaveMean,
        p_resultSaveInvVariance,
        p_resultRunningMean,
        p_resultRunningVariance
         epsilon,
         exponentialAverageFactor,
         a_g_n_c_wis_lengths,
         a_g_n_c_wis_strides,
         b_g_k_c_xs_lengths,
         b_g_k_c_xs_strides,
         e_g_n_k_wos_lengths,
         e_g_n_k_wos_strides,
         conv_bias_g_n_k_wos_strides,
         bn_scale_g_n_k_wos_strides,
         bn_bias_g_n_k_wos_strides,
         bn_mean_var_g_n_k_wos_strides,
        conv_filter_strides,
        conv_filter_dilations,
        input_left_pads,
        input_right_pads,
        a_element_op,
        b_element_op,
        cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGroupedConvBiasBnormFwd_Xdl_CShuffle_V3"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CDEBlockTransferScalarPerVector_NPerBlock << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer]
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
