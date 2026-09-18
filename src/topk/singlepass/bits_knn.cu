#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "bits/array_view.hpp"
#include "bits/cuch.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/topk/singlepass/bits_kernel.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/detail/definitions_common.hpp"

#ifdef TOPK_SINGLEPASS_USE_MINIMAL
#include "bits/topk/singlepass/detail/definitions_minimal.hpp"
#else
#ifndef TOPK_SINGLEPASS_USE_ALL
#define TOPK_SINGLEPASS_USE_ALL // to suppress further errors
#error "TOPK_SINGLEPASS_USE_ALL or TOPK_SINGLEPASS_USE_MINIMAL must be defined"
#endif
#include "bits/topk/singlepass/detail/definitions_all.hpp"
#endif

namespace
{

struct bits
{
    array_view<float, 2> in_dist;
    array_view<std::int32_t, 2> in_label;
    array_view<float, 2> out_dist;
    array_view<std::int32_t, 2> out_label;
    const std::int32_t* label_offsets;

    template <bool PREFETCH>
    void run(std::size_t block_size, std::size_t batch_size, std::size_t k)
    {
        if (!dynamic_switch<TOPK_SINGLEPASS_BITS_BLOCK_SIZES>(
                block_size, [=, this]<std::size_t BlockSize>() {
                    if (!dynamic_switch<TOPK_SINGLEPASS_BITS_BATCH_SIZES>(
                            batch_size, [=, this]<std::size_t BatchSize>() {
                                if (k <= 0 || !dynamic_switch_le<TOPK_SINGLEPASS_K_VALUES>(
                                                  k, [=, this]<std::size_t K>() {
                                                      run_bits_kernel<float, std::int32_t, PREFETCH,
                                                                      BlockSize, BatchSize, K>(
                                                          in_dist, in_label, out_dist, out_label, k,
                                                          label_offsets);
                                                  }))
                                {
                                    throw std::runtime_error("Unsupported k value: " +
                                                             std::to_string(k));
                                }
                            }))
                    {
                        throw std::runtime_error("Unsupported batch size: " +
                                                 std::to_string(batch_size));
                    }
                }))
        {
            throw std::runtime_error("Unsupported block size: " + std::to_string(block_size));
        }
    }
};

__global__ void populate_label_offsets_kernel(std::int32_t* label_offsets, std::size_t query_count,
                                              std::size_t parallel_count, std::size_t column_count)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= query_count * parallel_count)
        return;

    label_offsets[idx] = idx % parallel_count * column_count;
}

// Keep each logical row separate when splitting irregular lengths into equal partitions.
__global__ void pad_score_rows(array_view<float, 2> input, array_view<float, 2> padded)
{
    const auto count = padded.size(0) * padded.size(1);
    for (std::size_t idx = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x; idx < count;
         idx += std::size_t{blockDim.x} * gridDim.x)
    {
        const auto row = idx / padded.size(1);
        const auto col = idx % padded.size(1);
        padded(row, col) =
            col < input.size(1) ? input(row, col) : std::numeric_limits<float>::infinity();
    }
}

} // namespace

void bits_knn::selection()
{
    cuda_knn::selection();

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = out_dist_gpu(),
                .out_label = out_label_gpu(),
                .label_offsets = nullptr};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    constexpr bool PREFETCH = false;
    kernel.run<PREFETCH>(block_size, batch_size, k());

    cuda_stream::make_default().sync();
}

void bits_prefetch_knn::selection()
{
    cuda_knn::selection();

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = out_dist_gpu(),
                .out_label = out_label_gpu(),
                .label_offsets = nullptr};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    constexpr bool PREFETCH = true;
    kernel.run<PREFETCH>(block_size, batch_size, k());

    cuda_stream::make_default().sync();
}

void single_query_bits::initialize(const knn_args& args)
{
    if (args.deg == 0 || args.deg > args.point_count)
    {
        throw std::invalid_argument{"bits-sq requires 1 <= degree <= point_count"};
    }
    const auto partition_size = args.point_count / args.deg + (args.point_count % args.deg != 0);
    const auto max_index = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (partition_size > max_index / args.deg || args.query_count > max_index / args.deg ||
        args.k > max_index / args.deg ||
        args.query_count > max_elements / (partition_size * args.deg) ||
        (args.k != 0 && args.query_count > max_elements / args.deg / args.k))
    {
        throw std::invalid_argument{
            "bits-sq partition dimensions exceed supported index/allocation sizes"};
    }

    tmp_dist_.release();
    tmp_label_.release();
    label_offsets_.release();
    padded_dist_.release();
    cuda_knn::initialize(args);
    partition_size_ = partition_size;

    const auto input = in_dist_gpu();
    if (input.data() == nullptr || input.size(0) != query_count() ||
        input.size(1) != point_count() || input.stride(0) < point_count() || input.stride(1) != 1)
    {
        throw std::invalid_argument{
            "bits-sq requires a row-major score matrix matching the input shape"};
    }

    if (args_.deg > 1)
    {
        const auto rows = query_count() * args_.deg;
        tmp_dist_ = cuda_array<float, 2>{{rows, k()}};
        tmp_label_ = cuda_array<std::int32_t, 2>{{rows, k()}};
        label_offsets_ = cuda_array<std::int32_t, 1>{{rows}};

        populate_label_offsets_kernel<<<(rows + 255) / 256, 256>>>(
            label_offsets_.view().data(), query_count(), args_.deg, partition_size_);
        CUCH(cudaGetLastError());

        if (point_count() % args_.deg != 0 || input.stride(0) != point_count())
        {
            padded_dist_ = cuda_array<float, 2>{{query_count(), partition_size_ * args_.deg}};
        }
        cuda_stream::make_default().sync();
    }
}

void single_query_bits::selection()
{
    cuda_knn::selection();
    constexpr bool PREFETCH = true;

    auto input = in_dist_gpu();
    if (padded_dist_.view().data() != nullptr)
    {
        const auto count = padded_dist_.view().size();
        const auto blocks = std::min<std::size_t>((count + 255) / 256, 65535);
        pad_score_rows<<<blocks, 256>>>(input, padded_dist_.view());
        CUCH(cudaGetLastError());
        input = padded_dist_.view();
    }
    if (args_.deg > 1)
    {
        input = array_view<float, 2>{
            input.data(), {query_count() * args_.deg, partition_size_}, {partition_size_, 1}};
    }

    bits kernel{.in_dist = input,
                .in_label = {},
                .out_dist = args_.deg == 1 ? out_dist_gpu() : tmp_dist_.view(),
                .out_label = args_.deg == 1 ? out_label_gpu() : tmp_label_.view(),
                .label_offsets = label_offsets_.view().data()};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    kernel.run<PREFETCH>(block_size, batch_size, k());

    if (args_.deg > 1)
    {
        kernel.in_dist = array_view<float, 2>{
            kernel.out_dist.data(), {query_count(), k() * args_.deg}, {k() * args_.deg, 1}};
        kernel.in_label = array_view<std::int32_t, 2>{
            kernel.out_label.data(), {query_count(), k() * args_.deg}, {k() * args_.deg, 1}};
        kernel.label_offsets = nullptr;
        kernel.out_dist = out_dist_gpu();
        kernel.out_label = out_label_gpu();

        kernel.run<PREFETCH>(block_size, batch_size, k());
    }

    cuda_stream::make_default().sync();
}
