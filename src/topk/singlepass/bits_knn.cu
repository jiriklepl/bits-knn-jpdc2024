#include <cassert>
#include <cstddef>
#include <cstdint>
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
                block_size, [=, *this]<std::size_t BlockSize>() {
                    if (!dynamic_switch<TOPK_SINGLEPASS_BITS_BATCH_SIZES>(
                            batch_size, [=, *this]<std::size_t BatchSize>() {
                                if (k <= 0 || !dynamic_switch_le<TOPK_SINGLEPASS_K_VALUES>(
                                                  k, [=, *this]<std::size_t K>() {
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
    cuda_knn::initialize(args);

    if (args_.deg > 1)
    {
        tmp_dist_ = cuda_array<float, 2>{{args_.deg, k()}};
        tmp_label_ = cuda_array<std::int32_t, 2>{{args_.deg, k()}};
        label_offsets_ = cuda_array<std::int32_t, 1>{{args_.deg}};

        populate_label_offsets_kernel<<<(args_.deg + 255) / 256, 256>>>(
            label_offsets_.view().data(), 1, args_.deg, k());
        CUCH(cudaGetLastError());
    }
}

void single_query_bits::selection()
{
    cuda_knn::selection();

    constexpr bool PREFETCH = true;

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = args_.deg == 1 ? out_dist_gpu() : tmp_dist_.view(),
                .out_label = args_.deg == 1 ? out_label_gpu() : tmp_label_.view(),
                .label_offsets = nullptr};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    const std::size_t column_count = (kernel.in_dist.size(1) + args_.deg - 1) / args_.deg;
    const std::size_t row_count = kernel.in_dist.size(0) * args_.deg;

    if (args_.deg > 1 && (tmp_dist_.view().size(0) < row_count || tmp_dist_.view().size(1) < k() ||
                          kernel.label_offsets == nullptr))
    {
        tmp_dist_ = cuda_array<float, 2>{{row_count, k()}};
        tmp_label_ = cuda_array<std::int32_t, 2>{{row_count, k()}};
        label_offsets_ = cuda_array<std::int32_t, 1>{{row_count}};

        kernel.out_dist = tmp_dist_.view();
        kernel.out_label = tmp_label_.view();
        kernel.label_offsets = label_offsets_.view().data();

        populate_label_offsets_kernel<<<(row_count + 255) / 256, 256>>>(
            label_offsets_.view().data(), kernel.in_dist.size(0), args_.deg, column_count);
        CUCH(cudaGetLastError());
    }

    kernel.in_dist =
        array_view<float, 2>{kernel.in_dist.data(), {row_count, column_count}, {column_count, 1}};

    kernel.run<PREFETCH>(block_size, batch_size, k());

    if (args_.deg > 1)
    {
        kernel.in_dist = array_view<float, 2>{kernel.out_dist.data(),
                                              {out_dist_gpu().size(0), k() * args_.deg},
                                              {k() * args_.deg, 1}};
        kernel.in_label = array_view<std::int32_t, 2>{kernel.out_label.data(),
                                                      {out_label_gpu().size(0), k() * args_.deg},
                                                      {k() * args_.deg, 1}};
        kernel.label_offsets = nullptr;
        kernel.out_dist = out_dist_gpu();
        kernel.out_label = out_label_gpu();

        kernel.run<PREFETCH>(block_size, batch_size, k());
    }

    cuda_stream::make_default().sync();
}
