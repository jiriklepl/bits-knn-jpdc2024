#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bits/array_view.hpp"
#include "bits/cuda_array.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"

#include "bits/topk/singlepass/bits_kernel.hpp"
#include "bits/topk/singlepass/bits_knn.hpp"

namespace
{

struct bits
{
    array_view<float, 2> in_dist;
    array_view<std::int32_t, 2> in_label;
    array_view<float, 2> out_dist;
    array_view<std::int32_t, 2> out_label;
    const std::int32_t* label_offsets;
    const float* norms;

    template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE,
              std::size_t BATCH_SIZE, std::size_t K>
    void run(std::size_t k)
    {
        run_bits_kernel<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, K>(
            in_dist, in_label, out_dist, out_label, k, label_offsets, norms);
    }

    template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE,
              std::size_t BATCH_SIZE>
    void run(std::size_t k)
    {
        if (k > 0 && k <= 16)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 16>(k);
        }
        else if (k > 0 && k <= 32)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 32>(k);
        }
        else if (k > 0 && k <= 64)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 64>(k);
        }
        else if (k > 0 && k <= 128)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 128>(k);
        }
        else if (k > 0 && k <= 256)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 256>(k);
        }
        else if (k > 0 && k <= 512)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 512>(k);
        }
        else if (k > 0 && k <= 1024)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 1024>(k);
        }
        else if (k > 0 && k <= 2048)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, BATCH_SIZE, 2048>(k);
        }
        else
        {
            throw std::runtime_error("Unsupported k value");
        }
    }

    template <bool PREFETCH, bool ADD_NORMS, std::size_t BLOCK_SIZE>
    void run(std::size_t batch_size, std::size_t k)
    {
        if (batch_size == 1)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 1>(k);
        }
        else if (batch_size == 2)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 2>(k);
        }
        else if (batch_size == 3)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 3>(k);
        }
        else if (batch_size == 4)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 4>(k);
        }
        else if (batch_size == 5)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 5>(k);
        }
        else if (batch_size == 6)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 6>(k);
        }
        else if (batch_size == 7)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 7>(k);
        }
        else if (batch_size == 8)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 8>(k);
        }
        else if (batch_size == 9)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 9>(k);
        }
        else if (batch_size == 10)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 10>(k);
        }
        else if (batch_size == 11)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 11>(k);
        }
        else if (batch_size == 12)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 12>(k);
        }
        else if (batch_size == 13)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 13>(k);
        }
        else if (batch_size == 14)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 14>(k);
        }
        else if (batch_size == 15)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 15>(k);
        }
        else if (batch_size == 16)
        {
            run<PREFETCH, ADD_NORMS, BLOCK_SIZE, 16>(k);
        }
        else
        {
            throw std::runtime_error("Unsupported batch size");
        }
    }

    template <bool PREFETCH, bool ADD_NORMS>
    void run(std::size_t block_size, std::size_t batch_size, std::size_t k)
    {
        if (block_size == 128)
        {
            run<PREFETCH, ADD_NORMS, 128>(batch_size, k);
        }
        else if (block_size == 256)
        {
            run<PREFETCH, ADD_NORMS, 256>(batch_size, k);
        }
        else if (block_size == 512)
        {
            run<PREFETCH, ADD_NORMS, 512>(batch_size, k);
        }
        else
        {
            throw std::runtime_error("Unsupported block size");
        }
    }
};

__global__ void populate_label_offsets_kernel(std::int32_t* label_offsets, std::size_t query_count,
                                              std::size_t parallel_count, std::size_t column_count)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < query_count * parallel_count)
    {
        label_offsets[idx] = idx % parallel_count * column_count;
    }
}

} // namespace

void bits_knn::initialize(const knn_args& args) { cuda_knn::initialize(args); }

void bits_knn::selection()
{
    cuda_knn::selection();

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = out_dist_gpu(),
                .out_label = out_label_gpu(),
                .label_offsets = nullptr,
                .norms = nullptr};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    constexpr bool PREFETCH = false;
    constexpr bool ADD_NORMS = false;
    kernel.run<PREFETCH, ADD_NORMS>(block_size, batch_size, k());

    cuda_stream::make_default().sync();
}

void bits_prefetch_knn::selection()
{
    cuda_knn::selection();

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = out_dist_gpu(),
                .out_label = out_label_gpu(),
                .label_offsets = nullptr,
                .norms = nullptr};
    const auto batch_size = args_.items_per_thread[0];
    const auto block_size = args_.selection_block_size;

    constexpr bool PREFETCH = true;
    constexpr bool ADD_NORMS = false;
    kernel.run<PREFETCH, ADD_NORMS>(block_size, batch_size, k());

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
    }
}

void single_query_bits::selection()
{
    cuda_knn::selection();

    constexpr bool PREFETCH = true;
    constexpr bool ADD_NORMS = false;

    bits kernel{.in_dist = in_dist_gpu(),
                .in_label = {}, // implicit (compute indices as labels)
                .out_dist = args_.deg == 1 ? out_dist_gpu() : tmp_dist_.view(),
                .out_label = args_.deg == 1 ? out_label_gpu() : tmp_label_.view(),
                .label_offsets = nullptr,
                .norms = nullptr};
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
    }

    kernel.in_dist =
        array_view<float, 2>{kernel.in_dist.data(), {row_count, column_count}, {column_count, 1}};

    kernel.run<PREFETCH, ADD_NORMS>(block_size, batch_size, k());

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

        kernel.run<PREFETCH, ADD_NORMS>(block_size, batch_size, k());
    }

    cuda_stream::make_default().sync();
}
