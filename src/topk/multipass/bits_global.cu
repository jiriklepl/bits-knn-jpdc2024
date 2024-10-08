#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bits/array_view.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/knn.hpp"
#include "bits/topk/multipass/bits_global.hpp"
#include "bits/topk/multipass/bits_global_kernel.hpp"

namespace
{

template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE, std::size_t BATCH_SIZE>
void call_global_kernel(array_view<float, 2> dist, array_view<float, 2> out_dist,
                        array_view<std::int32_t, 2> out_label, std::size_t k)
{
    run_bits_global_kernel<BUFFER_SIZE, BLOCK_SIZE, BATCH_SIZE>(dist, out_dist, out_label, k);
}

template <std::size_t BUFFER_SIZE, std::size_t BLOCK_SIZE>
void call_global_kernel(array_view<float, 2> dist, array_view<float, 2> out_dist,
                        array_view<std::int32_t, 2> out_label, std::size_t k,
                        std::size_t batch_size)
{
    if (batch_size == 1)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 1>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 2)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 2>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 3)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 3>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 4)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 4>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 5)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 5>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 6)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 6>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 7)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 7>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 8)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 8>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 9)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 9>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 10)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 10>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 11)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 11>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 12)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 12>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 13)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 13>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 14)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 14>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 15)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 15>(dist, out_dist, out_label, k);
    }
    else if (batch_size == 16)
    {
        call_global_kernel<BUFFER_SIZE, BLOCK_SIZE, 16>(dist, out_dist, out_label, k);
    }
    else
    {
        throw std::runtime_error("Unsupported batch size");
    }
}

template <std::size_t BUFFER_SIZE>
void call_global_kernel(array_view<float, 2> dist, array_view<float, 2> out_dist,
                        array_view<std::int32_t, 2> out_label, std::size_t k,
                        std::size_t block_size, std::size_t batch_size)
{
    if (block_size == 128)
    {
        call_global_kernel<BUFFER_SIZE, 128>(dist, out_dist, out_label, k, batch_size);
    }
    else if (block_size == 256)
    {
        call_global_kernel<BUFFER_SIZE, 256>(dist, out_dist, out_label, k, batch_size);
    }
    else if (block_size == 512)
    {
        call_global_kernel<BUFFER_SIZE, 512>(dist, out_dist, out_label, k, batch_size);
    }
    else if (block_size == 1024)
    {
        call_global_kernel<BUFFER_SIZE, 1024>(dist, out_dist, out_label, k, batch_size);
    }
    else if (block_size == 2048)
    {
        call_global_kernel<BUFFER_SIZE, 2048>(dist, out_dist, out_label, k, batch_size);
    }
    else
    {
        throw std::runtime_error("Unsupported block size");
    }
}

} // namespace

void bits_global::selection()
{
    cuda_knn::selection();

    const auto block_size = selection_block_size();
    const auto batch_size = args_.items_per_thread[0];

    auto dist = in_dist_gpu();
    auto out_dist = out_dist_gpu_.view();
    auto out_label = out_label_gpu_.view();

    if (k() <= 128)
    {
        call_global_kernel<128>(dist, out_dist, out_label, k(), block_size, batch_size);
    }
    else if (k() == 256)
    {
        call_global_kernel<256>(dist, out_dist, out_label, k(), block_size, batch_size);
    }
    else if (k() == 512)
    {
        call_global_kernel<512>(dist, out_dist, out_label, k(), block_size, batch_size);
    }
    else if (k() == 1024)
    {
        call_global_kernel<1024>(dist, out_dist, out_label, k(), block_size, batch_size);
    }
    else if (k() >= 2048)
    {
        call_global_kernel<2048>(dist, out_dist, out_label, k(), block_size, batch_size);
    }
    else
    {
        throw std::runtime_error("Unsupported k value");
    }

    cuda_stream::make_default().sync();
}

std::vector<knn::pair_t> bits_global::finish()
{
    std::vector<knn::pair_t> result(query_count() * k());
    std::vector<float> dist(query_count() * k());
    std::vector<std::int32_t> label(query_count() * k());

    transfer_begin_.record();
    cuda_stream::make_default()
        .copy_from_gpu_async(dist.data(), out_dist_gpu_.view())
        .copy_from_gpu_async(label.data(), out_label_gpu_.view())
        .sync();
    transfer_end_.record();
    transfer_end_.sync();

    for (std::size_t i = 0; i < query_count(); ++i)
    {
        for (std::size_t j = 0; j < k(); ++j)
        {
            const auto idx = i * k() + j;
            auto& out = result[idx];
            out.distance = dist[idx];
            out.index = label[idx];
        }
    }

    // output of this method should be a list of sorted top k results
    for (std::size_t i = 0; i < query_count(); ++i)
    {
        const auto begin = result.begin() + i * k();
        const auto end = begin + k();
        std::sort(begin, end, [](auto&& lhs, auto&& rhs) { return lhs.distance < rhs.distance; });
    }

    return result;
}
