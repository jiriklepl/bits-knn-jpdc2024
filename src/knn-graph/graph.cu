#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "bits/knn-graph/graph.hpp"
#include "bits/knn_args.hpp"
#include "bits/topk/singlepass/bits_kernel.hpp"

#include "bits/distance/abstract_gemm.cuh"

void knn_graph::prepare(const knn_args& args)
{
    args_ = args;
    db_gpu_ = cuda_array<float, 2>{{args_.dim, args_.point_count}};
    for (std::size_t i = 0; i < knn_graph::CUDA_STREAMS; ++i)
    {
        dist_gpu_[i] = cuda_array<float, 2>{{args_.query_count, args_.point_count}};
        topk_dist_gpu_[i] = cuda_array<float, 2>{{args_.query_count, args_.k}};
        topk_label_gpu_[i] = cuda_array<std::int32_t, 2>{{args_.query_count, args_.k}};
    }
    topk_label_.resize(args_.point_count * args_.k);

    cuda_stream::make_default()
        .copy_to_gpu_async(db_gpu_.view().data(), args_.points, args_.dim * args_.point_count)
        .sync();
}

void knn_graph::run()
{
    std::array<cuda_stream, CUDA_STREAMS> streams;

    std::size_t dim = args_.dim;
    std::size_t query_batch = args_.query_count;
    std::size_t db_size = args_.point_count;
    std::size_t k = args_.k;
    auto db_gpu = db_gpu_.view();

    std::size_t stream_idx = 0;
    for (std::size_t i = 0; i < db_size; i += query_batch, ++stream_idx)
    {
        if (stream_idx >= knn_graph::CUDA_STREAMS)
        {
            stream_idx = 0;
        }
        auto& stream = streams[stream_idx];
        auto dist_gpu = dist_gpu_[stream_idx].view();
        auto topk_dist_gpu = topk_dist_gpu_[stream_idx].view();
        auto topk_label_gpu = topk_label_gpu_[stream_idx].view();

        // compute distances
        // we assume `query_batch` is a divisor of `db_size`
        run_abstract_gemm<partial_l2_ops>(dim, query_batch, db_size, /*stride_a=*/db_size,
                                          /*stride_b=*/db_size, db_gpu.data() + i, db_gpu.data(),
                                          dist_gpu.data(), stream);

        // select top k
        constexpr std::size_t BLOCK_SIZE = 128;
        constexpr std::size_t BATCH_SIZE = 16;
        if (k > 0 && k <= 16)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 16>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 32)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 32>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 64)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 64>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 128)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 128>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 256)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 256>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 512)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 512>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 1024)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 1024>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else if (k > 0 && k <= 2048)
        {
            run_bits_kernel<false, false, false, BLOCK_SIZE, BATCH_SIZE, 2048>(
                dist_gpu, {}, topk_dist_gpu, topk_label_gpu, k, nullptr, nullptr, stream.get());
        }
        else
        {
            throw std::runtime_error("Unsupported k value");
        }

        // copy the result to the CPU
        stream.copy_from_gpu_async(topk_label_.data() + k * i, topk_label_gpu.data(),
                                   topk_label_gpu.size());
    }

    for (auto& stream : streams)
    {
        stream.sync();
    }
}
