#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

#include "bits/topk/singlepass/detail/definitions_common.hpp"

#include "bits/topk/singlepass/block_select_runner.cuh"
#include "bits/topk/singlepass/warp_select_runner.cuh"

namespace
{

// convert array view to a tensor
template <typename T, std::size_t DIM>
faiss::gpu::Tensor<T, DIM, true> to_tensor(array_view<T, DIM> view)
{
    std::int64_t strides[DIM];
    std::int64_t sizes[DIM];

    for (std::size_t i = 0; i < DIM; ++i)
    {
        sizes[i] = static_cast<std::int64_t>(view.size(i));
        strides[i] = static_cast<std::int64_t>(view.stride(i));
    }

    return faiss::gpu::Tensor<T, DIM, true>{view.data(), sizes, strides};
}

} // namespace

void warp_select::selection()
{
    cuda_knn::selection();

    warp_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                           .out_dist_tensor = to_tensor(out_dist_gpu()),
                           .out_label_tensor = to_tensor(out_label_gpu()),
                           .block_size = 128,      // default thread block size in FIASS
                           .thread_queue_size = 2, // default thread queue size in FIASS
                           .k = (std::int32_t)k()};

    // the default configuration in v1.7.4
    if (k() == 32)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 32>();
    }
    else if (k() == 64)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 3;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 64>();
    }
    else if (k() == 128)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 3;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 128>();
    }
    else if (k() == 256)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 4;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 256>();
    }
    else if (k() == 512)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 512>();
    }
    else if (k() == 1024)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 1024>();
    }
    else if (k() == 2048)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 2048>();
    }
    else
    {
        throw std::runtime_error{"Unsupported k value: " + std::to_string(k())};
    }

    cuda_stream::make_default().sync();
}

void block_select::selection()
{
    cuda_knn::selection();

    block_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                            .out_dist_tensor = to_tensor(out_dist_gpu()),
                            .out_label_tensor = to_tensor(out_label_gpu()),
                            .block_size = k() <= 1024 ? 128 : 64,
                            .thread_queue_size = 2, // default thread queue size in FIASS
                            .k = (std::int32_t)k()};

    // the default configuration in v1.7.4
    if (k() == 32)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 32>();
    }
    else if (k() == 64)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 3;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 64>();
    }
    else if (k() == 128)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 3;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 128>();
    }
    else if (k() == 256)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 4;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 256>();
    }
    else if (k() == 512)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 512>();
    }
    else if (k() == 1024)
    {
        constexpr int BLOCK_SIZE = 128;
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 1024>();
    }
    else
    {
        throw std::runtime_error{"Unsupported k value: " + std::to_string(k())};
    }

    cuda_stream::make_default().sync();
}

void warp_select_tunable::selection()
{
    cuda_knn::selection();

    warp_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                           .out_dist_tensor = to_tensor(out_dist_gpu()),
                           .out_label_tensor = to_tensor(out_label_gpu()),
                           .block_size = (std::int32_t)selection_block_size(),
                           .thread_queue_size = (std::int32_t)args_.items_per_thread[0],
                           .k = (std::int32_t)k()};

    if (!dynamic_switch<TOPK_SINGLEPASS_FAISS_BLOCK_SIZES>(
            run.block_size, [=, &run]<std::size_t BlockSize>() {
                if (!dynamic_switch<TOPK_SINGLEPASS_FAISS_THREAD_QUEUES>(
                        run.thread_queue_size, [=, &run]<std::size_t ThreadQueueSize>() {
                            if (!dynamic_switch<TOPK_SINGLEPASS_K_VALUES>(
                                    run.k, [=, &run]<std::size_t K>() {
                                        run.template operator()<BlockSize, ThreadQueueSize, K>();
                                    }))
                            {
                                throw std::runtime_error{"Unsupported k value: " +
                                                         std::to_string(run.k)};
                            }
                        }))
                {
                    throw std::runtime_error{"Unsupported thread queue size: " +
                                             std::to_string(run.thread_queue_size)};
                }
            }))
    {
        throw std::runtime_error{"Unsupported block size: " + std::to_string(run.block_size)};
    }

    cuda_stream::make_default().sync();
}

void block_select_tunable::selection()
{
    cuda_knn::selection();

    block_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                            .out_dist_tensor = to_tensor(out_dist_gpu()),
                            .out_label_tensor = to_tensor(out_label_gpu()),
                            .block_size = static_cast<std::int32_t>(selection_block_size()),
                            .thread_queue_size =
                                static_cast<std::int32_t>(args_.items_per_thread[0]),
                            .k = static_cast<std::int32_t>(k())};

    if (!dynamic_switch<TOPK_SINGLEPASS_FAISS_BLOCK_SIZES>(
            run.block_size, [=, &run]<std::size_t BlockSize>() {
                if (!dynamic_switch<TOPK_SINGLEPASS_FAISS_THREAD_QUEUES>(
                        run.thread_queue_size, [=, &run]<std::size_t ThreadQueueSize>() {
                            if (!dynamic_switch<TOPK_SINGLEPASS_K_VALUES>(
                                    run.k, [=, &run]<std::size_t K>() {
                                        if constexpr (K <= 1024)
                                        {
                                            run.template
                                            operator()<BlockSize, ThreadQueueSize, K>();
                                        }
                                        else
                                        {
                                            throw std::runtime_error{"Unsupported k value: " +
                                                                     std::to_string(run.k)};
                                        }
                                    }))
                            {
                                throw std::runtime_error{"Unsupported k value: " +
                                                         std::to_string(run.k)};
                            }
                        }))
                {
                    throw std::runtime_error{"Unsupported thread queue size: " +
                                             std::to_string(run.thread_queue_size)};
                }
            }))
    {
        throw std::runtime_error{"Unsupported block size: " + std::to_string(run.block_size)};
    }

    cuda_stream::make_default().sync();
}
