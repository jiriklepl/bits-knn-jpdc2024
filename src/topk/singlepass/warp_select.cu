#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/WarpSelectKernel.cuh>

#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/expand.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

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
    else if (k() == 2048)
    {
        constexpr int BLOCK_SIZE = 64;
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

void warp_select_tunable::selection()
{
    cuda_knn::selection();

    warp_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                           .out_dist_tensor = to_tensor(out_dist_gpu()),
                           .out_label_tensor = to_tensor(out_label_gpu()),
                           .block_size = (std::int32_t)selection_block_size(),
                           .thread_queue_size = (std::int32_t)args_.items_per_thread[0],
                           .k = (std::int32_t)k()};

    using thread_block_choice_t = choice<128>;
    using thread_queue_choice_t = choice<2, 3, 4, 5, 6, 7, 8, 9, 10>;
    using k_choice_t = choice<32, 64, 128, 256, 512, 1024, 2048>;

    expand<thread_block_choice_t, thread_queue_choice_t, k_choice_t>(run);

    cuda_stream::make_default().sync();
}

void warp_select_tuned::selection()
{
    cuda_knn::selection();

    constexpr int BLOCK_SIZE = 128;

    warp_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                           .out_dist_tensor = to_tensor(out_dist_gpu()),
                           .out_label_tensor = to_tensor(out_label_gpu()),
                           .block_size = BLOCK_SIZE,
                           .thread_queue_size = 2, // default thread queue size in FIASS
                           .k = (std::int32_t)k()};

    if (k() == 32)
    {
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 32>();
    }
    else if (k() == 64)
    {
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 64>();
    }
    else if (k() == 128)
    {
        constexpr int THREAD_QUEUE_SIZE = 6;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 128>();
    }
    else if (k() == 256)
    {
        constexpr int THREAD_QUEUE_SIZE = 4;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 256>();
    }
    else if (k() == 512)
    {
        constexpr int THREAD_QUEUE_SIZE = 5;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 512>();
    }
    else if (k() == 1024)
    {
        constexpr int THREAD_QUEUE_SIZE = 9;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 1024>();
    }
    else if (k() == 2048)
    {
        constexpr int THREAD_QUEUE_SIZE = 10;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 2048>();
    }
    else
    {
        throw std::runtime_error{"Unsupported k value: " + std::to_string(k())};
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

    using thread_block_choice_t = choice<128>;
    using thread_queue_choice_t = choice<2, 3, 4, 5, 6, 7, 8, 9, 10>;
    using k_choice_t = choice<32, 64, 128, 256, 512, 1024, 2048>;

    expand<thread_block_choice_t, thread_queue_choice_t, k_choice_t>(run);

    cuda_stream::make_default().sync();
}

void block_select_tuned::selection()
{
    cuda_knn::selection();

    constexpr int BLOCK_SIZE = 128;

    block_select_runner run{.dist_tensor = to_tensor(in_dist_gpu()),
                            .out_dist_tensor = to_tensor(out_dist_gpu()),
                            .out_label_tensor = to_tensor(out_label_gpu()),
                            .block_size = BLOCK_SIZE,
                            .thread_queue_size = 2, // default thread queue size in FIASS
                            .k = static_cast<std::int32_t>(k())};

    if (k() == 32)
    {
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 32>();
    }
    else if (k() == 64)
    {
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 64>();
    }
    else if (k() == 128)
    {
        constexpr int THREAD_QUEUE_SIZE = 2;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 128>();
    }
    else if (k() == 256)
    {
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 256>();
    }
    else if (k() == 512)
    {
        constexpr int THREAD_QUEUE_SIZE = 8;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 512>();
    }
    else if (k() == 1024)
    {
        constexpr int THREAD_QUEUE_SIZE = 10;

        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<BLOCK_SIZE, THREAD_QUEUE_SIZE, 1024>();
    }
    else if (k() == 2048)
    {
        constexpr int THREAD_QUEUE_SIZE = 10;

        run.block_size = 64;
        run.thread_queue_size = THREAD_QUEUE_SIZE;
        run.template operator()<64, THREAD_QUEUE_SIZE, 2048>();
    }
    else
    {
        throw std::runtime_error{"Unsupported k value: " + std::to_string(k())};
    }

    cuda_stream::make_default().sync();
}
