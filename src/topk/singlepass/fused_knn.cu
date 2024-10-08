#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bits/cuda_array.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/topk/singlepass/fused_knn.hpp"

#include "bits/topk/bitonic_sort_regs.cuh"
#include "bits/topk/singlepass/fused_kernel.cuh"

namespace
{

/** Auxiliary function to transform runtime variable values to template constants.
 */
template <std::size_t K, std::int32_t REG_QUERY_COUNT, std::int32_t REG_POINT_COUNT>
void run(fused_kernel_runner& kernel)
{
    if (kernel.block_size == 4)
    {
        kernel.operator()<K, REG_QUERY_COUNT, REG_POINT_COUNT, 4>();
    }
    else if (kernel.block_size == 8)
    {
        kernel.operator()<K, REG_QUERY_COUNT, REG_POINT_COUNT, 8>();
    }
    else if (kernel.block_size == 16)
    {
        kernel.operator()<K, REG_QUERY_COUNT, REG_POINT_COUNT, 16>();
    }
    else
    {
        throw std::runtime_error{"Unsupported block size"};
    }
}

/** Auxiliary function to transform runtime variable values to template constants.
 */
template <std::size_t K, std::int32_t REG_QUERY_COUNT>
void run(fused_kernel_runner& kernel)
{
    if (kernel.items_per_thread[1] == 4)
    {
        run<K, REG_QUERY_COUNT, 4>(kernel);
    }
    else
    {
        throw std::runtime_error{"Unsupported point register count"};
    }
}

/** Auxiliary function to transform runtime variable values to template constants.
 */
template <std::size_t K>
void run(fused_kernel_runner& kernel)
{
    if (kernel.items_per_thread[0] == 2)
    {
        run<K, 2>(kernel);
    }
    else if (kernel.items_per_thread[0] == 4)
    {
        run<K, 4>(kernel);
    }
    else if (kernel.items_per_thread[0] == 8)
    {
        run<K, 8>(kernel);
    }
    else
    {
        throw std::runtime_error{"Unsupported query register count"};
    }
}

/** Auxiliary function to transform runtime variable values to template constants.
 */
void run(fused_kernel_runner& kernel)
{
    if (kernel.k == 2)
    {
        run<2>(kernel);
    }
    else if (kernel.k == 4)
    {
        run<4>(kernel);
    }
    else if (kernel.k == 8)
    {
        run<8>(kernel);
    }
    else if (kernel.k == 16)
    {
        run<16>(kernel);
    }
    else if (kernel.k == 32)
    {
        run<32>(kernel);
    }
    else if (kernel.k == 64)
    {
        run<64>(kernel);
    }
    else if (kernel.k == 128)
    {
        run<128>(kernel);
    }
    else
    {
        throw std::runtime_error{"Unsupported k value"};
    }
}

} // namespace

void fused_regs_knn::initialize(const knn_args& args)
{
    // skip allocation in cuda_knn::initialize()
    knn::initialize(args);

    out_dist_gpu_ = cuda_array<float, 2>{{query_count(), k()}};
    out_label_gpu_ = cuda_array<std::int32_t, 2>{{query_count(), k()}};
    points_gpu_ = cuda_array<float, 2>{{dim(), point_count()}};
    queries_gpu_ = cuda_array<float, 2>{{dim(), query_count()}};

    // transpose the points and query matrices
    std::vector<float> points_transposed(points_gpu_.view().size());
    std::vector<float> queries_transposed(queries_gpu_.view().size());
    for (std::size_t i = 0; i < point_count(); ++i)
    {
        for (std::size_t j = 0; j < dim(); ++j)
        {
            points_transposed[j * point_count() + i] = args_.points[i * dim() + j];
        }
    }

    for (std::size_t i = 0; i < query_count(); ++i)
    {
        for (std::size_t j = 0; j < dim(); ++j)
        {
            queries_transposed[j * query_count() + i] = args_.queries[i * dim() + j];
        }
    }

    cuda_stream::make_default()
        .copy_to_gpu_async(points_gpu_.view(), points_transposed.data())
        .copy_to_gpu_async(queries_gpu_.view(), queries_transposed.data())
        .sync();
}

void fused_regs_knn::distances()
{
    // no computation
}

void fused_regs_knn::selection()
{
    fused_kernel_runner run{.points = points_gpu_.view(),
                            .queries = queries_gpu_.view(),
                            .out_dist = out_dist_gpu(),
                            .out_label = out_label_gpu(),
                            .k = k(),
                            .block_size = selection_block_size(),
                            .items_per_thread = {4, 4}};

    constexpr int POINT_REGS = 4;

    if (k() == 2)
    {
        constexpr int QUERY_REGS = 4;
        constexpr int BLOCK_SIZE = 8;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<2, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 4)
    {
        constexpr int QUERY_REGS = 8;
        constexpr int BLOCK_SIZE = 4;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<4, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 8)
    {
        constexpr int QUERY_REGS = 8;
        constexpr int BLOCK_SIZE = 4;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<8, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 16)
    {
        constexpr int QUERY_REGS = 4;
        constexpr int BLOCK_SIZE = 8;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<16, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 32)
    {
        constexpr int QUERY_REGS = 4;
        constexpr int BLOCK_SIZE = 8;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<32, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 64)
    {
        constexpr int QUERY_REGS = 2;
        constexpr int BLOCK_SIZE = 8;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<64, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else if (k() == 128)
    {
        constexpr int QUERY_REGS = 2;
        constexpr int BLOCK_SIZE = 8;

        run.block_size = BLOCK_SIZE;
        run.items_per_thread[0] = QUERY_REGS;
        run.template operator()<128, QUERY_REGS, POINT_REGS, BLOCK_SIZE>();
    }
    else
    {
        throw std::runtime_error{"Unsupported k value"};
    }

    cuda_stream::make_default().sync();
}

void fused_regs_knn_tunable::selection()
{
    auto points = points_gpu_.view();
    auto queries = queries_gpu_.view();

    fused_kernel_runner kernel;
    kernel.points = points;
    kernel.queries = queries;
    kernel.out_dist = out_dist_gpu();
    kernel.out_label = out_label_gpu();
    kernel.k = k();
    kernel.block_size = selection_block_size();
    kernel.items_per_thread = args_.items_per_thread;
    run(kernel);

    cuda_stream::make_default().sync();
}
