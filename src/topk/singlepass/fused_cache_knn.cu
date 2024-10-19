#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bits/cuda_stream.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/topk/singlepass/fused_cache_kernel.hpp"
#include "bits/topk/singlepass/fused_cache_knn.hpp"

#include "bits/topk/singlepass/detail/definitions_common.hpp"

#include "bits/topk/singlepass/fused_cache_kernel_structs.cuh"

namespace
{

struct fused_cache
{
    const float* queries;
    const float* db;
    std::int32_t dim;
    std::int32_t num_queries;
    std::int32_t num_db;
    float* out_dist;
    std::int32_t* out_label;

    template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG,
              std::int32_t BLOCK_QUERY_DIM, std::int32_t DIM_MULT, std::int32_t K>
    void run()
    {
        constexpr std::int32_t BLOCK_SIZE = 256;
        constexpr std::int32_t BLOCK_DB_DIM = BLOCK_SIZE / BLOCK_QUERY_DIM;
        constexpr std::int32_t DIM_TILE = DIM_MULT * DIM_REG;
        constexpr std::int32_t QUERIES_PER_BLOCK = BLOCK_QUERY_DIM * QUERY_REG;

        using kernel_t = fused_cache_kernel<QUERY_REG, DB_REG, DIM_REG, BLOCK_QUERY_DIM,
                                            BLOCK_DB_DIM, DIM_MULT, K>;

        kernel_t kernel{.tmp_storage = nullptr,
                        .queries = queries,
                        .db = db,
                        .dim = dim,
                        .num_queries = num_queries,
                        .num_db = num_db,
                        .out_dist = out_dist,
                        .out_label = out_label};

        const dim3 block(BLOCK_QUERY_DIM, BLOCK_DB_DIM, 1);
        const dim3 grid((num_queries + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK, 1, 1);

        if (dim < DIM_TILE)
        {
            throw std::runtime_error{"Dimension of vectors must not be lower than " +
                                     std::to_string(DIM_TILE) + ", but is " + std::to_string(dim)};
        }

        launch_fused_cache(kernel, grid, block);
    }

    template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG,
              std::int32_t QUERY_BLOCK_DIM, std::int32_t DIM_MULT>
    void run(std::int32_t k)
    {
        if (!dynamic_switch<TOPK_SINGLEPASS_FUSED_CACHE_K_VALUES>(k, [&]<std::size_t K>() {
            run<QUERY_REG, DB_REG, DIM_REG, QUERY_BLOCK_DIM, DIM_MULT, K>();
        })) {
            throw std::runtime_error("Unsupported k value: " + std::to_string(k));
        }
    }

    template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG,
              std::int32_t QUERY_BLOCK_DIM>
    void run(std::int32_t dim_mult, std::int32_t k)
    {
        if (!dynamic_switch<1, 2, 4>(dim_mult, [&]<std::size_t DIM_MULT>() {
            run<QUERY_REG, DB_REG, DIM_REG, QUERY_BLOCK_DIM, DIM_MULT>(k);
        })) {
            throw std::runtime_error("Unsupported dim_mult value: " + std::to_string(dim_mult));
        }
    }

    template <std::int32_t QUERY_REG, std::int32_t DB_REG, std::int32_t DIM_REG>
    void run(std::int32_t query_block_size, std::int32_t dim_mult, std::int32_t k)
    {
        if (!dynamic_switch<1, 2, 4>(query_block_size, [&]<std::size_t QUERY_BLOCK_DIM>() {
            run<QUERY_REG, DB_REG, DIM_REG, QUERY_BLOCK_DIM>(dim_mult, k);
        })) {
            throw std::runtime_error("Unsupported query_block_size value: " +
                                     std::to_string(query_block_size));
        }
    }

    template <std::int32_t QUERY_REG, std::int32_t DB_REG>
    void run(std::int32_t dim_reg, std::int32_t query_block_size, std::int32_t dim_mult,
             std::int32_t k)
    {
        if (!dynamic_switch<1, 2, 4>(dim_reg, [&]<std::size_t DIM_REG>() {
            run<QUERY_REG, DB_REG, DIM_REG>(query_block_size, dim_mult, k);
        })) {
            throw std::runtime_error("Unsupported dim_reg value: " + std::to_string(dim_reg));
        }
    }

    template <std::int32_t QUERY_REG>
    void run(std::int32_t db_reg, std::int32_t dim_reg, std::int32_t query_block_size,
             std::int32_t dim_mult, std::int32_t k)
    {
        if (!dynamic_switch<4, 8, 16>(db_reg, [&]<std::size_t DB_REG>() {
            run<QUERY_REG, DB_REG>(dim_reg, query_block_size, dim_mult, k);
        })) {
            throw std::runtime_error("Unsupported db_reg value: " + std::to_string(db_reg));
        }
    }

    void run(std::int32_t queries_reg, std::int32_t db_reg, std::int32_t dim_reg,
             std::int32_t query_block_size, std::int32_t dim_mult, std::int32_t k)
    {
        if (!dynamic_switch<2, 4, 8, 16>(queries_reg, [&]<std::size_t QUERY_REG>() {
            run<QUERY_REG>(db_reg, dim_reg, query_block_size, dim_mult, k);
        })) {
            throw std::runtime_error("Unsupported queries_reg value: " + std::to_string(queries_reg));
        }
    }
};

} // namespace

void fused_cache_knn::initialize(const knn_args& args)
{
    // skip allocation in cuda_knn::initialize()
    knn::initialize(args);

    out_dist_gpu_ = cuda_array<float, 2>{{query_count(), k()}};
    out_label_gpu_ = cuda_array<std::int32_t, 2>{{query_count(), k()}};
    points_gpu_ = cuda_array<float, 2>{{dim(), point_count()}};
    queries_gpu_ = cuda_array<float, 2>{{query_count(), dim()}};

    // transpose the DB matrix if necessary
    auto points = args_.points;
    std::vector<float> points_transposed;
    if (args.points_layout != matrix_layout::column_major)
    {
        points_transposed.resize(points_gpu_.view().size());
        for (std::size_t i = 0; i < point_count(); ++i)
        {
            for (std::size_t j = 0; j < dim(); ++j)
            {
                points_transposed[j * point_count() + i] = args_.points[i * dim() + j];
            }
        }
        points = points_transposed.data();
    }

    // transpose the query matrix if necessary
    auto queries = args_.queries;
    std::vector<float> queries_transposed;
    if (args.queries_layout != matrix_layout::column_major)
    {
        queries_transposed.resize(queries_gpu_.view().size());
        for (std::size_t i = 0; i < query_count(); ++i)
        {
            for (std::size_t j = 0; j < dim(); ++j)
            {
                queries_transposed[j * query_count() + i] = args_.queries[i * dim() + j];
            }
        }
        queries = queries_transposed.data();
    }

    cuda_stream::make_default()
        .copy_to_gpu_async(points_gpu_.view(), points)
        .copy_to_gpu_async(queries_gpu_.view(), queries)
        .sync();
}

void fused_cache_knn::distances()
{
    // no computation
}

void fused_cache_knn::selection()
{
    fused_cache kernel{.queries = queries_gpu_.view().data(),
                       .db = points_gpu_.view().data(),
                       .dim = (std::int32_t)args_.dim,
                       .num_queries = (std::int32_t)query_count(),
                       .num_db = (std::int32_t)point_count(),
                       .out_dist = out_dist_gpu_.view().data(),
                       .out_label = out_label_gpu_.view().data()};

    kernel.run(args_.items_per_thread[0], args_.items_per_thread[1], args_.items_per_thread[2],
               args_.selection_block_size, args_.deg, k());

    cuda_stream::make_default().sync();
}
