#include <cstddef>
#include <stdexcept>

#include "bits/array_view.hpp"
#include "bits/distance/tiled_distance.hpp"
#include "bits/dynamic_switch.hpp"
#include "bits/layout.hpp"

namespace
{

// tiles of the points matrix and the query matrix are loaded to shared memory
// My implementation of the algorithm from "A Practical GPU Based KNN Algorithm" Kuang et al.
template <std::size_t TILE_SIZE, std::size_t BLOCK_SIZE>
__global__ void tile_dist_kernel(array_view<float, 2> A, array_view<float, 2> B,
                                 array_view<float, 2> distances)
{
    constexpr std::size_t A_DIM = 0;
    constexpr std::size_t B_DIM = 0;
    constexpr std::size_t OTHER_DIM = 1;

    __shared__ float a_shm[TILE_SIZE][TILE_SIZE + 5]; // larger sizes to avoid bank conflicts
    __shared__ float b_shm[TILE_SIZE][TILE_SIZE + 5];

    const auto dim = A.size(OTHER_DIM);
    const auto a_count = A.size(A_DIM);
    const auto b_count = B.size(B_DIM);
    const auto grid_col_count = (a_count + TILE_SIZE - 1) / TILE_SIZE;
    const auto tid_x = threadIdx.x / TILE_SIZE;
    const auto tid_y = threadIdx.x % TILE_SIZE;
    const auto bid_x = blockIdx.x / grid_col_count;
    const auto bid_y = blockIdx.x % grid_col_count;
    const auto global_row = tid_x + bid_x * TILE_SIZE;
    const auto global_col = tid_y + bid_y * TILE_SIZE;

    float part_result = 0;

    for (std::size_t offset = 0; offset < dim; offset += TILE_SIZE)
    {
        // load tiles to shared memory
        const auto a_row = bid_y * TILE_SIZE;
        const auto b_row = bid_x * TILE_SIZE;

#pragma unroll
        for (std::size_t i = 0; i < TILE_SIZE; i += BLOCK_SIZE)
        {
            const auto row = i + tid_x;

#pragma unroll
            for (std::size_t j = 0; j < TILE_SIZE; j += BLOCK_SIZE)
            {
                const auto col = j + tid_y;

                // load values from global memory
                if (offset + col >= dim || a_row + row >= a_count)
                {
                    a_shm[row][col] = 0;
                }
                else // there is an element in global memory for (row, col)
                {
                    a_shm[row][col] = A(a_row + row, offset + col);
                }

                // load values from global memory
                if (offset + col >= dim || b_row + row >= b_count)
                {
                    b_shm[row][col] = 0;
                }
                else // there is an element in global memory for (row, col)
                {
                    b_shm[row][col] = B(b_row + row, offset + col);
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (std::size_t i = 0; i < TILE_SIZE; ++i)
        {
            const float diff = b_shm[tid_x][i] - a_shm[tid_y][i];
            part_result += diff * diff;
        }

        __syncthreads();
    }

    // store the result to the global memory
    if (global_row < b_count && global_col < a_count)
    {
        distances(global_row, global_col) = part_result;
    }
}

void run_tiled_dist(array_view<float, 2> A, array_view<float, 2> B, array_view<float, 2> dist,
                    std::size_t tile_size)
{
    const auto block_size = tile_size * tile_size;

    auto block_count = (B.size(0) + tile_size - 1) / tile_size;
    block_count *= (A.size(0) + tile_size - 1) / tile_size;

    if (!dynamic_switch<8, 16, 32>(tile_size, [&]<std::size_t TileSize>() {
            tile_dist_kernel<TileSize, TileSize><<<block_count, block_size>>>(A, B, dist);
        }))
    {
        throw std::runtime_error("Unsupported tile size: " + std::to_string(tile_size));
    }
}

} // namespace

void tiled_distance::compute()
{
    if (args_.dist_layout == matrix_layout::row_major)
    {
        run_tiled_dist(points_gpu_.view(), queries_gpu_.view(), dist_gpu_.view(),
                       args_.dist_block_size);
    }
    else // column major
    {
        run_tiled_dist(queries_gpu_.view(), points_gpu_.view(), dist_gpu_.view(),
                       args_.dist_block_size);
    }
}
