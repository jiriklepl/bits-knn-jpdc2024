#ifndef COPY_KERNEL_CUH_
#define COPY_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "bits/memory.cuh"

template <std::size_t BLOCK_SIZE_, std::size_t ITEMS_PER_THREAD, std::size_t TILES_PER_BLOCK>
struct copy_kernel
{
    // number of threads in each thread block
    static constexpr std::size_t BLOCK_SIZE = BLOCK_SIZE_;
    // number of items that fit into registers of one block
    static constexpr std::size_t TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    struct tmp_storage_t
    {
    };

    float* global_dist_in;
    float* global_dist_out;
    std::int32_t* global_label_in;
    std::int32_t* global_label_out;
    std::int32_t input_size;

    __device__ __forceinline__ void set_tmp_storage(tmp_storage_t&) {}

    /** Copy pairs to the output buffer.
     */
    __device__ __forceinline__ void operator()()
    {
        float dist[ITEMS_PER_THREAD];
        std::int32_t label[ITEMS_PER_THREAD];

        std::int32_t i = blockIdx.x * TILE_SIZE * TILES_PER_BLOCK;
        const std::int32_t end =
            std::min<std::int32_t>(i + TILE_SIZE * TILES_PER_BLOCK, input_size);
        for (; i + TILE_SIZE <= end; i += TILE_SIZE)
        {
            load_striped<BLOCK_SIZE>(dist, global_dist_in + i);
            store_striped<BLOCK_SIZE>(dist, global_dist_out + i);

            load_striped<BLOCK_SIZE>(label, global_label_in + i);
            store_striped<BLOCK_SIZE>(label, global_label_out + i);
        }

        if (i < end)
        {
            load_striped<BLOCK_SIZE>(dist, global_dist_in + i, end - i);
            store_striped<BLOCK_SIZE>(dist, global_dist_out + i, end - i);

            load_striped<BLOCK_SIZE>(label, global_label_in + i, end - i);
            store_striped<BLOCK_SIZE>(label, global_label_out + i, end - i);
        }
    }
};

#endif // COPY_KERNEL_CUH_
