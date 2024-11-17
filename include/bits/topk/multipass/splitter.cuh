#ifndef SPLITTER_CUH_
#define SPLITTER_CUH_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <curand_kernel.h>

#include "bits/topk/bitonic_sort_regs.cuh"

/** Tag to indicate to the `splitter` class that it should initialize the cuRAND library.
 *
 * It is invalid to use `splitter::sample()` without this tag.
 */
struct enable_sample
{
};

/** Split distances (floats) to buckets based on randomly selected splitters.
 *
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam SAMPLES_PER_THREAD number of samples selected by each thread.
 * @tparam BUCKET_BITS log2 of the number of buckets.
 * @tparam T type of the distances.
 */
template <std::size_t BLOCK_SIZE, std::size_t SAMPLES_PER_THREAD, std::size_t BUCKET_BITS,
          typename T>
struct splitter
{
    using key_t = T;
    using bucket_t = std::conditional_t<BUCKET_BITS <= 8, std::uint8_t, std::uint16_t>;

    // total number of buckets
    static constexpr std::size_t BUCKET_COUNT = 1 << BUCKET_BITS;
    // number of samples
    static constexpr std::size_t SAMPLE_SIZE = BLOCK_SIZE * SAMPLES_PER_THREAD;
    // number of splitters
    static constexpr std::size_t SPLITTER_COUNT = 2 * BUCKET_COUNT - 1;
    // number of splitters per thread
    static constexpr std::size_t SPLITTERS_PER_THREAD =
        (SPLITTER_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /** Shared memory used by the splitter.
     */
    union tmp_storage_t
    {
        // shared memory for sorting
        struct
        {
            key_t keys[BLOCK_SIZE * 2];
            std::int32_t values[BLOCK_SIZE * 2];
        };

        // shared memory for splitter tree
        key_t splitters[2 * BUCKET_COUNT - 1];
    };

    /** State of the splitter (the splitter distances in global memory).
     */
    struct state_t
    {
        key_t* values;
    };

    // shared memory used by this algorithm
    tmp_storage_t& tmp_storage;
    // random number generator state
    curandState_t state;

    /** Initialize the splitter without initializing the cuRAND library
     *
     * Splitter created with this constructor cannot use the `sample()` method.
     *
     * @param storage reference to the shared memory used by this splitter.
     */
    __device__ __forceinline__ explicit splitter(tmp_storage_t& storage) : tmp_storage(storage) {}

    /** Initialize the splitter and the cuRAND library.
     *
     * @param storage reference to the shared memory used by this splitter.
     */
    __device__ __forceinline__ splitter(tmp_storage_t& storage, enable_sample)
        : tmp_storage(storage)
    {
        constexpr std::size_t SEED = 42;
        curand_init(SEED, threadIdx.x, 0, &state);
    }

    /** Load splitters from global memory.
     *
     * @param state state in global memory
     */
    __device__ __forceinline__ void load(state_t state)
    {
#pragma unroll
        for (std::size_t i = 0; i < SPLITTERS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (idx < SPLITTER_COUNT)
            {
                tmp_storage.splitters[idx] = state.values[idx];
            }
        }
        __syncthreads();
    }

    /** Store splitters to global memory
     *
     * @param state state in global memory
     */
    __device__ __forceinline__ void store(state_t state)
    {
#pragma unroll
        for (std::size_t i = 0; i < SPLITTERS_PER_THREAD; ++i)
        {
            const auto idx = threadIdx.x + i * BLOCK_SIZE;
            if (idx < SPLITTER_COUNT)
            {
                state.values[idx] = tmp_storage.splitters[idx];
            }
        }
        __syncthreads();
    }

    /** Select a random sample from [ @p segment_begin , @p segment_end ), select splitters and
     * create an internal datastructure for searchning the splitters.
     *
     * This method sould only be used if the splitter was created with `enable_sample`.
     *
     * @param data_ptr Base data pointer
     * @param segment_begin Segment begin in @p data_ptr
     * @param segment_end Segment end in @p data_ptr
     */
    __device__ __forceinline__ void sample(key_t* data_ptr, std::size_t segment_begin,
                                           std::size_t segment_end)
    {
        // load a random sample
        key_t sample[SAMPLES_PER_THREAD];
        std::int32_t values[SAMPLES_PER_THREAD];

        const auto segment_size = segment_end - segment_begin;

#pragma unroll
        for (std::size_t i = 0; i < SAMPLES_PER_THREAD; ++i)
        {
            const auto idx = segment_begin + curand(&state) % segment_size;
            sample[i] = data_ptr[idx];
        }

        // sort the sample
        block_sort<1, BLOCK_SIZE, SAMPLE_SIZE>(sample, values, tmp_storage.keys,
                                               tmp_storage.values);

        // select splitters and construct an implicit binary tree in shared memory
        constexpr std::size_t STRIDE = SAMPLE_SIZE / BUCKET_COUNT;
#pragma unroll
        for (std::size_t i = 0; i < SAMPLES_PER_THREAD; ++i)
        {
            std::uint32_t idx = threadIdx.x * SAMPLES_PER_THREAD + i;
            if ((idx % STRIDE) == 0)
            {
                std::uint32_t value_idx = idx / STRIDE;
                std::uint32_t leaf_idx = BUCKET_COUNT - 1 + value_idx;
                tmp_storage.splitters[leaf_idx] = sample[i];

                // value at index 0 is the only value that does not have an inner node
                if (value_idx <= 0)
                {
                    continue;
                }

                // Find the highest power of two that divides the index. This determines the level
                // in the inner nodes of the tree since indices on level i above the leaf level can
                // be written as 2^i * (1 + 2j) where j is offset on the level.
                std::uint32_t level = __clz(__brev(value_idx));
                std::uint32_t offset = value_idx >> (level + 1);
                // A complete binary tree with 2^i leafs has 2^{i + 1} - 1 nodes. We use this to
                // find the first position on level `level` and then add offset to determine the
                // final index
                std::uint32_t inner_idx = (1 << (BUCKET_BITS - 1 - level)) - 1 + offset;
                tmp_storage.splitters[inner_idx] = sample[i];
            }
        }

        __syncthreads();
    }

    /** Find bucket which contains @p value
     *
     * @param value Searched value
     * @return bucket which contains @p value
     */
    __device__ __forceinline__ bucket_t bucket(key_t value) const
    {
        bucket_t tree_idx = 0;

#pragma unroll
        for (std::size_t level = 0; level < BUCKET_BITS; ++level)
        {
            tree_idx = 2 * tree_idx + (value < tmp_storage.splitters[tree_idx] ? 1 : 2);
        }

        return tree_idx - (BUCKET_COUNT - 1);
    }

    /** Find buckets for each value in @p values
     *
     * @param[in] values Register array of values
     * @param[out] buckets Bucket index for each value
     */
    template <std::size_t ITEMS_PER_THREAD>
    __device__ __forceinline__ void bucket(key_t (&values)[ITEMS_PER_THREAD],
                                           bucket_t (&buckets)[ITEMS_PER_THREAD]) const
    {
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buckets[i] = 0;
        }

#pragma unroll
        for (std::size_t level = 0; level < BUCKET_BITS; ++level)
        {
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                buckets[i] =
                    2 * buckets[i] + (values[i] < tmp_storage.splitters[buckets[i]] ? 1 : 2);
            }
        }

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buckets[i] -= static_cast<bucket_t>(BUCKET_COUNT - 1);
        }
    }
};

#endif // SPLITTER_CUH_
