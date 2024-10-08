#ifndef HISTOGRAM_KERNEL_CUH_
#define HISTOGRAM_KERNEL_CUH_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "bits/topk/multipass/histogram.cuh"
#include "bits/topk/multipass/radix_splitter.cuh"
#include "bits/topk/multipass/splitter.cuh"

template <std::size_t BLOCK_SIZE_, std::size_t ITEMS_PER_THREAD, std::size_t BIN_BITS,
          std::size_t TILES_PER_BLOCK, class Splitter>
struct histogram_kernel
{
    // number of threads in each thread block
    inline static constexpr std::size_t BLOCK_SIZE = BLOCK_SIZE_;
    // number of items per thread block
    inline static constexpr std::size_t ITEMS_PER_BLOCK =
        BLOCK_SIZE * ITEMS_PER_THREAD * TILES_PER_BLOCK;
    // size of the histogram
    inline static constexpr std::size_t HIST_SIZE = 1 << BIN_BITS;

    using splitter_t = Splitter;
    using histogram_t = histogram<BLOCK_SIZE, ITEMS_PER_THREAD, BIN_BITS>;

    struct tmp_storage_t
    {
        typename histogram_t::tmp_storage_t histogram;
        typename splitter_t::tmp_storage_t splitter;
    };

    tmp_storage_t* tmp_storage;

    // range with input distances
    float* global_dist;
    // table of privatized histograms in global memory
    std::int32_t* global_hist;
    // size of the input
    std::int32_t input_size;
    // the first bit in each key
    typename splitter_t::state_t splitter_state;

    __device__ __forceinline__ void set_tmp_storage(tmp_storage_t& shm) { tmp_storage = &shm; }

    /** Range of the input buffer of distances assigned to this thread block
     *
     * @return range of input buffer assigned to this thread block
     */
    __device__ __forceinline__ std::pair<float*, float*> bucket()
    {
        std::size_t begin = std::min<std::size_t>(blockIdx.x * ITEMS_PER_BLOCK, input_size);
        std::size_t end = std::min<std::size_t>(begin + ITEMS_PER_BLOCK, input_size);
        return {global_dist + begin, global_dist + end};
    }

    /** Compute a privatized histogram of the assigned segment of input.
     */
    __device__ __forceinline__ void operator()()
    {
        // initialize distance-to-bucket splitter
        splitter_t splitter{tmp_storage->splitter};
        splitter.load(splitter_state);

        // compute histogram of the assigned segment of distances
        const auto [begin, end] = bucket();
        histogram_t histogram{tmp_storage->histogram};
        histogram.reset();
        histogram.process(begin, 0, end - begin, splitter);

        // write the local histogram to the histogram table in a striped layout
        for (std::size_t i = threadIdx.x; i < histogram_t::BUCKET_COUNT; i += BLOCK_SIZE)
        {
            global_hist[i * gridDim.x + blockIdx.x] = histogram.tmp_storage.counter[i];
        }
    }
};

#endif // HISTOGRAM_KERNEL_CUH_
