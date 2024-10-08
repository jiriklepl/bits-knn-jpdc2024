#ifndef BUFFERED_FILTER_CUH_
#define BUFFERED_FILTER_CUH_

#include <cstdint>

#include <cub/block/block_scan.cuh>

#include "bits/memory.cuh"

/** No value.
 */
struct null_value_t
{
};

/** Buffer of key/value pairs.
 *
 * @tparam BUFFER_SIZE maximal buffer size.
 * @tparam Key type of the keys in the buffer.
 * @tparam Value type of the values in the buffer.
 */
template <std::size_t BUFFER_SIZE, typename Key, typename Value>
struct buffer
{
    /** Keys in the buffer.
     */
    Key keys[BUFFER_SIZE];

    /** Values in the buffer.
     */
    Value values[BUFFER_SIZE];

    /** Actual size of the buffer.
     */
    std::int32_t size;
};

/** Buffer of keys.
 *
 * @tparam BUFFER_SIZE maximal buffer size.
 * @tparam Key type of the keys.
 */
template <std::size_t BUFFER_SIZE, typename Key>
struct buffer<BUFFER_SIZE, Key, null_value_t>
{
    Key keys[BUFFER_SIZE];
    std::int32_t size;
};

/** Pointer to a key/value buffer in memory.
 *
 * @tparam Key type of the keys.
 * @tparam Value type of the values.
 */
template <typename Key, typename Value>
struct buffer_ptr
{
    Key* key_ptr;
    Value* value_ptr;
};

/** Pointer to a key buffer in memory.
 *
 * @tparam Key type of the keys.
 */
template <typename Key>
struct buffer_ptr<Key, null_value_t>
{
    Key* key_ptr;
};

/** Register array of key/value pairs.
 *
 * @tparam ITEMS_PER_THREAD size of the array.
 * @tparam Key type of the keys.
 * @tparam Value type of the values.
 */
template <std::size_t ITEMS_PER_THREAD, typename Key, typename Value>
struct reg_array
{
    Key keys[ITEMS_PER_THREAD];
    Value values[ITEMS_PER_THREAD];
};

/** Register array of keys.
 *
 * @tparam ITEMS_PER_THREAD size of the array.
 * @tparam Key type of the keys.
 */
template <std::size_t ITEMS_PER_THREAD, typename Key>
struct reg_array<ITEMS_PER_THREAD, Key, null_value_t>
{
    Key keys[ITEMS_PER_THREAD];
};

/** Algorithm used to add values to a buffer.
 */
enum alloc_algorithm
{
    /** Use `atomicAdd` to allocate buffer spots.
     */
    ALLOC_DIRECT,

    /** Use a block-wide prefix sum to allocate buffer spots.
     */
    ALLOC_RANK
};

/** Filter values which satisfy a predicate using a shared memory buffer to coalesce global memory
 * writes.
 *
 * @tparam BLOCK_SIZE number of threads in each thread block.
 * @tparam ITEMS_PER_THREAD number of registers per thread.
 * @tparam BUFFER_SIZE size of the buffer in shared memory.
 * @tparam Key type of the keys.
 * @tparam Value type of the values (can be `null_value_t`)
 * @tparam ALLOC algorithm used to add values to the shared memory buffer.
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, std::size_t BUFFER_SIZE,
          typename Key, typename Value, alloc_algorithm ALLOC = ALLOC_RANK>
struct buffered_filter
{
public:
    using key_t = Key;
    using value_t = Value;

    using reg_array_t = reg_array<ITEMS_PER_THREAD, Key, Value>;
    using buffer_t = buffer<BUFFER_SIZE, Key, Value>;
    using buffer_ptr_t = buffer_ptr<Key, Value>;
    using scan_t = cub::BlockScan<std::int16_t, BLOCK_SIZE, cub::BLOCK_SCAN_RAKING>;

    /** Shared memory storage used by this data structure.
     */
    struct tmp_storage_t
    {
        buffer_t buffer;
        typename scan_t::TempStorage scan;
    };

    // shared storage
    tmp_storage_t& tmp_storage;

    /** Create buffered filter
     *
     * @param storage shared memory storage
     */
    __device__ __forceinline__ explicit buffered_filter(tmp_storage_t& storage)
        : tmp_storage(storage)
    {
    }

    /** Reset buffer size to 0
     */
    __device__ __forceinline__ void reset()
    {
        tmp_storage.buffer.size = 0;

        __syncthreads();
    }

    /** Allocate buffer spots using `atomicAdd` on buffer size.
     *
     * @tparam Predicate type of predicate applied to keys (function which given a key returns a
     * bool)
     *
     * @param tile register array of loaded keys/values (in striped arrangement)
     * @param value_count number of valid values
     * @param predicate predicate applied to all keys in @p tile
     * @param[out] buffer_pos Index in the buffer. Value `-1` indicates no spot (the caller should
     * not add the value to the buffer). Values can be greater than or equal to `BUFFER_SIZE` in
     * which case the caller should insert values to buffer in multiple iterations calling
     * flush_buffer() after each iteration.
     */
    template <class Predicate>
    __device__ __forceinline__ void allocate_direct(reg_array_t& tile, std::int32_t value_count,
                                                    std::int16_t (&buffer_pos)[ITEMS_PER_THREAD],
                                                    Predicate& predicate)
    {
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buffer_pos[i] = -1;

            const auto idx = i * BLOCK_SIZE + threadIdx.x;
            if (idx < value_count && predicate(tile.keys[i]))
            {
                buffer_pos[i] = atomicAdd(&tmp_storage.buffer.size, 1);
            }
        }
    }

    /** Allocate buffer spots using a block-wide prefix sum.
     *
     * @tparam Predicate type of predicate applied to keys (function which given a key returns a
     * bool)
     *
     * @param tile register array of loaded keys/values (in striped arrangement)
     * @param value_count number of valid values
     * @param predicate predicate applied to all keys in @p tile
     * @param[out] buffer_pos Index in the buffer. Value `-1` indicates no spot (the caller should
     * not add the value to the buffer). Values can be greater than or equal to `BUFFER_SIZE` in
     * which case the caller should insert values to buffer in multiple iterations calling
     * flush_buffer() after each iteration.
     */
    template <class Predicate>
    __device__ __forceinline__ void allocate_rank(reg_array_t& tile, std::int32_t value_count,
                                                  std::int16_t (&buffer_pos)[ITEMS_PER_THREAD],
                                                  Predicate& predicate)
    {
        std::int16_t prefix = 0;

        // find valid values for which the predicate is true
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buffer_pos[i] = 0;

            const auto idx = i * BLOCK_SIZE + threadIdx.x;
            if (idx < value_count && predicate(tile.keys[i]))
            {
                buffer_pos[i] = 1;
                ++prefix;
            }
        }

        // execute prefix sum to find indices allocated to this thread
        scan_t{tmp_storage.scan}.ExclusiveSum(prefix, prefix);
        // broadcast buffer size
        prefix += tmp_storage.buffer.size;

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            if (buffer_pos[i] == 0)
            {
                buffer_pos[i] = -1; // no position (do not insert this value to the buffer)
            }
            else
            {
                buffer_pos[i] = prefix++;
            }
        }

        __syncthreads();

        // the last thread updates the buffer size
        if (threadIdx.x + 1 >= BLOCK_SIZE)
        {
            tmp_storage.buffer.size = prefix;
        }
    }

    template <class Predicate>
    __device__ __forceinline__ void
    allocate_rank_striped(reg_array_t& tile, std::int32_t value_count,
                          std::int16_t (&buffer_pos)[ITEMS_PER_THREAD], Predicate& predicate)
    {
        std::int16_t buffer_size = tmp_storage.buffer.size;

#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            std::int32_t idx = i * BLOCK_SIZE + threadIdx.x;
            std::int16_t indicator = 0;
            if (idx < value_count && predicate(tile.keys[i]))
            {
                indicator = 1;
            }
            const auto prefix = buffer_size;
            scan_t{tmp_storage.scan}.ExclusiveSum(indicator, buffer_pos[i], buffer_size);
            buffer_size += prefix;

            if (indicator == 0)
            {
                buffer_pos[i] = -1;
            }
            else
            {
                buffer_pos[i] += prefix;
            }
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            tmp_storage.buffer.size = buffer_size;
        }
    }

    /** Apply @p predicate to all values in @p tile and add values which pass the test to buffer.
     *
     * @tparam Predicate type of predicate applied to keys
     *
     * @param tile Tile of keys/values in registers
     * @param value_count Number of valid values in the tile (assuming striped arrangement)
     * @param predicate Predicate applied to keys in @p tile
     */
    template <class Predicate>
    __device__ __forceinline__ void process_tile(reg_array_t& tile, std::int32_t value_count,
                                                 Predicate& predicate)
    {
        static_assert(ALLOC == ALLOC_RANK || ALLOC == ALLOC_DIRECT);

        std::int16_t buffer_pos[ITEMS_PER_THREAD];
        if constexpr (ALLOC == ALLOC_RANK)
        {
            allocate_rank(tile, value_count, buffer_pos, predicate);
        }
        else
        {
            allocate_direct(tile, value_count, buffer_pos, predicate);
        }

        for (;;)
        {
            bool overflow = false;

            // add values to buffers
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                // check for buffer overflow
                overflow |= buffer_pos[i] >= static_cast<std::int16_t>(BUFFER_SIZE);

                // store values to the buffer if possible
                if (0 <= buffer_pos[i] && buffer_pos[i] < static_cast<std::int16_t>(BUFFER_SIZE))
                {
                    tmp_storage.buffer.keys[buffer_pos[i]] = tile.keys[i];

                    if constexpr (!std::is_same_v<value_t, null_value_t>)
                    {
                        tmp_storage.buffer.values[buffer_pos[i]] = tile.values[i];
                    }

                    buffer_pos[i] = -1; // this value has already been added to the buffer
                }
            }

            // check whether the buffer has overflown
            if (!__syncthreads_or(overflow))
            {
                break;
            }

            // write buffer keys to global memory
            flush_buffer();

            // decrement buffer position
#pragma unroll
            for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
            {
                buffer_pos[i] -= static_cast<std::int16_t>(BUFFER_SIZE);
            }
        }

        __syncthreads();
    }

    /** Write all values in buffer to `out_ + write_head_`
     *
     * @tparam FLUSH_IF_NOT_FULL if true, this function will write the values to global memory
     * regardless of whether the buffer is full. Otherwise, the values are only written to global
     * memory if the buffer is full.
     */
    template <bool FLUSH_IF_NOT_FULL = false>
    __device__ __forceinline__ void flush_buffer()
    {
        // check for overflow
        auto buffer_size = tmp_storage.buffer.size;
        if (!FLUSH_IF_NOT_FULL && buffer_size < static_cast<std::int32_t>(BUFFER_SIZE))
        {
            return; // do not flush
        }

        // threads can hold more values in registers than `BUFFER_SIZE` but physically
        // there are only BUFFER_SIZE slots
        buffer_size = std::min<std::int32_t>(buffer_size, static_cast<std::int32_t>(BUFFER_SIZE));

        // store the buffer to global memory
        for (std::size_t i = threadIdx.x; i < buffer_size; i += BLOCK_SIZE)
        {
            out_.key_ptr[write_head_ + i] = tmp_storage.buffer.keys[i];

            if constexpr (!std::is_same_v<value_t, null_value_t>)
            {
                out_.value_ptr[write_head_ + i] = tmp_storage.buffer.values[i];
            }
        }

        // move write head after the newly written values
        write_head_ += buffer_size;

        __syncthreads();

        // update buffer size
        if (threadIdx.x == 0)
        {
            tmp_storage.buffer.size -= buffer_size;
        }

        __syncthreads();
    }

    /** Read all key/value pairs in the [ @p begin , @p end ) segment, apply @p predicate to all
     * keys, and store values for which the predicate returns true to @p out
     *
     * @tparam Predicate Predicate functor type
     *
     * @param begin Pointer to the beginning of keys/values
     * @param end Pointer to the end of keys/values
     * @param out Pointer to the output of keys/values; can be set to @p begin
     * @param predicate predicate which when applied to a key returns a bool value
     * @return number of values written to @p out
     */
    template <class Predicate>
    __device__ __forceinline__ std::int32_t process(buffer_ptr_t begin, buffer_ptr_t end,
                                                    buffer_ptr_t out, Predicate&& predicate)
    {
        constexpr std::int32_t TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

        // reset write head
        out_ = out;
        write_head_ = 0;

        std::int32_t idx = 0;
        const std::int32_t size = end.key_ptr - begin.key_ptr;

        for (; idx + TILE_SIZE < size; idx += TILE_SIZE)
        {
            reg_array_t tile;

            // load a tile
            load_striped<BLOCK_SIZE>(tile.keys, begin.key_ptr + idx);
            if constexpr (!std::is_same_v<value_t, null_value_t>)
            {
                load_striped<BLOCK_SIZE>(tile.values, begin.value_ptr + idx);
            }

            __syncthreads();

            // store all values to a buffer
            process_tile(tile, TILE_SIZE, predicate);
        }

        if (idx < size)
        {
            reg_array_t tile;

            // load a tile
            load_striped<BLOCK_SIZE>(tile.keys, begin.key_ptr + idx, size - idx);
            if constexpr (!std::is_same_v<value_t, null_value_t>)
            {
                load_striped<BLOCK_SIZE>(tile.values, begin.value_ptr + idx, size - idx);
            }

            __syncthreads();

            // store all values to a buffer
            process_tile(tile, size - idx, predicate);
        }

        // write all remaining values in buffer to global memory
        constexpr bool FLUSH_IF_NOT_FULL = true;
        flush_buffer<FLUSH_IF_NOT_FULL>();

        return write_head_;
    }

private:
    // base pointer for the output
    buffer_ptr_t out_;
    // current output index
    std::int32_t write_head_;
};

#endif // BUFFERED_FILTER_CUH_
