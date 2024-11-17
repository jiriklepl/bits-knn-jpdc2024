#ifndef MEMORY_CUH_
#define MEMORY_CUH_

#include <cstdint>
#include <type_traits>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

/** this performs a swap of two values of trivially copyable types (may be faster than std::swap in unobvious
 * cases)
 *
 * @tparam T the type of the first parameter
 * @tparam U the type of the second parameter
 &
 * @param a the first value being swapped
 * @param b the second value being swapped
 */
template <typename T, typename U>
constexpr __forceinline__
    __host__ __device__ std::enable_if_t<std::is_trivially_copy_assignable_v<T> &&
                                         std::is_trivially_copy_assignable_v<U>>
    swap_values(T& a, U& b) noexcept
{
    const auto tmp = a;
    a = b;
    b = tmp;
}

/** Load striped data from @p mem_values to @p values
 *
 * For values in memory: `m_0, m_1, m_2, m_3, ...` it loads
 * `m_{tid}, m_{tid + BLOCK_SIZE}, m_{tid + BLOCK_SIZE * 2}, ..., m_{tid + BLOCK_SIZE *
 * (ITEMS_PER_THREAD - 1)}` to registers @p values
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers for the values
 * @param mem_values Pointer to memory from which this method should load the data
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void load_striped(T (&values)[ITEMS_PER_THREAD], const T* mem_values)
{
    const auto block = cooperative_groups::this_thread_block();

#pragma unroll
    for (std::uint32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        values[i] = mem_values[block.thread_rank() + BLOCK_SIZE * i];
    }
}

/** Load striped data from @p mem_values to @p values
 *
 * For values in memory: `m_0, m_1, m_2, m_3, ...` it loads
 * `m_{tid}, m_{tid + BLOCK_SIZE}, m_{tid + BLOCK_SIZE * 2}, ..., m_{tid + BLOCK_SIZE *
 * (ITEMS_PER_THREAD - 1)}` to registers @p values
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers for the values
 * @param mem_values Pointer to memory from which this method should load the data
 * @param value_count Total number of values in memory
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void load_striped(T (&values)[ITEMS_PER_THREAD], const T* mem_values,
                                    std::size_t value_count)
{
    const auto block = cooperative_groups::this_thread_block();

#pragma unroll
    for (std::uint32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = block.thread_rank() + BLOCK_SIZE * i;
        if (idx < value_count)
        {
            values[i] = mem_values[idx];
        }
    }
}

/** Store striped data from @p values to @p mem_values
 *
 * For values in memory: `m_0, m_1, m_2, m_3, ...` it sets
 * `m_{tid}, m_{tid + BLOCK_SIZE}, m_{tid + BLOCK_SIZE * 2}, ..., m_{tid + BLOCK_SIZE *
 * (ITEMS_PER_THREAD - 1)}` to consecutive values from @p values
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers with the values to store
 * @param mem_values Pointer to memory to which the data will be written
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void store_striped(T (&values)[ITEMS_PER_THREAD], T* mem_values)
{
    const auto block = cooperative_groups::this_thread_block();

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        mem_values[block.thread_rank() + BLOCK_SIZE * i] = values[i];
    }
}

/** Store striped data from @p values to @p mem_values with a check
 *
 * For values in memory: `m_0, m_1, m_2, m_3, ...` it sets
 * `m_{tid}, m_{tid + BLOCK_SIZE}, m_{tid + BLOCK_SIZE * 2}, ..., m_{tid + BLOCK_SIZE *
 * (ITEMS_PER_THREAD - 1)}` to consecutive values from @p values
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers with the values to store
 * @param mem_values Pointer to memory to which the data will be written
 * @param value_count Total number of values in memory
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void store_striped(T (&values)[ITEMS_PER_THREAD], T* mem_values,
                                     std::size_t value_count)
{
    const auto block = cooperative_groups::this_thread_block();

#pragma unroll
    for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        const auto idx = block.thread_rank() + BLOCK_SIZE * i;
        if (idx < value_count)
        {
            mem_values[idx] = values[i];
        }
    }
}

/** Given values in @p values in striped arrangement, transpose them to blocked arrangement
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Values in strided arrangement
 * @param shm_values Auxiliary shared memory for the transposition
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void transpose_to_blocked(T (&values)[ITEMS_PER_THREAD], T* shm_values)
{
    const auto block = cooperative_groups::this_thread_block();

// store the values to shared memory
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        shm_values[block.thread_rank() + BLOCK_SIZE * i] = values[i];
    }

    __syncthreads();

// load the values from shared memory in blocked arrangement
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        values[i] = shm_values[block.thread_rank() * ITEMS_PER_THREAD + i];
    }

    __syncthreads();
}

/** Given values in @p values in blocked arrangement, transpose them to striped arrangement
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Values in blocked arrangement
 * @param shm_values Auxiliary shared memory for the transposition
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void transpose_to_striped(T (&values)[ITEMS_PER_THREAD], T* shm_values)
{
    const auto block = cooperative_groups::this_thread_block();

// store the values to shared memory
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        shm_values[block.thread_rank() * ITEMS_PER_THREAD + i] = values[i];
    }

    __syncthreads();

// load the values from shared memory in blocked arrangement
#pragma unroll
    for (std::int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    {
        values[i] = shm_values[block.thread_rank() + BLOCK_SIZE * i];
    }

    __syncthreads();
}

/** Load values from @p mem_values in striped arrangement and transpose them to blocked arrangement
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers for the loaded values
 * @param mem_values Memory from which this thread should load its values (i.e.,  this thread
 *                   loads `mem_values[0], mem_values[BLOCK_SIZE], mem_values[2 * BLOCK_SIZE] ...`)
 * @param shm_values Auxiliary shared memory for the transposition
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void load_blocked(T (&values)[ITEMS_PER_THREAD], T* mem_values, T* shm_values)
{
    // load striped data from memory
    load_striped<BLOCK_SIZE>(values, mem_values);
    // transpose the data to blocked arrangement
    transpose_to_blocked<BLOCK_SIZE>(values, shm_values);
}

/** Load values from @p mem_values in striped arrangement and transpose them to blocked arrangement
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers for the loaded values
 * @param mem_values Memory from which this thread should load its values (i.e.,  this thread
 *                   loads `mem_values[0], mem_values[BLOCK_SIZE], mem_values[2 * BLOCK_SIZE] ...`)
 * @param shm_values Auxiliary shared memory for the transposition
 * @param value_count Total number of values in memory
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void load_blocked(T (&values)[ITEMS_PER_THREAD], T* mem_values, T* shm_values,
                                    std::size_t value_count)
{
    // load striped data from memory
    load_striped<BLOCK_SIZE>(values, mem_values, value_count);
    // transpose the data to blocked arrangement
    transpose_to_blocked<BLOCK_SIZE>(values, shm_values);
}

/** Store values from @p values in blocked arrangement to @p mem_values
 *
 * @tparam BLOCK_SIZE thread block size (number of threads)
 * @tparam ITEMS_PER_THREAD number of registers per thread used to hold data
 * @tparam T type of the data elements
 *
 * @param values Registers with the data in blocked arrangement
 * @param mem_values Destination memory for the data (i.e.,  this thread writes values to
 *                   `mem_values[0], mem_values[BLOCK_SIZE], mem_values[2 * BLOCK_SIZE] ...`)
 * @param shm_values Auxiliary shared memory for data transposition
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
inline __device__ void store_blocked(T (&values)[ITEMS_PER_THREAD], T* mem_values, T* shm_values)
{
    // transpose the data to striped arrangement
    transpose_to_striped<BLOCK_SIZE>(values, shm_values);
    // store the striped data
    store_striped<BLOCK_SIZE>(values, mem_values);
}

/** Copy a range of values [ @p in_begin , @p in_end ) to @p out_ptr
 *
 * @tparam BLOCK_SIZE number of threads in a thread block
 * @tparam ITEMS_PER_THREAD number of values per thread to load to registers
 * @tparam T type of the values
 *
 * @param in_begin Input begin pointer
 * @param in_end Input end pointer
 * @param out_ptr Output pointer
 */
template <std::size_t BLOCK_SIZE, std::size_t ITEMS_PER_THREAD, typename T>
__forceinline__ __device__ void copy(const T* in_begin, const T* in_end, T* out_ptr)
{
    constexpr std::size_t TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;
    for (; in_begin + TILE_SIZE < in_end; in_begin += TILE_SIZE)
    {
        T values[ITEMS_PER_THREAD];
        load_striped<BLOCK_SIZE>(values, in_begin);

        __syncthreads();

        store_striped<BLOCK_SIZE>(values, out_ptr);
        out_ptr += TILE_SIZE;
    }

    if (in_begin < in_end)
    {
        const auto value_count = in_end - in_begin;

        T values[ITEMS_PER_THREAD];
        load_striped<BLOCK_SIZE>(values, in_begin, value_count);

        __syncthreads();

        store_striped<BLOCK_SIZE>(values, out_ptr, value_count);
    }
}

#endif // MEMORY_CUH_
