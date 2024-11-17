#ifndef RADIX_SPLITTER_CUH_
#define RADIX_SPLITTER_CUH_

#include <cstddef>
#include <cstdint>
#include <type_traits>

/** Split distances according to a radix digit.
 *
 * @tparam RADIX_BITS number of radix bits of each digit.
 */
template <std::size_t RADIX_BITS>
struct radix_splitter
{
    /** Shared memory used by this component (empty) to conform to the splitter interface.
     */
    struct tmp_storage_t
    {
    };

    /** State of the splitter.
     */
    struct state_t
    {
        /** The first bit in the [0, 32) range to consider.
         */
        std::int32_t begin_bit;
    };

    using bucket_t = std::conditional_t<RADIX_BITS <= 8, std::uint8_t, std::uint16_t>;

    static constexpr std::uint32_t DIGIT_MASK = (1 << RADIX_BITS) - 1;

    std::uint32_t begin_bit;

    /** Create a radix splitter.
     *
     * @param begin_bit the first bit in the [0, 32) range to consider.
     */
    __device__ __forceinline__ explicit radix_splitter(std::uint32_t begin_bit)
        : begin_bit(begin_bit)
    {
    }

    /** Empty constructor to conform to the splitter interface.
     */
    __device__ __forceinline__ explicit radix_splitter(tmp_storage_t&) {}

    /** Set the begin bit to @p state
     *
     * @param state the next bit to consider in keys
     */
    __device__ __forceinline__ void load(state_t state) { begin_bit = state.begin_bit; }

    /** No-op
     */
    __device__ __forceinline__ void store(state_t) {}

    /** Transform a floating point @p value to an unsigned 32-bit integer s.t. `a < b` iff
     * `key(a) < key(b)`.
     *
     * @param value a floating point value
     * @return unsigned 32-bit integer that represents @p value and can be sorted by its bits.
     */
    __device__ __forceinline__ std::uint32_t key(float value) const
    {
        std::uint32_t mask = value < 0 ? 0xFFFFFFFF : 0x80000000;
        return *reinterpret_cast<std::uint32_t*>(&value) ^ mask;
    }

    /** Extract the radix digit of @p value
     *
     * @param value a floating point value
     * @return radix digit of @p value
     */
    __device__ __forceinline__ bucket_t extract_digit(float value) const
    {
        return (key(value) >> begin_bit) & DIGIT_MASK;
    }

    /** Find bucket which contains @p value
     *
     * @param value Searched value
     * @return bucket which contains @p value
     */
    __device__ __forceinline__ bucket_t bucket(float value) const { return extract_digit(value); }

    /** Find buckets for each value in @p values
     *
     * @param[in] values Register array of values
     * @param[out] buckets Bucket index for each value
     */
    template <std::size_t ITEMS_PER_THREAD>
    __device__ __forceinline__ void bucket(float (&values)[ITEMS_PER_THREAD],
                                           bucket_t (&buckets)[ITEMS_PER_THREAD]) const
    {
#pragma unroll
        for (std::size_t i = 0; i < ITEMS_PER_THREAD; ++i)
        {
            buckets[i] = extract_digit(values[i]);
        }
    }
};

#endif // RADIX_SPLITTER_CUH_
