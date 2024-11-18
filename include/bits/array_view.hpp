#ifndef BITS_ARRAY_VIEW_HPP_
#define BITS_ARRAY_VIEW_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_runtime.h>

/** View a contiguous block of memory as a row-major DIM dimensional array.
 *
 * @tparam T type of stored elements
 * @tparam DIM dimension
 */
template <typename T, std::size_t DIM>
class array_view
{
public:
    static_assert(DIM > 0);

    __host__ __device__ array_view() : data_(nullptr), element_count_{0}, stride_{0} {}

    /** Create cuda matrix from a GPU pointer
     *
     * @param ptr Pointer the start of the array
     * @param element_count Number of elements in each dimension
     * @param pitch
     */
    __host__ __device__ array_view(T* ptr, const std::size_t (&element_count)[DIM],
                                   const std::size_t (&stride)[DIM])
        : data_(ptr)
    {
#pragma unroll
        for (std::size_t i = 0; i < DIM; ++i)
        {
            element_count_[i] = element_count[i];
            stride_[i] = stride[i];
        }
    }

    /** Number of elements along ith dimension
     *
     * @param i Index of a dimension [0, DIM)
     *
     * @returns number of elements in the dimension @p i
     */
    __host__ __device__ std::size_t size(std::size_t i) const { return element_count_[i]; }

    /** Get stride for ith dimension
     *
     * @param i Index of a dimension [0, DIM)
     *
     * @returns stride for dimension @p i
     */
    __host__ __device__ std::size_t stride(std::size_t i) const { return stride_[i]; }

    /** Get total number of elements in the array including padding due to bigger strides.
     *
     * @returns total number of elements including padding
     */
    __host__ __device__ std::size_t size() const
    {
        std::size_t total_size = size(0);

#pragma push
#pragma nv_diag_suppress = unsigned_compare_with_zero

#pragma unroll
        for (std::size_t i = 0; i < DIM - 1; ++i)
        {
            total_size *= stride_[i];
        }

#pragma pop

        return total_size;
    }

    __host__ __device__ T& operator()(const std::size_t (&indices)[DIM]) const
    {
        std::size_t index = 0;

#pragma push
#pragma nv_diag_suppress = unsigned_compare_with_zero

#pragma unroll
        for (std::size_t i = 0; i < DIM - 1; ++i)
        {
            index = (index + indices[i]) * stride_[i];
        }
#pragma pop

        return data_[index + indices[DIM - 1]];
    }

    __host__ __device__ T* ptr(std::size_t i, std::size_t j) const
    {
        return data_ + (i * stride_[0] + j);
    }

    __host__ __device__ T& operator()(std::size_t i, std::size_t j) const
    {
        static_assert(DIM == 2);
        return data_[i * stride_[0] + j];
    }

    __host__ __device__ T& operator()(std::size_t i) const
    {
        static_assert(DIM == 1);
        return data_[i];
    }

    /** Get raw pointer to the memory
     *
     * @returns raw pointer to memory
     */
    __host__ __device__ T* data() const { return data_; }

    __host__ __device__ const T* cdata() const { return data_; }

    /** Set the raw pointer to the data
     *
     * @param ptr New raw pointer to the data
     */
    __host__ __device__ void set_data(T* ptr) { data_ = ptr; }

private:
    T* data_;
    std::array<std::size_t, DIM> element_count_;
    std::array<std::size_t, DIM> stride_;
};

// check if array_view is trivially copyable for the types and dimensionalities we use
static_assert(std::is_trivially_copyable_v<array_view<std::int32_t, 1>>);
static_assert(std::is_trivially_copyable_v<array_view<std::int32_t, 2>>);
static_assert(std::is_trivially_copyable_v<array_view<float, 1>>);
static_assert(std::is_trivially_copyable_v<array_view<float, 2>>);

#endif // BITS_ARRAY_VIEW_HPP_
