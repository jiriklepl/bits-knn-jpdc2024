#ifndef CUDA_ARRAY_HPP_
#define CUDA_ARRAY_HPP_

#include <cassert>
#include <cstddef>

#include "bits/array_view.hpp"
#include "bits/cuda_ptr.hpp"

// use this tag in the cuda_array constructor to tell it to use pitched allocation
class pitched_allocation
{
};

/** `DIM` dimensional array in GPU memory space.
 *
 * @tparam T type of the values.
 * @tparam DIM dimension of the array.
 */
template <typename T, std::size_t DIM>
class cuda_array
{
public:
    static_assert(DIM > 0, "Dimension of cuda_array must not be 0");

    using view_t = array_view<T, DIM>;

    cuda_array() = default;

    /** Allocate a new 2D array of size @p size using pitched allocation
     *
     * @param size Dimensions of the array
     */
    cuda_array(const std::size_t (&size)[DIM], pitched_allocation)
    {
        static_assert(DIM == 2, "Pitched allocation can only be used for 2D arrays.");

        void* ptr = nullptr;
        std::size_t pitch = 0;
        CUCH(cudaMallocPitch(&ptr, &pitch, size[1] * sizeof(T), size[0] * sizeof(T)));
        CUCH(cudaMemset2D(ptr, pitch, 0, size[1] * sizeof(T), size[0] * sizeof(T)));
        data_ = cuda_ptr<T>(static_cast<T*>(ptr));
        view_ = view_t{data_.get(), size, {pitch / sizeof(T), 1}};
    }

    /** Allocate a new 2D array of size @p size
     *
     * @param size Dimensions of the array
     */
    explicit cuda_array(const std::size_t (&size)[DIM])
    {
        // compute the total size
        auto total_size = size[0];
        for (std::size_t i = 1; i < DIM; ++i)
        {
            total_size *= size[i];
        }

        // create tight strides
        std::size_t stride[DIM];
        for (std::size_t i = 0; i < DIM - 1; ++i)
        {
            stride[i] = size[i + 1];
        }
        stride[DIM - 1] = 1;

        data_ = make_cuda_ptr<T>(total_size);
        view_ = view_t{data_.get(), size, stride};
    }

    /** Allocate a new 2D array of size @p size with custom padding
     *
     * @param size Dimensions of the array
     * @param stride Strides
     */
    cuda_array(const std::size_t (&size)[DIM], const std::size_t (&stride)[DIM])
    {
        for (std::size_t i = 0; i < DIM - 1; ++i)
        {
            assert(stride[i] >= size[i + 1]);
        }

        // compute the total size
        auto total_size = size[0];
        for (std::size_t i = 0; i < DIM - 1; ++i)
        {
            total_size *= stride[i];
        }

        data_ = make_cuda_ptr<T>(total_size);
        view_ = view_t{data_.get(), size, stride};
    }

    /** Allocate a new region of memory with shape taken from @p view
     *
     * @tparam U type used by @p view
     *
     * @param view memory view from which the shape is copied
     * @return cuda array with the same shape as @p view
     */
    template <typename U>
    inline static cuda_array with_shape(array_view<U, DIM> view)
    {
        std::size_t size[DIM];
        std::size_t stride[DIM];
        for (std::size_t i = 0; i < DIM; ++i)
        {
            size[i] = view.size(i);
            stride[i] = view.stride(i);
        }
        return cuda_array{size, stride};
    }

    /** Free memory allocated by this class
     */
    void release()
    {
        data_.reset(nullptr);
        view_ = view_t{};
    }

    /** Get view of the memory (in the GPU memory space)
     *
     * @returns view of the memory
     */
    view_t view() const { return view_; }

private:
    cuda_ptr<T> data_;
    view_t view_;
};

#endif // CUDA_ARRAY_HPP_
