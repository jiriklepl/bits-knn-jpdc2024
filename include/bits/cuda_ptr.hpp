#ifndef BITS_CUDA_PTR_HPP_
#define BITS_CUDA_PTR_HPP_

#include <cstddef>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include "cuch.hpp"

/** Deleter functor which uses cudaFree.
 *
 * @tparam T type of the values to delete.
 */
template <typename T>
class cuda_deleter
{
public:
    void operator()(T* ptr) const { CUCH(cudaFree(ptr)); }
};

/** Unique pointer to a GPU memory space.
 *
 * @tparam T type of the values the pointer points to.
 */
template <typename T>
using cuda_ptr = std::unique_ptr<T, cuda_deleter<T>>;

/** Allocate array of size @p count on GPU using cudaMalloc
 *
 * @param count Number of elements in the array
 */
template <typename T>
inline cuda_ptr<T> make_cuda_ptr(std::size_t count)
{
    T* ptr = nullptr;
    CUCH(cudaMalloc((void**)&ptr, count * sizeof(T)));
    return cuda_ptr<T>{ptr};
}

/** Allocate a 2D block of memory on the GPU using cudaMallocPitch
 *
 * @p width, @p height and @p pitch are number of elements of size sizeof(T)
 */
template <typename T>
inline std::pair<cuda_ptr<T>, std::size_t> make_cuda_ptr(std::size_t width, std::size_t height)
{
    std::size_t pitch = 0;
    T* ptr = nullptr;
    CUCH(cudaMallocPitch((void**)&ptr, &pitch, width * sizeof(T), height * sizeof(T)));
    return std::make_pair(cuda_ptr<T>{ptr}, pitch / sizeof(T));
}

#endif // BITS_CUDA_PTR_HPP_
