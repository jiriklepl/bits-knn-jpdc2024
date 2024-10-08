#ifndef PTX_UTILS_CUH__
#define PTX_UTILS_CUH__

#include <cuda_runtime.h>

/** Prefetch a cache line containing @p ptr to L2
 *
 * @param ptr Pointer to global memory
 */
template <typename T>
__device__ __forceinline__ void prefetch(const T* ptr)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

#endif // PTX_UTILS_CUH__
