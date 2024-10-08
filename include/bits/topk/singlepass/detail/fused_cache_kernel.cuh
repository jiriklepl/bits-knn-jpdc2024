#ifndef FUSED_KERNEL_CACHE_CUH_
#define FUSED_KERNEL_CACHE_CUH_

#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

#include "bits/array_view.hpp"
#include "bits/knn.hpp"
#include "bits/topk/singlepass/fused_cache_kernel.hpp"

#include "bits/ptx_utils.cuh"
#include "bits/topk/bitonic_sort_regs.cuh"
#include "bits/topk/singlepass/fused_cache_kernel_structs.cuh"

/** Fused cache kernel function declaration.
 */
#define DECL_FC_KERNEL(query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k)     \
    template void launch_fused_cache<fused_cache_kernel<                                           \
        query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k>>(                  \
        fused_cache_kernel<query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult,    \
                           k>,                                                                     \
        dim3, dim3)

template <class Kernel>
__global__ void __launch_bounds__(Kernel::BLOCK_SIZE, Kernel::MIN_BLOCKS_PER_SM)
    launch_fused_cache(Kernel kernel)
{
    extern __shared__ std::uint8_t shm[];

    kernel.set_tmp_storage(reinterpret_cast<typename Kernel::tmp_storage_t*>(shm));
    kernel.run();
}

template <class Kernel>
void launch_fused_cache(Kernel kernel, dim3 grid, dim3 block)
{
    launch_fused_cache<Kernel><<<grid, block, sizeof(typename Kernel::tmp_storage_t)>>>(kernel);
}

#endif // FUSED_KERNEL_CACHE_CUH_
