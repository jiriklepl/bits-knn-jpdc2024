#ifndef BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_HPP
#define BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_HPP

#include <cuda_runtime.h>

/** Fused Euclidean distance computation kernel and top k selection kernel which does not load the
 * whole vectors to shared memory.
 *
 * Input matrices are split into tiles. The tiles do not necessarily cover all dimensions like in
 * the `fused_regs_kernel`. Query vectors are thus loaded from global memory several times, but
 * these requests can usually be satisfied from the cache.
 *
 * The memory management is inspired by the MAGMA GEMM kernel. In each iteration, we load the next
 * tile to registers and then we run the distance computation on data previously loaded to shared
 * memory. Smaller sub-tiles are loaded from shared memory to registers. The core computation works
 * with these register sub-tiles.
 *
 * The final computation computes the dot product between all query and DB vectors. It also
 * computes the norms of the DB vectors. When we have the dot products and norms computed, the
 * kernel finishes the distances by computing `norm(x)^2 - 2 * dot(x, y)` where `x` is a DB vector
 * and `y` is a query vector (i.e., it does not compute actual Euclidean distances; however, simply
 * adding `norm(y)^2` to the computed quantity produces squared Euclidean distances). This
 * modification possibly uses less registers and has less FFMA instructions in the critical loop.
 * It does not affect the order the top k results.
 *
 * @tparam QUERY_REG number of registers along the query dimension.
 * @tparam DB_REG number of registers along the DB dimension.
 * @tparam DIM_REG number of registers for vector dimension.
 * @tparam BLOCK_QUERY_DIM thread block size along the query dimension.
 * @tparam BLOCK_DB_DIM thread block size along the DB dimension.
 * @tparam DIM_MULT `DIM_REG` will be multiplied by this constant to determine the number of
 * vector dimensions loaded in shared memory for each tile.
 * @tparam K size of the output (must be a power-of-two).
 */
template <class Kernel>
extern void launch_fused_cache(Kernel kernel, dim3 grid, dim3 block);

#endif // BITS_TOPK_SINGLEPASS_FUSED_KERNEL_CACHE_HPP
