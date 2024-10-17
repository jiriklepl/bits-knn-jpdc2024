#ifndef FUSED_KERNEL_RUNNER_CUH_
#define FUSED_KERNEL_RUNNER_CUH_

#include <array>
#include <cstddef>
#include <cstdint>

#include "bits/array_view.hpp"

/** Run the fused kernel.
 */
struct fused_kernel_runner
{
    /** Matrix of the database vectors on the GPU.
     */
    array_view<float, 2> points;

    /** Matrix of the query vectors on the GPU.
     */
    array_view<float, 2> queries;

    /** Top k distances for each query.
     */
    array_view<float, 2> out_dist;

    /** Top k labels for each query.
     */
    array_view<std::int32_t, 2> out_label;

    /** Size of the output.
     */
    std::size_t k;

    /** Number of threads in each thread block.
     */
    std::size_t block_size;

    /** Items per thread from command line.
     *
     * The first item is the number of registers for queries. The second item is the number of
     * registers for database vectors.
     */
    std::array<std::size_t, 3> items_per_thread;

    /** Run the fused kernel.
     *
     * @tparam K the size of the output.
     * @tparam REG_QUERY_COUNT number of registers for queries.
     * @tparam REG_POINT_COUNT number of registers for database vectors.
     * @tparam BLOCK_QUERY_DIM size of each thread block along the query dimension. All thread
     * blocks have 128 threads in total.
     */
    template <std::int32_t K, std::int32_t REG_QUERY_COUNT, std::int32_t REG_POINT_COUNT,
              std::int32_t BLOCK_QUERY_DIM>
    void operator()();
};

#endif // FUSED_KERNEL_RUNNER_CUH_
