#ifndef BITS_TOPK_SINGLEPASS_FUSED_TC_KERNEL_RUNNER_HPP_
#define BITS_TOPK_SINGLEPASS_FUSED_TC_KERNEL_RUNNER_HPP_

#include <cstddef>
#include <cstdint>

#include "bits/array_view.hpp"

/** Run the fused kernel.
 */
template <typename Policy>
struct fused_tc_kernel_runner
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

    array_view<typename Policy::input_t, 2> in_points;
    array_view<typename Policy::input_t, 2> in_queries;

    array_view<typename Policy::input_t, 2> in_point_residues;
    array_view<typename Policy::input_t, 2> in_query_residues;

    array_view<float, 1> in_point_norms;
    array_view<float, 1> in_query_norms;

    /** Size of the output.
     */
    std::size_t k;

    /** Number of threads in each thread block.
     */
    std::size_t block_size;

    /** TODO: Add description.
     */
    template <std::int32_t K, std::int32_t BLOCK_SIZE>
    void operator()();
};

#endif // BITS_TOPK_SINGLEPASS_FUSED_TC_KERNEL_RUNNER_HPP_
