#ifndef BITS_VERIFY_HPP_
#define BITS_VERIFY_HPP_

#include <cstddef>
#include <vector>

#include "bits/knn.hpp"

/** Verify that two kNN results are the same. This method normalizes the results so that items
 * with equal distance are sorted by their label. No error is reported if the labels are
 * different for the kth element but the distances are equal.
 *
 * @param expected_result Expected result
 * @param actual_result actual result
 * @param k number of nearest neighbors
 * @returns true iff the results are the same up to equal distances
 */
bool verify(const std::vector<knn::pair_t>& expected_result,
            const std::vector<knn::pair_t>& actual_result, std::size_t k);

#endif // BITS_VERIFY_HPP_
