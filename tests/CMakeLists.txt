set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)

target_include_directories(test PRIVATE
    ${TEST_DIR}
    ${TEST_DIR}/distance
    ${TEST_DIR}/topk
)

set(TESTS
    distance/baseline_dist_test.cpp
    distance/eigen_dist_test.cpp
    distance/tiled_distance_test.cpp
    distance/magma_dist_test.cpp
    distance/cutlass_dist_test.cpp
    distance/cublas_dist_test.cpp
    topk/partial_bitonic_test.cpp
    topk/bits_knn_test.cpp
    topk/fused_knn_test.cpp
    topk/fused_cache_kernel_test.cu
    topk/cub_knn_test.cpp
    topk/memory_test.cu
    topk/bitonic_sorter_test.cu
    topk/parallel_knn_test.cpp
    array_view_test.cpp
    bitonic_sort_regs_test.cu
)

target_sources(test PRIVATE ${TESTS})
