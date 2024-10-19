set(TOPK_SINGLEPASS_SRC
    bits_knn.cu
    cub_knn.cu
    fused_cache_knn.cu
    fused_knn.cu
    fused_tc_knn.cu
    grid_select.cu
    partial_bitonic_buffered.cu
    partial_bitonic.cu
    rapidsai_fused.cu
    warp_select.cu
)

target_include_directories(topk-singlepass-all PRIVATE ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/include/)
target_include_directories(topk-singlepass-minimal PRIVATE ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/include/)
# target_include_directories(topk-singlepass-test PRIVATE ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/include/)

target_link_directories(topk-singlepass-all PUBLIC ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/third_party/)
target_link_directories(topk-singlepass-minimal PUBLIC ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/third_party/)
# target_link_directories(topk-singlepass-test PUBLIC ${CMAKE_SOURCE_DIR}/external/gpu_topK_benchmark/third_party/)

target_sources(topk-singlepass-all PRIVATE ${TOPK_SINGLEPASS_SRC})
target_sources(topk-singlepass-minimal PRIVATE ${TOPK_SINGLEPASS_SRC})
# target_sources(topk-singlepass-test PRIVATE ${TOPK_SINGLEPASS_SRC})

target_link_libraries(topk-singlepass-all PRIVATE raft::raft gridselect)
target_link_libraries(topk-singlepass-minimal PRIVATE raft::raft gridselect)
# target_link_libraries(topk-singlepass-test PRIVATE raft::raft gridselect)
