# Reuse the selection implementations with a bounded set of BITS configurations.
# Keep this target independent of the unrelated fused and distance benchmarks.
add_library(application-selectors STATIC EXCLUDE_FROM_ALL
    ${SRC_DIR}/knn.cpp
    ${SRC_DIR}/cuda_knn.cpp
    ${SRC_DIR}/cuda_stream.cu
    ${SRC_DIR}/distance/cuda_distance.cpp
    ${SRC_DIR}/distance/magma_distance.cu
    ${SRC_DIR}/topk/singlepass/bits_knn.cu
    ${SRC_DIR}/topk/singlepass/grid_select.cu
    ${SRC_DIR}/topk/singlepass/warp_select.cu
    ${SRC_DIR}/topk/multipass/air_topk.cu)
foreach(k 16 32 64 128 256 512 1024 2048)
    foreach(block 128 256 512)
        foreach(batch 1 4 7 8 13 16)
            target_sources(application-selectors PRIVATE
                ${SRC_DIR}/topk/singlepass/detail/bts_${k}_${batch}_${block}.cu)
        endforeach()
    endforeach()
    target_sources(application-selectors PRIVATE ${SRC_DIR}/topk/detail/warp_select${k}.cu)
    if(k LESS_EQUAL 1024)
        target_sources(application-selectors PRIVATE ${SRC_DIR}/topk/detail/block_select${k}.cu)
    endif()
endforeach()
target_compile_definitions(application-selectors PRIVATE TOPK_SINGLEPASS_USE_MINIMAL)
# This property is local to this directory's targets; the kNN builds keep their
# existing full/minimal dispatch tables.
set_source_files_properties(${SRC_DIR}/topk/singlepass/bits_knn.cu PROPERTIES
    COMPILE_DEFINITIONS TOPK_SINGLEPASS_USE_APPLICATIONS)
target_link_directories(application-selectors PUBLIC ${LIBS_DIR}/gpu_topK_benchmark/third_party)
target_link_libraries(application-selectors PUBLIC CUDA::cudart raft::raft gridselect)

add_library(application-operators STATIC EXCLUDE_FROM_ALL
    database_topn.cu database_reference.cpp
    token_sampling.cu sampling_reference.cpp
    gradient_compression.cu gradient_reference.cpp)
target_link_libraries(application-operators PUBLIC CUDA::cudart)

add_executable(database-topn EXCLUDE_FROM_ALL main.cpp)
add_executable(token-sampling EXCLUDE_FROM_ALL tensor_main.cpp)
target_compile_definitions(token-sampling PRIVATE APPLICATION_TOKEN_SAMPLING)
add_executable(gradient-compression EXCLUDE_FROM_ALL tensor_main.cpp)
add_executable(test-applications EXCLUDE_FROM_ALL
    ${TEST_DIR}/applications/database_topn_test.cpp
    ${TEST_DIR}/applications/token_sampling_test.cpp
    ${TEST_DIR}/applications/gradient_compression_test.cpp)
target_link_libraries(test-applications PRIVATE Catch2::Catch2WithMain)
foreach(target database-topn token-sampling gradient-compression test-applications)
    target_link_libraries(${target} PRIVATE application-operators application-selectors)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endforeach()
