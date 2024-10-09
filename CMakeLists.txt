cmake_minimum_required (VERSION 3.24)

include(CheckLanguage)
include(ExternalProject)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/external)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/tests)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-omit-frame-pointer -Wno-unknown-pragmas")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)

# set the host compiler for CUDA to be the same as the C++ compiler
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-unknown-pragmas,-fno-omit-frame-pointer,-Wno-unknown-pragmas")
set(CMAKE_CUDA_STANDARD 20)

set(KNN_LIBRARIES CUDA::cudart CUDA::cublas CUDA::cuda_driver OpenMP::OpenMP_CXX)

option(CMAKE_CUDA_ARCHITECTURES "" "native")

# set the default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(knn LANGUAGES CXX CUDA)

# build the main project
set(knn_sources
    ${SRC_DIR}/knn.cpp
    ${SRC_DIR}/cuda_knn.cpp
    ${SRC_DIR}/cuda_stream.cu
    ${SRC_DIR}/verify.cpp
)

# compile only required sources from FAISS
set(faiss_sources
    ${LIBS_DIR}/faiss/faiss/gpu/utils/DeviceUtils.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/BlockSelectFloat.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloat1.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloat32.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloat64.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloat128.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloat256.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatF512.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatF1024.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatF2048.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatT512.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatT1024.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/blockselect/BlockSelectFloatT2048.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/WarpSelectFloat.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloat1.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloat32.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloat64.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloat128.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloat256.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatF512.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatF1024.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatF2048.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatT512.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatT1024.cu
    ${LIBS_DIR}/faiss/faiss/gpu/utils/warpselect/WarpSelectFloatT2048.cu
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-fdiagnostics-color=always)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-fcolor-diagnostics)
endif()

# find dependencies
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Directories
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${INCLUDE_DIR}
    ${LIBS_DIR}/cutlass/include
    ${LIBS_DIR}/cxxopts/include
    ${LIBS_DIR}/eigen
    ${LIBS_DIR}/faiss
    ${LIBS_DIR}/radik
)

# Executables
add_library(knn-static SHARED ${knn_sources} ${faiss_sources})

add_library(dist-all SHARED)
add_library(dist-minimal SHARED)

add_library(topk SHARED)
add_library(topk-multipass-all SHARED)
add_library(topk-multipass-minimal SHARED)
add_library(topk-singlepass SHARED)
add_library(topk-singlepass-detail-all SHARED)
add_library(topk-singlepass-detail-minimal SHARED)

add_executable(knn ${SRC_DIR}/main.cpp)
add_executable(knn-minimal ${SRC_DIR}/main.cpp)
add_executable(test)

# add raft (air-topk)
add_subdirectory(${LIBS_DIR}/raft/cpp EXCLUDE_FROM_ALL)

# add Catch2
add_subdirectory(${LIBS_DIR}/Catch2 EXCLUDE_FROM_ALL)

# set cuda properties
set_target_properties(knn-static PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(knn PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(knn-minimal PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(test PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(raft PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")


set_target_properties(knn-static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(knn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(knn-minimal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(knn-static PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(knn PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(knn-minimal PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(test PRIVATE "${OpenMP_CXX_FLAGS}")

# compile all parts of the project
add_subdirectory(${SRC_DIR}/distance)
add_subdirectory(${SRC_DIR}/topk)
add_subdirectory(${SRC_DIR}/topk/singlepass)
add_subdirectory(${SRC_DIR}/topk/singlepass/detail)
add_subdirectory(${SRC_DIR}/topk/multipass)
add_subdirectory(${TEST_DIR})

# link
target_link_libraries(knn PRIVATE ${KNN_LIBRARIES} knn-static dist-all topk topk-multipass-all topk-singlepass topk-singlepass-detail-all)
target_link_libraries(knn-minimal PRIVATE ${KNN_LIBRARIES} knn-static dist-minimal topk topk-multipass-minimal topk-singlepass topk-singlepass-detail-minimal)
target_link_libraries(test PRIVATE ${KNN_LIBRARIES} knn-static dist-all topk topk-singlepass Catch2::Catch2WithMain)
