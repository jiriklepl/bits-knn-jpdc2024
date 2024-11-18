#!/usr/bin/env python3

import os

K_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
FUSED_K_VALUES = [4, 8, 16, 32, 64, 128, 256]
FUSED_CACHE_K_VALUES = [4, 8, 16, 32, 64, 128, 256]

FAISS_BLOCK_SIZES = [64, 128]
FAISS_THREAD_QUEUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

BITS_BATCH_SIZES_ALL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
BITS_BLOCK_SIZES_ALL = [128, 256, 512]

# WE ASSUME THAT THE MINIMAL CONFIGURATION IS A SUBSET OF THE ALL CONFIGURATION
BITS_BATCH_SIZES_MINIMAL = [1, 7, 16]
BITS_BLOCK_SIZES_MINIMAL = [128]

FC_DIM_REGS_ALL = [1, 2, 4]
FC_DIM_MULTS_ALL = [1, 2, 4]
FC_QUERY_REGS_ALL = [2, 4, 8, 16]
FC_DB_REGS_ALL = [4, 8, 16]
FC_BLOCK_QUERY_DIMS_ALL = [1, 2, 4]

# WE ASSUME THAT THE MINIMAL CONFIGURATION IS A SUBSET OF THE ALL CONFIGURATION
FC_DIM_REGS_MINIMAL = [2]
FC_DIM_MULTS_MINIMAL = [2]
FC_QUERY_REGS_MINIMAL = [8]
FC_DB_REGS_MINIMAL = [4]
FC_BLOCK_QUERY_DIMS_MINIMAL = [4]

FUSED_BLOCK_QUERY_DIMS = [4, 8, 16]
FUSED_QUERY_REGS = [2, 4, 8]
FUSED_POINTS_REGS = [4]

FUSED_TC_VAL_TYPES = ["half", "bfloat16", "double"] # don't change this one
FUSED_TC_BLOCK_SIZES = [128, 256, 512]


# Checks for the mentioned assumptions
assert all([x in BITS_BATCH_SIZES_ALL for x in BITS_BATCH_SIZES_MINIMAL])
assert all([x in BITS_BLOCK_SIZES_ALL for x in BITS_BLOCK_SIZES_MINIMAL])

assert all([x in FC_DIM_REGS_ALL for x in FC_DIM_REGS_MINIMAL])
assert all([x in FC_DIM_MULTS_ALL for x in FC_DIM_MULTS_MINIMAL])
assert all([x in FC_QUERY_REGS_ALL for x in FC_QUERY_REGS_MINIMAL])
assert all([x in FC_DB_REGS_ALL for x in FC_DB_REGS_MINIMAL])
assert all([x in FC_BLOCK_QUERY_DIMS_ALL for x in FC_BLOCK_QUERY_DIMS_MINIMAL])

assert all([x in ["half", "bfloat16", "double"] for x in FUSED_TC_VAL_TYPES])


# create directory structure
os.makedirs("topk/detail", exist_ok=True)
os.makedirs("topk/singlepass/detail", exist_ok=True)
os.makedirs("../include/bits/topk/singlepass/detail", exist_ok=True)


def comma_separated(values):
    return ", ".join(map(str, values))

def define(name : str, values : list, file):
    print(f"#define {name} {comma_separated(values)}", file=file)

# generate source files for the bits kernel
def bits(value_t : str, index_t : str, prefetch, block_size, batch_size, k, file):
    print(
        f"DECL_BITS_KERNEL({value_t}, {index_t}, {prefetch}, {block_size}, {batch_size}, {k});",
        file=file,
    )

# generate source files for the fused cache kernel
def fc(
    query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k, file
):
    print(
        f"DECL_FC_KERNEL({query_reg}, {db_reg}, {dim_reg}, {block_query_dim}, {block_db_dim}, {dim_mult}, {k});",
        file=file,
    )

def fused_kernel(k, reg_query, reg_points, block_query_dim, file):
    print(
        f"template void fused_kernel_runner::operator()<{k}, {reg_query}, {reg_points}, {block_query_dim}>();",
        file=file,
    )

def fused_tc_kernel(val_type, block_size, k, file):
    print(
        f"template void fused_tc_kernel_runner<fused_tc_{val_type}_policy>::operator()<{k}, {block_size}>();",
        file=file,
    )

def warp_select(block_size, thread_queue, k, file):
    print(
        f"template void warp_select_runner::operator()<{block_size}, {thread_queue}, {k}>();",
        file=file,
    )

def block_select(block_size, thread_queue, k, file):
    print(
        f"template void block_select_runner::operator()<{block_size}, {thread_queue}, {k}>();",
        file=file,
    )

with open(
    "../include/bits/topk/singlepass/detail/definitions_common.hpp", "w"
) as definitions:
    print("#ifndef TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_", file=definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_K_VALUES", K_VALUES, definitions)
    define("TOPK_SINGLEPASS_FUSED_K_VALUES", FUSED_K_VALUES, definitions)
    define("TOPK_SINGLEPASS_FUSED_CACHE_K_VALUES", FUSED_CACHE_K_VALUES, definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_FAISS_BLOCK_SIZES", FAISS_BLOCK_SIZES, definitions)
    define("TOPK_SINGLEPASS_FAISS_THREAD_QUEUES", FAISS_THREAD_QUEUES, definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_FUSED_BLOCK_QUERY_DIMS", FUSED_BLOCK_QUERY_DIMS, definitions)
    define("TOPK_SINGLEPASS_FUSED_QUERY_REGS", FUSED_QUERY_REGS, definitions)
    define("TOPK_SINGLEPASS_FUSED_POINTS_REGS", FUSED_POINTS_REGS, definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_FUSED_TC_VAL_TYPES", FUSED_TC_VAL_TYPES, definitions)
    define("TOPK_SINGLEPASS_FUSED_TC_BLOCK_SIZES", FUSED_TC_BLOCK_SIZES, definitions)
    print("", file=definitions)

    print("#endif // TOPK_SINGLEPASS_DETAIL_DEFINITIONS_COMMON_HPP_", file=definitions)

with open(
    "../include/bits/topk/singlepass/detail/definitions_all.hpp", "w"
) as definitions:
    print("#ifndef TOPK_SINGLEPASS_DETAIL_DEFINITIONS_ALL_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DETAIL_DEFINITIONS_ALL_HPP_", file=definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_BITS_BATCH_SIZES", BITS_BATCH_SIZES_ALL, definitions)
    define("TOPK_SINGLEPASS_BITS_BLOCK_SIZES", BITS_BLOCK_SIZES_ALL, definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_FC_DIM_REGS", FC_DIM_REGS_ALL, definitions)
    define("TOPK_SINGLEPASS_FC_DIM_MULTS", FC_DIM_MULTS_ALL, definitions)
    define("TOPK_SINGLEPASS_FC_QUERY_REGS", FC_QUERY_REGS_ALL, definitions)
    define("TOPK_SINGLEPASS_FC_DB_REGS", FC_DB_REGS_ALL, definitions)
    define("TOPK_SINGLEPASS_FC_BLOCK_QUERY_DIMS", FC_BLOCK_QUERY_DIMS_ALL, definitions)
    print("", file=definitions)

    print(f"#endif // TOPK_SINGLEPASS_DETAIL_DEFINITIONS_ALL_HPP_", file=definitions)

with open(
    "../include/bits/topk/singlepass/detail/definitions_minimal.hpp", "w"
) as definitions:
    print("#ifndef TOPK_SINGLEPASS_DETAIL_DEFINITIONS_MINIMAL_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DETAIL_DEFINITIONS_MINIMAL_HPP_", file=definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_BITS_BATCH_SIZES", BITS_BATCH_SIZES_MINIMAL, definitions)
    define("TOPK_SINGLEPASS_BITS_BLOCK_SIZES", BITS_BLOCK_SIZES_MINIMAL, definitions)
    print("", file=definitions)

    define("TOPK_SINGLEPASS_FC_DIM_REGS", FC_DIM_REGS_MINIMAL, definitions)
    define("TOPK_SINGLEPASS_FC_DIM_MULTS", FC_DIM_MULTS_MINIMAL, definitions)
    define("TOPK_SINGLEPASS_FC_QUERY_REGS", FC_QUERY_REGS_MINIMAL, definitions)
    define("TOPK_SINGLEPASS_FC_DB_REGS", FC_DB_REGS_MINIMAL, definitions)
    define("TOPK_SINGLEPASS_FC_BLOCK_QUERY_DIMS", FC_BLOCK_QUERY_DIMS_MINIMAL, definitions)
    print("", file=definitions)

    print(f"#endif // TOPK_SINGLEPASS_DETAIL_DEFINITIONS_MINIMAL_HPP_", file=definitions)

with open("topk/singlepass/detail/CMakeLists.txt", "w") as cmake:
    print("set(TOPK_SINGLEPASS_ALL_SRC", file=cmake)

    for batch_size in BITS_BATCH_SIZES_ALL:
        for block_size in BITS_BLOCK_SIZES_ALL:
            for k in K_VALUES:
                name = f"bts_{k}_{batch_size}_{block_size}.cu"
                path = f"topk/singlepass/detail/{name}"

                print(f"    {name}", file=cmake)
                with open(path, "w") as f:
                    print(
                        f'#include "bits/topk/singlepass/detail/bits_kernel.cuh"',
                        file=f,
                    )
                    print("", file=f)

                    for prefetch in ["false", "true"]:
                        bits("float", "std::int32_t", prefetch, block_size, batch_size, k, f)

    for dim_reg in FC_DIM_REGS_ALL:
        for dim_mult in FC_DIM_MULTS_ALL:
            for query_reg in FC_QUERY_REGS_ALL:
                for db_reg in FC_DB_REGS_ALL:
                    for k in FUSED_CACHE_K_VALUES:
                        name = f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu"
                        path = f"topk/singlepass/detail/{name}"

                        print(f"    {name}", file=cmake)
                        with open(path, "w") as f:
                            print(
                                f'#include "bits/topk/singlepass/detail/fused_cache_kernel.cuh"',
                                file=f,
                            )
                            print("", file=f)

                            for block_query_dim in FC_BLOCK_QUERY_DIMS_ALL:
                                block_db_dim = int(256 / block_query_dim)
                                fc(
                                    query_reg,
                                    db_reg,
                                    dim_reg,
                                    block_query_dim,
                                    block_db_dim,
                                    dim_mult,
                                    k,
                                    f,
                                )

    print(")", file=cmake)
    print("", file=cmake)

    print(
        "target_sources(topk-singlepass-all PRIVATE ${TOPK_SINGLEPASS_ALL_SRC})",
        file=cmake,
    )
    print(
        "target_compile_definitions(topk-singlepass-all PRIVATE TOPK_SINGLEPASS_USE_ALL)",
        file=cmake,
    )
    print("", file=cmake)

    print("set(TOPK_SINGLEPASS_MINIMAL_SRC", file=cmake)

    for batch_size in BITS_BATCH_SIZES_MINIMAL:
        for block_size in BITS_BLOCK_SIZES_MINIMAL:
            for k in K_VALUES:
                name = f"bts_{k}_{batch_size}_{block_size}.cu"
                path = f"topk/singlepass/detail/{name}"

                print(f"    {name}", file=cmake)

    for dim_reg in FC_DIM_REGS_MINIMAL:
        for dim_mult in FC_DIM_MULTS_MINIMAL:
            for query_reg in FC_QUERY_REGS_MINIMAL:
                for db_reg in FC_DB_REGS_MINIMAL:
                    for k in FUSED_CACHE_K_VALUES:
                        name = f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu"
                        path = f"topk/singlepass/detail/{name}"

                        print(f"    {name}", file=cmake)

    print(")", file=cmake)
    print("", file=cmake)

    print(
        "target_sources(topk-singlepass-minimal PRIVATE ${TOPK_SINGLEPASS_MINIMAL_SRC})",
        file=cmake,
    )
    print(
        "target_compile_definitions(topk-singlepass-minimal PRIVATE TOPK_SINGLEPASS_USE_MINIMAL)",
        file=cmake,
    )

with open("topk/CMakeLists.txt", "w") as cmake:
    print("set(TOPK_SRC", file=cmake)
    print("    serial_knn.cpp", file=cmake)
    print("    parallel_knn.cpp", file=cmake)

    for k in FUSED_K_VALUES:
        name = f"detail/fused_kernel{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f'#include "bits/topk/singlepass/detail/fused_kernel.cuh"', file=f)

            for block_query_dim in FUSED_BLOCK_QUERY_DIMS:
                print("", file=f)
                for reg_query in FUSED_QUERY_REGS:
                    for reg_points in FUSED_POINTS_REGS:
                        fused_kernel(k, reg_query, reg_points, block_query_dim, f)

    # fused-tc-*
    for val_type in FUSED_TC_VAL_TYPES:
        for block_size in FUSED_TC_BLOCK_SIZES:
            name = f"detail/fused_tc_kernel_{val_type}_{block_size}.cu"
            path = f"topk/{name}"

            print(f"    {name}", file=cmake)
            with open(path, "w") as f:
                print(
                    f'#include "bits/topk/singlepass/detail/fused_tc_kernel.cuh"',
                    file=f,
                )
                print("", file=f)

                for k in FUSED_K_VALUES:
                    fused_tc_kernel(val_type, block_size, k, f)

    # warp-select
    for k in K_VALUES:
        name = f"detail/warp_select{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f'#include "bits/topk/singlepass/detail/warp_select.cuh"', file=f)
            print("", file=f)

            for block_size in FAISS_BLOCK_SIZES:
                for thread_queue in FAISS_THREAD_QUEUES:
                    warp_select(block_size, thread_queue, k, f)

    # block-select
    for k in K_VALUES:
        name = f"detail/block_select{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f'#include "bits/topk/singlepass/detail/block_select.cuh"', file=f)
            print("", file=f)

            for block_size in FAISS_BLOCK_SIZES:
                for thread_queue in FAISS_THREAD_QUEUES:
                    block_select(block_size, thread_queue, k, f)

    print(")", file=cmake)
    print("", file=cmake)

    print("target_sources(topk PRIVATE ${TOPK_SRC})", file=cmake)
