#!/usr/bin/env python3

import os

K_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048]
FUSED_K_VALUES = [2, 4, 8, 16, 32, 64, 128]
FUSED_CACHE_K_VALUES = [4, 8, 16, 32, 64, 128]

# create directory structure
os.makedirs("topk/detail", exist_ok=True)
os.makedirs("topk/singlepass/detail", exist_ok=True)
os.makedirs("../include/bits/topk/singlepass/detail", exist_ok=True)

def comma_separated(values):
    return ",".join(map(str, values))

with open("../include/bits/topk/singlepass/detail/definitions_common.hpp", "w") as definitions:
    print("#ifndef TOPK_SINGLEPASS_DEFINITIONS_COMMON_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DEFINITIONS_COMMON_HPP_", file=definitions)

    print(f"#define TOPK_SINGLEPASS_K_VALUES {comma_separated(K_VALUES)}", file=definitions)
    print(f"#define TOPK_SINGLEPASS_FUSED_K_VALUES {comma_separated(FUSED_K_VALUES)}", file=definitions)
    print(f"#define TOPK_SINGLEPASS_FUSED_CACHE_K_VALUES {comma_separated(FUSED_CACHE_K_VALUES)}", file=definitions)

    print("#endif // TOPK_SINGLEPASS_DEFINITIONS_COMMON_HPP_", file=definitions)

with open("../include/bits/topk/singlepass/detail/definitions_all.hpp", "w") as definitions:
    print("#ifndef TOPK_SINGLEPASS_DEFINITIONS_ALL_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DEFINITIONS_ALL_HPP_", file=definitions)

    print(f"#endif // TOPK_SINGLEPASS_DEFINITIONS_ALL_HPP_", file=definitions)

with open("../include/bits/topk/singlepass/detail/definitions_minimal.hpp", "w") as definitions:
    print("#ifndef TOPK_SINGLEPASS_DEFINITIONS_MINIMAL_HPP_", file=definitions)
    print("#define TOPK_SINGLEPASS_DEFINITIONS_MINIMAL_HPP_", file=definitions)

    print(f"#endif // TOPK_SINGLEPASS_DEFINITIONS_MINIMAL_HPP_", file=definitions)

with open("topk/singlepass/detail/CMakeLists.txt", "w") as cmake:
    print("set(TOPK_SINGLEPASS_ALL_SRC", file=cmake)

    # generate source files for the bits kernel
    def bits(prefetch, add_norms, block_size, batch_size, k, file):
        print(f"DECL_BITS_KERNEL({prefetch}, {add_norms}, {block_size}, {batch_size}, {k});", file=file)

    # generate source files for the fused cache kernel
    def fc(query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k, file):
        print(f"DECL_FC_KERNEL({query_reg}, {db_reg}, {dim_reg}, {block_query_dim}, {block_db_dim}, {dim_mult}, {k});", file=file)

    for batch_size in range(1, 17):
        # [128, 256, 512]
        for block_size in [512, 128, 256]:
            for k in K_VALUES:
                name = f"bts_{k}_{batch_size}_{block_size}.cu"
                path = f"topk/singlepass/detail/{name}"

                print(f"    {name}", file=cmake)
                with open(path, "w") as f:
                    print(f"#include \"bits/topk/singlepass/detail/bits_kernel.cuh\"", file=f)
                    print("", file=f)

                    for prefetch in ["false", "true"]:
                            for add_norms in ["false"]:
                                bits(prefetch, add_norms, block_size, batch_size, k, f)

    for dim_reg in [1, 2, 4]:
        for dim_mult in [1, 2, 4]:
            # [2, 4, 8, 16]
            for query_reg in [16, 2, 8, 4]:
                # [4, 8, 16]
                for db_reg in [16, 4, 8]:
                    for k in FUSED_CACHE_K_VALUES:
                        name = f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu"
                        path = f"topk/singlepass/detail/{name}"

                        print(f"    {name}", file=cmake)
                        with open(path, "w") as f:
                            print(f"#include \"bits/topk/singlepass/detail/fused_cache_kernel.cuh\"", file=f)
                            print("", file=f)

                            for block_query_dim in [1, 2, 4]:
                                block_db_dim = int(256 / block_query_dim)
                                fc(query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k, f)

    print(")", file=cmake)
    print("", file=cmake)

    print("target_sources(topk-singlepass-all PRIVATE ${TOPK_SINGLEPASS_ALL_SRC})", file=cmake)
    print(f"target_compile_definitions(topk-singlepass-all PRIVATE TOPK_SINGLEPASS_USE_ALL)", file=cmake)
    print("", file=cmake)

    # TODO: add source files for the tests
    # print("target_sources(topk-singlepass-test PRIVATE ${TOPK_SINGLEPASS_ALL_SRC})", file=cmake)
    # print(f"target_compile_definitions(topk-singlepass-test PRIVATE \"TOPK_SINGLEPASS_K_VALUES={comma_separated(K_VALUES)}\")", file=cmake)
    # print(f"target_compile_definitions(topk-singlepass-test PRIVATE \"TOPK_SINGLEPASS_FUSED_K_VALUES={comma_separated(FUSED_K_VALUES)}\")", file=cmake)
    # print("", file=cmake)

    print("set(TOPK_SINGLEPASS_MINIMAL_SRC", file=cmake)

    for batch_size in [16]:
        for block_size in [256]:
            for k in K_VALUES:
                name = f"bts_{k}_{batch_size}_{block_size}.cu"
                path = f"topk/singlepass/detail/{name}"

                print(f"    {name}", file=cmake)

    for dim_reg in [2]:
        for dim_mult in [2]:
            for query_reg in [8]:
                for db_reg in [4]:
                    for k in FUSED_CACHE_K_VALUES:
                        name = f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu"
                        path = f"topk/singlepass/detail/{name}"

                        print(f"    {name}", file=cmake)

    print(")", file=cmake)
    print("", file=cmake)

    print("target_sources(topk-singlepass-minimal PRIVATE ${TOPK_SINGLEPASS_MINIMAL_SRC})", file=cmake)
    print(f"target_compile_definitions(topk-singlepass-minimal PRIVATE TOPK_SINGLEPASS_USE_MINIMAL)", file=cmake)

with open("topk/CMakeLists.txt", "w") as cmake:
    print("set(TOPK_SRC", file=cmake)
    print("    serial_knn.cpp", file=cmake)
    print("    parallel_knn.cpp", file=cmake)

    def fused_kernel(k, reg_query, reg_points, block_query_dim, file):
        print(f"template void fused_kernel_runner::operator()<{k}, {reg_query}, {reg_points}, {block_query_dim}>();", file=file)

    def fused_tc_kernel(val_type, block_size, k, file):
        print(f"template void fused_tc_kernel_runner<fused_tc_{val_type}_policy>::operator()<{k}, {block_size}>();", file=file)

    def warp_select(block_size, thread_queue, k, file):
        print(f"template void warp_select_runner::operator()<{block_size}, {thread_queue}, {k}>();", file=file)

    def block_select(block_size, thread_queue, k, file):
        print(f"template void block_select_runner::operator()<{block_size}, {thread_queue}, {k}>();", file=file)

    for k in FUSED_K_VALUES:
        name = f"detail/fused_kernel{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f"#include \"bits/topk/singlepass/detail/fused_kernel.cuh\"", file=f)

            for block_query_dim in [4, 8, 16]:
                print("", file=f)
                for reg_query in [2, 4, 8]:
                    for reg_points in [4]:
                        fused_kernel(k, reg_query, reg_points, block_query_dim, f)

    for val_type in ["half", "bfloat16", "double"]:
        for block_size in [128, 256, 512]:
            name = f"detail/fused_tc_kernel_{val_type}_{block_size}.cu"
            path = f"topk/{name}"

            print(f"    {name}", file=cmake)
            with open(path, "w") as f:
                print(f"#include \"bits/topk/singlepass/detail/fused_tc_kernel.cuh\"", file=f)
                print("", file=f)

                for k in FUSED_K_VALUES:
                    fused_tc_kernel(val_type, block_size, k, f)

    for k in K_VALUES:
        name = f"detail/warp_select{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f"#include \"bits/topk/singlepass/detail/warp_select.cuh\"", file=f)
            print("", file=f)

            for block_size in [64, 128]:
                for thread_queue in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    warp_select(block_size, thread_queue, k, f)

    for k in K_VALUES:
        name = f"detail/block_select{k}.cu"
        path = f"topk/{name}"

        print(f"    {name}", file=cmake)
        with open(path, "w") as f:
            print(f"#include \"bits/topk/singlepass/detail/block_select.cuh\"", file=f)
            print("", file=f)

            for block_size in [64, 128]:
                for thread_queue in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    block_select(block_size, thread_queue, k, f)

    print(")", file=cmake)
    print("", file=cmake)

    print("target_sources(topk PRIVATE ${TOPK_SRC})", file=cmake)
