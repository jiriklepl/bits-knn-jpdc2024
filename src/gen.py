import os

# create directory structure
os.makedirs("topk/singlepass/detail", exist_ok=True)

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
            # [16, 32, 64, 128, 256, 512, 1024, 2048]
            for k in [2048, 16, 1024, 32, 512, 64, 256, 128]:
                name = f"bts_{k}_{batch_size}_{block_size}.cu"
                path = f"topk/singlepass/detail/{name}"

                print(name, file=cmake)
                with open(path, "w") as f:
                    print(f"#include \"bits/topk/singlepass/detail/bits_kernel.cuh\"", file=f)
                    for prefetch in ["false", "true"]:
                            for add_norms in ["false"]:
                                bits(prefetch, add_norms, block_size, batch_size, k, f)

    for dim_reg in [1, 2, 4]:
        for dim_mult in [1, 2, 4]:
            # [2, 4, 8, 16]
            for query_reg in [16, 2, 8, 4]:
                # [4, 8, 16]
                for db_reg in [16, 4, 8]:
                    # [4, 8, 16, 32, 64, 128]
                    for k in [128, 4, 64, 8, 32, 16]:
                        name = f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu"
                        path = f"topk/singlepass/detail/{name}"

                        print(name, file=cmake)
                        with open(path, "w") as f:
                            print(f"#include \"bits/topk/singlepass/detail/fused_cache_kernel.cuh\"", file=f)
                            for block_query_dim in [1, 2, 4]:
                                block_db_dim = int(256 / block_query_dim)
                                fc(query_reg, db_reg, dim_reg, block_query_dim, block_db_dim, dim_mult, k, f)

    print(")", file=cmake)
    
    
    print("target_sources(topk-singlepass-detail-all PRIVATE ${TOPK_SINGLEPASS_ALL_SRC})", file=cmake)

    print("set(TOPK_SINGLEPASS_MINIMAL_SRC", file=cmake)
    
    for batch_size in [16]:
        for block_size in [256]:
            for k in [2048, 16, 1024, 32, 512, 64, 256, 128]:
                print(f"bts_{k}_{batch_size}_{block_size}.cu", file=cmake)

    for dim_reg in [2]:
        for dim_mult in [2]:
            for query_reg in [8]:
                for db_reg in [4]:
                    for k in [128, 4, 64, 8, 32, 16]:
                        print(f"fc_{query_reg}_{db_reg}_{dim_reg}_{dim_mult}_{k}.cu", file=cmake)

    print(")", file=cmake)

    print("target_sources(topk-singlepass-detail-minimal PRIVATE ${TOPK_SINGLEPASS_MINIMAL_SRC})", file=cmake)
