#!/bin/bash

# Setting SLURM_JOB_ID outside of SLURM is not recommended outside of testing

command_name="$0"

print_warning() {
    printf -- "%s: Warning: %s\n" "$command_name" "$1" >&2
}

print_error() {
    printf -- "%s: Error: %s\n" "$command_name" "$1" >&2
    exit 1
}

gpu_deduce() {
    if [ -z "$worker" ] || [ -z "$GPU" ]; then
        # try to determine the GPU from the CUDA_ARCHITECTURES variable using nvidia-smi
        if [ -n "$worker" ]; then
            if [ "$worker" == "ampere01" ]; then
                GPU="L40"
            elif [ "$worker" == "ampere02" ]; then
                GPU="A100"
            elif [ "$worker" == "hopper01" ]; then
                GPU="H100"
            elif [ "$worker" == "volta05" ]; then
                GPU="V100"
            else                                                     # default values
                print_warning "unknown worker, using default values" # see below
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="80"
                fi
            fi
        elif [ -z "$GPU" ]; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
            case "$GPU_NAME" in
            *"NVIDIA V100"* | *"Tesla V100"*)
                GPU="V100"
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="70"
                fi
                ;;
            *"NVIDIA A100"* | *"Tesla A100"*)
                GPU="A100"
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="80"
                fi
                ;;
            *"NVIDIA H100"* | *"Tesla H100"*)
                GPU="H100"
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="90"
                fi
                ;;
            *"NVIDIA L40"* | *"Tesla L40"*)
                GPU="L40"
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="89"
                fi
                ;;
            *)
                print_warning "unknown GPU architecture, using default values" # see below
                if [ -z "$CUDA_ARCHITECTURES" ]; then
                    CUDA_ARCHITECTURES="80"
                fi
                ;;
            esac
        fi
    fi
}

if [ -z "$worker" ] && [ -n "$SLURM_JOB_ID" ]; then
    print_error "worker is not set; for gpulab, use one of ampere01, ... or volta01, ..."
fi

if [ -z "$root_dir" ]; then
    root_dir=$(realpath "$(dirname "$0")")
    print_warning "root_dir is not set, using $root_dir"
fi

if ! [ -d "$root_dir" ]; then
    print_error "root_dir does not exist: $root_dir"
fi

if [ -z "$build_dir" ]; then
    build_dir=build-release
    print_warning "build_dir is not set, using $build_dir"
fi

# if build_dir is not an absolute path, make it absolute
if [ "${build_dir:0:1}" != "/" ]; then
    build_dir="$root_dir/$build_dir"
fi

export knn="$build_dir/knn"

# parameters: optima_file algorithm query_count k dim
config_generic() {
    # find the best configuration for the algorithm in optima_file
    awk -F, -v algorithm="$2" -v GPU="$GPU" -v query_count="$3" -v k="$4" -v dim="$5" '
        BEGIN {
            slack = 0
            score = 0

            block_size = 256
            deg = 1
            items_per_thread = 1
            items_per_thread2 = 1
            items_per_thread3 = 1
        }
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                col[$i] = i
            }

            if (!col["algorithm"] || !col["GPU"]) {
                exit 1
            }

            next
        }
        $col["algorithm"] == algorithm && (GPU == "None" || $col["GPU"] == GPU) {
            new_score = 0
            new_slack = 1

            if (col["block_size"] && $col["block_size"] != "") {
                new_block_size = $col["block_size"]
            }

            if (col["deg"] && $col["deg"] != "") {
                new_deg = $col["deg"]
            }

            if (col["items_per_thread"] && $col["items_per_thread"] != "") {
                new_items_per_thread = $col["items_per_thread"]
            }

            if (col["items_per_thread2"] && $col["items_per_thread2"] != "") {
                new_items_per_thread2 = $col["items_per_thread2"]
            }

            if (col["items_per_thread3"] && $col["items_per_thread3"] != "") {
                new_items_per_thread3 = $col["items_per_thread3"]
            }

            if (query_count != "None") {
                if (col["query_count"] && $col["query_count"] != "") {
                    if ($col["query_count"] >= query_count) {
                        new_score += 1
                        new_slack *= (1.0 * $col["query_count"] / query_count)
                    } else { # $col["query_count"] < query_count
                        new_slack *= (1.0 * query_count / $col["query_count"])
                    }
                }
            }

            if (k != "None") {
                if (col["k"] && $col["k"] != "") {
                    if ($col["k"] >= k) {
                        new_score += 1
                        new_slack *= (1.0 * $col["k"] / k)
                    } else { # $col["k"] < k
                        new_slack *= (1.0 * k / $col["k"])
                    }
                }
            }

            if (dim != "None") {
                if (col["dim"] && $col["dim"] != "") {
                    if ($col["dim"] >= dim) {
                        new_score += 1
                        new_slack *= (1.0 * $col["dim"] / dim)
                    } else { # $col["dim"] < dim
                        new_slack *= (1.0 * dim / $col["dim"])
                    }
                }
            }

            if (new_score >= score && (slack == 0 || new_slack < slack)) {
                score = new_score
                slack = new_slack

                block_size = new_block_size
                deg = new_deg
                items_per_thread = new_items_per_thread
                items_per_thread2 = new_items_per_thread2
                items_per_thread3 = new_items_per_thread3
            }
        }
        END {
            print block_size, deg, items_per_thread, items_per_thread2, items_per_thread3
        }
    ' "$1"
}

# parameters: algorithm query_count k [dim]
# output: block_size deg items_per_thread items_per_thread2 items_per_thread3
#
# if dim is not provided, it is set to "None"
config_algorithm() {
    if [ $# -eq 3 ]; then
        dim="None"
    else
        dim=$4
    fi

    optima_file="$root_dir/scripts/optima.csv"

    config_generic "$optima_file" "$1" "$2" "$3" "$dim"
}

# parameters: algorithm query_count dim [k]
# output: block_size deg items_per_thread items_per_thread2 items_per_thread3
config_distance() {
    if [ $# -eq 3 ]; then
        k="None"
    else
        k=$4
    fi

    optima_file="$root_dir/scripts/optima-dist.csv"

    config_generic "$optima_file" "$1" "$2" "$k" "$3"
}
