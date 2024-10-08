#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "bits/knn.hpp"
#include "bits/knn_args.hpp"
#include "bits/layout.hpp"
#include "bits/utils.hpp"
#include "bits/verify.hpp"

#include "bits/data_generator.hpp"
#include "bits/normal_generator.hpp"
#include "bits/radix_adversarial_generator.hpp"
#include "bits/uniform_generator.hpp"

#include "bits/data_preprocessor.hpp"
#include "bits/identity_preprocessor.hpp"
#include "bits/ordering_preprocessor.hpp"

#include "bits/distance/baseline_distance.hpp"
#include "bits/distance/cublas_distance.hpp"
#include "bits/distance/cutlass_distance.hpp"
#include "bits/distance/dist_runner.hpp"
#include "bits/distance/magma_distance.hpp"
#include "bits/distance/tiled_distance.hpp"

#include "bits/topk/parallel_knn.hpp"
#include "bits/topk/serial_knn.hpp"

#include "bits/topk/multipass/air_topk.hpp"
#include "bits/topk/multipass/radik_knn.hpp"

#include "bits/topk/singlepass/bits_knn.hpp"
#include "bits/topk/singlepass/cub_knn.hpp"
#include "bits/topk/singlepass/fused_cache_knn.hpp"
#include "bits/topk/singlepass/fused_knn.hpp"
#include "bits/topk/singlepass/fused_tc_knn.hpp"
#include "bits/topk/singlepass/grid_select.hpp"
#include "bits/topk/singlepass/partial_bitonic.hpp"
#include "bits/topk/singlepass/partial_bitonic_buffered.hpp"
#include "bits/topk/singlepass/rapidsai_fused.hpp"
#include "bits/topk/singlepass/warp_select.hpp"

namespace
{

template <typename AlgorithmsRange>
void print_help(cxxopts::Options& options, const AlgorithmsRange& algorithms)
{
    std::cerr << options.help() << '\n';
    std::cerr << "Available algorithms:" << '\n';
    for (const auto& alg : algorithms)
    {
        std::cerr << "- " << alg->id() << "\n";
    }
}

std::vector<float> transpose(const float* matrix, std::size_t row_count, std::size_t col_count)
{
    std::vector<float> result(row_count * col_count);
    for (std::size_t i = 0; i < row_count; ++i)
    {
        for (std::size_t j = 0; j < col_count; ++j)
        {
            result[j * row_count + i] = matrix[i * col_count + j];
        }
    }
    return result;
}

void print_log_header()
{
    std::cout << "algorithm,generator,preprocessor,iteration,point_count,query_count,dim,"
                 "block_size,k,items_per_thread,deg,phase,time"
              << '\n';
}

void log(const knn_args& args, const std::string& algorithm, const std::string& generator,
         const std::string& preprocessor, const std::string& phase, std::size_t iteration,
         double time)
{
    const auto default_precision = std::cout.precision();
    std::cout << algorithm << "," << generator << "," << preprocessor << "," << iteration << ","
              << args.point_count << "," << args.query_count << "," << args.dim << ","
              << args.selection_block_size << "," << args.k << ",\"" << args.items_per_thread[0]
              << "," << args.items_per_thread[1] << "," << args.items_per_thread[2] << "\","
              << args.deg << "," << phase << "," << std::scientific
              << std::setprecision(std::numeric_limits<double>::digits10 + 1) << time
              << std::defaultfloat << std::setprecision(default_precision) << '\n';
}

} // namespace

int main(int argc, char** argv)
try
{
    std::vector<std::unique_ptr<knn>> algorithms;

    algorithms.push_back(std::make_unique<serial_knn>());
    algorithms.push_back(std::make_unique<parallel_knn>());

    algorithms.push_back(std::make_unique<partial_bitonic>());
    algorithms.push_back(std::make_unique<partial_bitonic_warp>());
    algorithms.push_back(std::make_unique<partial_bitonic_warp_static>());
    algorithms.push_back(std::make_unique<buffered_partial_bitonic>());
    algorithms.push_back(std::make_unique<partial_bitonic_arrays>());
    algorithms.push_back(std::make_unique<partial_bitonic_regs>());

    algorithms.push_back(std::make_unique<static_buffered_partial_bitonic>());

    algorithms.push_back(std::make_unique<bits_knn>());
    algorithms.push_back(std::make_unique<bits_prefetch_knn>());
    algorithms.push_back(std::make_unique<single_query_bits>());

    algorithms.push_back(std::make_unique<warp_select>());
    algorithms.push_back(std::make_unique<block_select>());
    algorithms.push_back(std::make_unique<warp_select_tunable>());
    algorithms.push_back(std::make_unique<block_select_tunable>());
    algorithms.push_back(std::make_unique<warp_select_tuned>());
    algorithms.push_back(std::make_unique<block_select_tuned>());

    algorithms.push_back(std::make_unique<air_topk>());
    algorithms.push_back(std::make_unique<grid_select>());

    algorithms.push_back(std::make_unique<fused_regs_knn>());
    algorithms.push_back(std::make_unique<fused_regs_knn_tunable>());

    algorithms.push_back(std::make_unique<fused_cache_knn>());

    algorithms.push_back(std::make_unique<fused_tc_half_knn>());
    algorithms.push_back(std::make_unique<fused_tc_bfloat16_knn>());
    algorithms.push_back(std::make_unique<fused_tc_double_knn>());

    algorithms.push_back(std::make_unique<rapidsai_fused>());

    algorithms.push_back(std::make_unique<cub_knn>());
    algorithms.push_back(std::make_unique<cub_direct>());

    algorithms.push_back(std::make_unique<radik_knn>());

    algorithms.push_back(std::make_unique<dist_runner<baseline_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<tiled_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<cublas_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<magma_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<magma_partial_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<magma_kl_distance>>());
    algorithms.push_back(std::make_unique<dist_runner<cutlass_distance>>());

    std::vector<std::unique_ptr<data_generator>> generators;

    generators.push_back(std::make_unique<uniform_generator>());
    generators.push_back(std::make_unique<normal_generator>());
    generators.push_back(std::make_unique<radix_adversarial_generator>());

    std::vector<std::unique_ptr<data_preprocessor>> preprocessors;

    preprocessors.push_back(std::make_unique<identity_preprocessor>());
    preprocessors.push_back(std::make_unique<ordering_preprocessor<std::less<>>>());
    preprocessors.push_back(std::make_unique<ordering_preprocessor<std::greater<>>>());

    cxxopts::Options options{"knn", "kNN"};

    // clang-format off
    options.add_options()
        ("a,algorithm", "Used algorithm",
            cxxopts::value<std::string>()->default_value("serial"))
        ("r,repeat", "Number of executions",
            cxxopts::value<std::string>()->default_value("1"))
        ("k,knn", "Number of nearest neighbors",
            cxxopts::value<std::string>()->default_value("1"))
        ("n,number", "Number of objects in the database",
            cxxopts::value<std::string>()->default_value("1024"))
        ("q,query", "Number of query objects",
            cxxopts::value<std::string>()->default_value("1024"))
        ("d,dimension", "Dimension of objects",
            cxxopts::value<std::string>()->default_value("10"))
        ("g,generator", "Data generator",
            cxxopts::value<std::string>()->default_value("uniform"))
        ("p,preprocessor", "Data preprocessor",
            cxxopts::value<std::string>()->default_value("identity"))
        ("seed", "Seed for the data generator",
            cxxopts::value<std::string>()->default_value("0"))
        ("block-size", "Block size for CUDA kernels",
            cxxopts::value<std::string>()->default_value("256"))
        ("items-per-thread", "Number of items per thread in the fused-regs kernel (two comma separated values)",
            cxxopts::value<std::string>()->default_value("1"))
        ("v,verify", "Verify the results against selected implementation",
            cxxopts::value<std::string>()->default_value("off"))
        ("point-layout", R"(Layout of the point matrix - "column" for column major or "row" for row major)",
            cxxopts::value<std::string>()->default_value("row"))
        ("query-layout", R"(Layout of the query matrix - "column" for column major or "row" for row major)",
            cxxopts::value<std::string>()->default_value("row"))
        ("deg", "degree of parallelism used for single-query problems",
            cxxopts::value<std::string>()->default_value("1"))
        ("no-output", "Do not copy the result back to CPU",
            cxxopts::value<bool>()->default_value("false"))
        ("random-distances", "Use random unique distances instead of actual distances.",
            cxxopts::value<bool>()->default_value("false"))
        ("header", "Print header with column names and exit",
            cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Show help message and exit");
    // clang-format on

    try
    {
        auto params = options.parse(argc, argv);

        if (params.count("help") != 0)
        {
            print_help(options, algorithms);
            return 0;
        }

        if (params.count("header") != 0)
        {
            print_log_header();
            return 0;
        }

        // parse command line arguments
        const std::size_t dim = parse_number(params["dimension"].as<std::string>());
        const std::size_t k = parse_number(params["knn"].as<std::string>());
        const std::size_t input_size = parse_number(params["number"].as<std::string>());
        const std::size_t query_size = parse_number(params["query"].as<std::string>());
        const std::size_t repeat_count = parse_number(params["repeat"].as<std::string>());
        const std::size_t seed = parse_number(params["seed"].as<std::string>());
        std::string algorithm_id = params["algorithm"].as<std::string>();
        std::string generator_id = params["generator"].as<std::string>();
        std::string preprocessor_id = params["preprocessor"].as<std::string>();
        const std::size_t block_size = parse_number(params["block-size"].as<std::string>());
        const std::string verify_alg = params["verify"].as<std::string>();
        const std::string layout_points = params["point-layout"].as<std::string>();
        const std::string layout_queries = params["query-layout"].as<std::string>();
        const std::array<std::size_t, 3> items_per_thread =
            parse_dim3(params["items-per-thread"].as<std::string>());

        // do basic validation
        if (dim <= 0)
        {
            std::cerr << "Dimension must be greater than 0" << '\n';
            return 1;
        }

        if (input_size <= 0)
        {
            std::cerr << "Number of objects in the database must be greater than 0" << '\n';
            return 1;
        }

        if (query_size <= 0)
        {
            std::cerr << "Number of query objects must be greater than 0" << '\n';
            return 1;
        }

        if (k <= 0)
        {
            std::cerr << "Number of nearest neighbors must be greater than 0" << '\n';
            return 1;
        }

        if (block_size <= 0)
        {
            std::cerr << "Block size must be greater than 0" << '\n';
            return 1;
        }

        const auto generator_sep = generator_id.find(':');
        std::string generator_params{};

        if (generator_sep != std::string::npos)
        {
            generator_params = generator_id.substr(generator_sep + 1);
            generator_id = generator_id.substr(0, generator_sep);
        }

        const auto gen_it =
            std::find_if(generators.begin(), generators.end(),
                         [&generator_id](auto& gen) { return gen->id() == generator_id; });
        if (gen_it == generators.end())
        {
            std::cerr << "Unknown generator: '" << generator_id << "'" << '\n';
            // TODO(jirka): print available generators
            return 1;
        }

        const auto generator = gen_it->get();

        generator->set_seed(seed);
        if (generator_sep != std::string::npos)
        {
            generator->set_params(generator_params);
        }

        auto data = generator->generate(input_size, dim);
        auto query = generator->generate(query_size, dim);

        // preprocess data
        const auto pre_it =
            std::find_if(preprocessors.begin(), preprocessors.end(),
                         [&preprocessor_id](auto& pre) { return pre->id() == preprocessor_id; });

        if (pre_it == preprocessors.end())
        {
            std::cerr << "Unknown preprocessor: '" << preprocessor_id << "'" << '\n';
            // TODO(jirka): print available preprocessors
            return 1;
        }

        const auto preprocessor = pre_it->get();

        preprocessor->preprocess(data, query, dim);

        // create kNN instance
        // TODO(jirka): add generator and preprocessor arguments
        knn_args args{.points = data.data(),
                      .queries = query.data(),
                      .point_count = input_size,
                      .query_count = query_size,
                      .dim = dim,
                      .points_layout = layout_points == "row" ? matrix_layout::row_major
                                                              : matrix_layout::column_major,
                      .queries_layout = layout_queries == "row" ? matrix_layout::row_major
                                                                : matrix_layout::column_major,
                      .dist_layout = matrix_layout::row_major,
                      .dist_block_size = block_size,
                      .selection_block_size = block_size,
                      .k = k,
                      .items_per_thread = items_per_thread,
                      .deg = parse_number(params["deg"].as<std::string>())};

        // transpose matrices to requested layout
        if (args.points_layout == matrix_layout::column_major)
        {
            data = transpose(data.data(), args.point_count, args.dim);
            args.points = data.data();
        }

        if (args.queries_layout == matrix_layout::column_major)
        {
            query = transpose(query.data(), args.query_count, args.dim);
            args.queries = query.data();
        }

        // run knn
        const bool no_output = params["no-output"].as<bool>();

        const auto alg_it =
            std::find_if(algorithms.begin(), algorithms.end(),
                         [&algorithm_id](auto& alg) { return alg->id() == algorithm_id; });
        if (alg_it == algorithms.end())
        {
            std::cerr << "Unknown algorithm: '" << algorithm_id << "'" << '\n';
            print_help(options, algorithms);
            return 1;
        }

        const auto alg = alg_it->get();
        if (no_output)
        {
            alg->no_output();
        }
        alg->initialize(args);

        // repeat the computation multiple times
        for (std::size_t i = 0; i < repeat_count; ++i)
        {
            std::atomic<std::uint32_t> sink{0};

            // prepare
            auto start = std::chrono::steady_clock::now();
            alg->prepare();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> duration = end - start;
            log(args, algorithm_id, generator_id, preprocessor_id, "prepare", i, duration.count());
            log(args, algorithm_id, generator_id, preprocessor_id, "transfer-in", i,
                alg->transfer_in_seconds());

            // compute distances
            start = std::chrono::steady_clock::now();
            alg->distances();
            end = std::chrono::steady_clock::now();
            duration = end - start;
            log(args, algorithm_id, generator_id, preprocessor_id, "distances", i,
                duration.count());

            if (params["random-distances"].as<bool>())
            {
                alg->set_random_distances();
            }

            // execute
            start = std::chrono::steady_clock::now();
            alg->selection();
            end = std::chrono::steady_clock::now();
            duration = end - start;
            log(args, algorithm_id, generator_id, preprocessor_id, "selection", i,
                duration.count());

            // postprocessing (e.g., sorting if selection provides unsorted results)
            start = std::chrono::steady_clock::now();
            alg->postprocessing();
            end = std::chrono::steady_clock::now();
            duration = end - start;
            log(args, algorithm_id, generator_id, preprocessor_id, "postprocessing", i,
                duration.count());

            // finish
            start = std::chrono::steady_clock::now();
            const auto result = alg->finish();
            end = std::chrono::steady_clock::now();
            duration = end - start;
            log(args, algorithm_id, generator_id, preprocessor_id, "finish", i, duration.count());
            log(args, algorithm_id, generator_id, preprocessor_id, "transfer-out", i,
                alg->transfer_out_seconds());

            if (!no_output)
            {
                for (std::size_t j = 0; j < k; ++j)
                {
                    sink += result[j].index;
                }
                std::cerr << "Checksum: " << sink << '\n';
            }

            // verify the results using some other implementation
            if (params.count("verify") != 0)
            {
                for (auto& other_alg : algorithms)
                {
                    if (other_alg->id() != verify_alg)
                    {
                        continue;
                    }

                    other_alg->initialize(args);
                    other_alg->prepare();
                    other_alg->distances();
                    other_alg->selection();
                    const auto expected_result = other_alg->finish();
                    verify(expected_result, result, k);
                }
            }
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

    return 0;
}
catch (std::exception& e)
{
    std::cerr << e.what() << '\n';
    return 1;
}
catch (...)
{
    std::cerr << "Unknown exception" << '\n';
    return 1;
}
