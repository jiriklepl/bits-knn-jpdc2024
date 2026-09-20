#include <algorithm>
#include <bit>
#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark.hpp"
#include "bits/applications/database_topn.hpp"
#include "bits/cuda_knn.hpp"
#include "bits/cuda_stream.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/utils.hpp"

using namespace applications::benchmark;

int main(int argc, char** argv)
try
{
    cxxopts::Options options(
        "database-topn",
        "GPU-resident FP32 top-N operator; normally launched through run-applications.py");
    options.add_options()("prices", "Little-endian FP32 price column",
                          cxxopts::value<std::string>())(
        "discounts", "Little-endian FP32 discount column", cxxopts::value<std::string>())(
        "payload", "Little-endian FP32 projected payload", cxxopts::value<std::string>())(
        "row-ids", "Little-endian uint64 row IDs", cxxopts::value<std::string>())(
        "rows", "Logical rows in one table", cxxopts::value<std::string>())(
        "k,topn", "Output rows", cxxopts::value<std::string>()->default_value("32"))(
        "backends", "Comma-separated comparison backends",
        cxxopts::value<std::string>()->default_value("bits-sq,air-topk,grid-select,block-select"))(
        "degree", "BITS split degree", cxxopts::value<std::string>()->default_value("32"))(
        "bits-block-size", "BITS thread block size: 128,256,512",
        cxxopts::value<std::string>()->default_value("512"))(
        "items-per-thread",
        "BITS batch override: 1,4,7,16 (default: bits/bits-prefetch=7, bits-sq=4)",
        cxxopts::value<std::string>())("repeat", "Measured repetitions",
                                       cxxopts::value<std::string>()->default_value("20"))(
        "warmup", "Untimed warmup repetitions", cxxopts::value<std::string>()->default_value("3"))(
        "dataset-id", "Manifest SHA-256",
        cxxopts::value<std::string>()->default_value("unmanifested"))(
        "output", "Write last verified projected rows for each backend as CSV",
        cxxopts::value<std::string>())("help", "Show help");
    const auto params = options.parse(argc, argv);
    if (params.count("help"))
    {
        std::cout << options.help() << '\n';
        return 0;
    }
    const auto n = parse_number(params["rows"].as<std::string>());
    const auto k = parse_number(params["k"].as<std::string>());
    const auto degree = parse_number(params["degree"].as<std::string>());
    std::optional<std::size_t> items;
    if (params.count("items-per-thread"))
        items = parse_number(params["items-per-thread"].as<std::string>());
    const auto bits_block_size = parse_number(params["bits-block-size"].as<std::string>());
    const auto repeat = parse_number(params["repeat"].as<std::string>());
    const auto warmup = parse_number(params["warmup"].as<std::string>());
    const auto dataset = params["dataset-id"].as<std::string>();
    if (n == 0 || n > std::size_t{std::numeric_limits<std::int32_t>::max()} || k == 0 || k > n ||
        k > 2048 || repeat == 0)
        throw std::invalid_argument{"Require int32 rows, 0 < k <= min(rows, 2048), repeat > 0"};
    if (dataset != "unmanifested" &&
        (dataset.size() != 64 ||
         dataset.find_first_not_of("0123456789abcdef") != std::string::npos))
        throw std::invalid_argument{"dataset-id must be a lowercase SHA-256"};

    applications::database_columns columns;
    columns.price = read_column<float>(params["prices"].as<std::string>(), n);
    columns.discount = read_column<float>(params["discounts"].as<std::string>(), n);
    columns.payload = read_column<float>(params["payload"].as<std::string>(), n);
    columns.row_id = read_column<std::uint64_t>(params["row-ids"].as<std::string>(), n);
    const auto expected = applications::reference_topn(columns, k);
    applications::database_topn query(n, k);
    const auto upload_seconds = measure([&] { query.upload(columns); });

    std::vector<backend> backends;
    std::set<std::string> seen;
    const auto names = params["backends"].as<std::string>();
    if (names.empty() || names.back() == ',')
        throw std::invalid_argument{"Empty backend name"};
    std::istringstream input(names);
    for (std::string name; std::getline(input, name, ',');)
    {
        if (!seen.insert(name).second)
            throw std::invalid_argument{"Duplicate backend: " + name};
        if (name == "bits-sq" && (degree == 0 || degree > n))
            throw std::invalid_argument{"BITS split degree must be between 1 and rows"};
        auto item = make_backend(name, k, degree, items, bits_block_size);
        knn_args args{};
        args.query_count = 1;
        args.point_count = n;
        args.k = k;
        args.deg = item.degree;
        args.selection_block_size = item.block_size;
        args.items_per_thread = {item.items, 1, 1};
        item.selector->set_dist_impl(std::make_unique<precomputed_scores>(query.scores()));
        item.selector->initialize(args);
        backends.push_back(std::move(item));
    }
    std::vector<applications::database_row> output(k);
    std::vector<std::vector<applications::database_row>> final_output(backends.size());
    auto pipeline = [&](backend& item) {
        query.transform();
        item.selector->selection();
        query.output(item.selector->out_dist_gpu(), item.selector->out_label_gpu(), item.sorted);
    };
    for (std::size_t i = 0; i < warmup; ++i)
        for (auto& item : backends)
            measure([&] { pipeline(item); });

    cudaDeviceProp properties{};
    int device = 0;
    CUCH(cudaGetDevice(&device));
    CUCH(cudaGetDeviceProperties(&properties, device));
    int runtime_version = 0, driver_version = 0;
    CUCH(cudaRuntimeGetVersion(&runtime_version));
    CUCH(cudaDriverGetVersion(&driver_version));
    std::cerr << "GPU: " << properties.name << "; CUDA runtime: " << runtime_version
              << "; CUDA driver: " << driver_version
              << "; one query; score=FP32(price * FP32(1-discount)); verification enabled\n";
    std::cout << "dataset_id,backend,rows,k,retention_ratio,degree,block_size,items_per_thread,"
                 "iteration,phase,seconds\n";
    std::cout << std::setprecision(17);
    auto log = [&](const backend& item, long iteration, const char* phase, double seconds) {
        std::cout << dataset << ',' << item.selector->id() << ',' << n << ',' << k << ','
                  << static_cast<double>(k) / n << ',' << item.degree << ',' << item.block_size
                  << ',' << item.items << ',' << iteration << ',' << phase << ',' << seconds
                  << '\n';
    };
    for (const auto& item : backends)
        log(item, -1, "upload_shared", upload_seconds);
    for (std::size_t iteration = 0; iteration < repeat; ++iteration)
    {
        // Rotate the starting backend; no repeated allocation or input duplication.
        for (std::size_t offset = 0; offset < backends.size(); ++offset)
        {
            const auto index = (iteration + offset) % backends.size();
            auto& item = backends[index];
            const auto total = measure([&] { pipeline(item); });
            const auto download = measure([&] { query.download(output); });
            applications::verify_topn(columns, expected, output);
            log(item, iteration, "operator", total);
            log(item, iteration, "download", download);
            // Independent profiling invocation; do not reconstruct operator latency from these.
            log(item, iteration, "transform_isolated", measure([&] { query.transform(); }));
            log(item, iteration, "selection_isolated",
                measure([&] { item.selector->selection(); }));
            log(item, iteration, "output_isolated", measure([&] {
                    query.output(item.selector->out_dist_gpu(), item.selector->out_label_gpu(),
                                 item.sorted);
                }));
            measure([&] { query.download(output); });
            applications::verify_topn(columns, expected, output);
            if (iteration + 1 == repeat)
                final_output[index] = output;
        }
    }
    if (params.count("output"))
    {
        const auto path = params["output"].as<std::string>();
        if (std::filesystem::exists(path))
            throw std::invalid_argument{"Output file already exists; choose a new file"};
        std::ofstream file(path);
        if (!file)
            throw std::runtime_error{"Cannot open output file"};
        file << "backend,rank,row_id,score,payload,source_index\n" << std::setprecision(9);
        for (std::size_t i = 0; i < backends.size(); ++i)
            for (std::size_t rank = 0; rank < k; ++rank)
            {
                const auto& row = final_output[i][rank];
                file << backends[i].selector->id() << ',' << rank << ',' << row.row_id << ','
                     << row.score << ',' << row.payload << ',' << row.source_index << '\n';
            }
        if (!file)
            throw std::runtime_error{"Writing output failed"};
    }
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
