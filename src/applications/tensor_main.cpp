#include <cmath>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>

#include "benchmark.hpp"
#include "bits/applications/gradient_compression.hpp"
#include "bits/applications/token_sampling.hpp"
#include "bits/distance/precomputed_scores.hpp"
#include "bits/utils.hpp"

using namespace applications;
using namespace applications::benchmark;

namespace
{
#ifdef APPLICATION_TOKEN_SAMPLING
constexpr bool sampling = true;
constexpr const char* operator_name = "token-sampling";
#else
constexpr bool sampling = false;
constexpr const char* operator_name = "gradient-compression";
#endif

[[maybe_unused]] float parse_temperature(const std::string& text)
{
    std::size_t consumed = 0;
    const auto parsed = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed) || parsed <= 0 ||
        parsed > std::numeric_limits<float>::max())
        throw std::invalid_argument{"temperature must be a positive finite FP32 value"};
    const auto value = static_cast<float>(parsed);
    if (value == 0)
        throw std::invalid_argument{"temperature must be a positive finite FP32 value"};
    return value;
}

struct sampling_pipeline
{
    std::vector<float> input, expected, probabilities;
    std::vector<std::int32_t> indices;
    std::vector<sampled_token> result;
    std::size_t batch, rows, k;
    float temperature;
    std::uint64_t seed;
    token_sampling operation;

    sampling_pipeline(std::vector<float> values, std::size_t b, std::size_t n, std::size_t count,
                      float temp, std::uint64_t random_seed)
        : input(std::move(values)), expected(reference_sampling_topk(input, b, n, count)),
          probabilities(b * count), indices(b * count), result(b), batch(b), rows(n), k(count),
          temperature(temp), seed(random_seed), operation(b, n, count, temp)
    {
        validate_logits(input, b, n, count, temp);
    }

    void upload() { operation.upload(input); }

    void transform() { operation.transform(); }

    auto scores() const { return operation.scores(); }

    void output(backend& item, std::uint64_t draw)
    {
        operation.output(item.selector->out_dist_gpu(), item.selector->out_label_gpu(), seed, draw);
    }

    void download() { operation.download(result); }

    void verify(std::uint64_t draw)
    {
        // Candidate distributions are validation data, outside the result-transfer timing.
        operation.download_distribution(indices, probabilities);
        cuda_stream::make_default().sync();
        verify_sampling(input, batch, rows, k, temperature, expected, indices, probabilities,
                        result, seed, draw);
    }

    static void output_header(std::ostream& stream)
    {
        stream << "backend,sequence,token,probability\n";
    }

    void write_output(std::ostream& stream, const std::string& name) const
    {
        for (std::size_t i = 0; i < result.size(); ++i)
            stream << name << ',' << i << ',' << result[i].token << ',' << result[i].probability
                   << '\n';
    }
};

struct gradient_pipeline
{
    std::vector<float> input, expected;
    std::vector<gradient_entry> result;
    gradient_compression operation;

    gradient_pipeline(std::vector<float> values, std::size_t n, std::size_t k)
        : input(std::move(values)), expected(reference_gradient_topk(input, k)), result(k),
          operation(n, k)
    {
    }

    void upload() { operation.upload(input); }

    void transform() { operation.transform(); }

    auto scores() const { return operation.scores(); }

    void output(backend& item, std::uint64_t)
    {
        operation.output(item.selector->out_dist_gpu(), item.selector->out_label_gpu());
    }

    void download() { operation.download(result); }

    void verify(std::uint64_t) { verify_gradient_topk(input, expected, result); }

    static void output_header(std::ostream& stream) { stream << "backend,index,value\n"; }

    void write_output(std::ostream& stream, const std::string& name) const
    {
        for (const auto& entry : result)
            stream << name << ',' << entry.index << ',' << entry.value << '\n';
    }
};

template <class Pipeline>
int run_benchmark(Pipeline& pipeline, const cxxopts::ParseResult& params, std::size_t rows,
                  std::size_t batch, std::size_t k, float temperature, std::uint64_t seed)
{
    const auto repeat = parse_number(params["repeat"].as<std::string>());
    const auto warmup = parse_number(params["warmup"].as<std::string>());
    const auto degree = parse_number(params["degree"].as<std::string>());
    const auto block = parse_number(params["bits-block-size"].as<std::string>());
    const auto dataset = params["dataset-id"].as<std::string>();
    if (repeat == 0 || repeat > static_cast<std::size_t>(std::numeric_limits<long>::max()))
        throw std::invalid_argument{"repeat must be positive and fit a signed iteration counter"};
    if (dataset != "unmanifested" &&
        (dataset.size() != 64 ||
         dataset.find_first_not_of("0123456789abcdef") != std::string::npos))
        throw std::invalid_argument{"dataset-id must be a lowercase SHA-256"};
    std::optional<std::size_t> items;
    if (params.count("items-per-thread"))
        items = parse_number(params["items-per-thread"].as<std::string>());

    std::ostringstream selected_output;
    std::optional<std::string> output_path;
    if (params.count("output"))
    {
        const auto path = params["output"].as<std::string>();
        if (std::filesystem::exists(path))
            throw std::invalid_argument{"Output file already exists; choose a new file"};
        output_path = path;
        Pipeline::output_header(selected_output);
        selected_output << std::setprecision(std::numeric_limits<float>::max_digits10);
    }

    std::vector<backend> backends;
    std::set<std::string> seen;
    const auto names = params["backends"].as<std::string>();
    if (names.empty() || names.back() == ',')
        throw std::invalid_argument{"Empty backend name"};
    std::istringstream list(names);
    for (std::string name; std::getline(list, name, ',');)
    {
        if (!seen.insert(name).second)
            throw std::invalid_argument{"Duplicate backend: " + name};
        if (name == "bits-sq" && (degree == 0 || degree > rows))
            throw std::invalid_argument{"bits split degree must be between 1 and candidate count"};
        auto item = make_backend(name, k, degree, items, block);
        knn_args args{};
        args.point_count = rows;
        args.query_count = batch;
        args.k = k;
        args.deg = item.degree;
        args.selection_block_size = item.block_size;
        args.items_per_thread = {item.items, 1, 1};
        item.selector->set_dist_impl(std::make_unique<precomputed_scores>(pipeline.scores()));
        item.selector->initialize(args);
        backends.push_back(std::move(item));
    }
    const auto upload = measure([&] { pipeline.upload(); });
    auto run = [&](backend& item, std::uint64_t draw) {
        pipeline.transform();
        item.selector->selection();
        pipeline.output(item, draw);
    };
    for (std::size_t i = 0; i < warmup; ++i)
        for (auto& item : backends)
            measure([&] { run(item, i); });

    cudaDeviceProp properties{};
    int device = 0, runtime = 0, driver = 0;
    CUCH(cudaGetDevice(&device));
    CUCH(cudaGetDeviceProperties(&properties, device));
    CUCH(cudaRuntimeGetVersion(&runtime));
    CUCH(cudaDriverGetVersion(&driver));
    std::cerr << "GPU: " << properties.name << "; CUDA runtime: " << runtime
              << "; CUDA driver: " << driver << "; operator: " << operator_name
              << "; verification enabled\n";
    // Publish timings only after all results and optional output have been verified.
    std::ostringstream timings;
    timings << "operator,dataset_id,backend,rows,batch_size,k,retention_ratio,degree,block_size,"
               "items_per_thread,temperature,seed,iteration,phase,seconds\n"
            << std::setprecision(17);
    auto log = [&](const backend& item, long iteration, const char* phase, double seconds) {
        timings << operator_name << ',' << dataset << ',' << item.selector->id() << ',' << rows
                << ',' << batch << ',' << k << ',' << static_cast<double>(k) / rows << ','
                << item.degree << ',' << item.block_size << ',' << item.items << ',' << temperature
                << ',' << seed << ',' << iteration << ',' << phase << ',' << seconds << '\n';
    };
    for (const auto& item : backends)
        log(item, -1, "upload_shared", upload);
    for (std::size_t iteration = 0; iteration < repeat; ++iteration)
    {
        for (std::size_t offset = 0; offset < backends.size(); ++offset)
        {
            auto& item = backends[(iteration + offset) % backends.size()];
            const auto total = measure([&] { run(item, iteration); });
            const auto download = measure([&] { pipeline.download(); });
            pipeline.verify(iteration);
            log(item, iteration, "operator", total);
            log(item, iteration, "download", download);
            log(item, iteration, "transform_isolated", measure([&] { pipeline.transform(); }));
            log(item, iteration, "selection_isolated",
                measure([&] { item.selector->selection(); }));
            log(item, iteration, "output_isolated",
                measure([&] { pipeline.output(item, iteration); }));
            measure([&] { pipeline.download(); });
            pipeline.verify(iteration);
            if (iteration + 1 == repeat && output_path.has_value())
                pipeline.write_output(selected_output, item.selector->id());
        }
    }
    if (output_path.has_value())
    {
        std::ofstream file(*output_path);
        file << selected_output.str();
        file.close();
        if (!file)
            throw std::runtime_error{"Writing output failed"};
    }
    std::cout << timings.str();
    return 0;
}
} // namespace

int main(int argc, char** argv)
try
{
    cxxopts::Options options(operator_name, "GPU-resident operator over a captured model tensor");
    options.add_options()("k,topk", "Selected candidates",
                          cxxopts::value<std::string>()->default_value("32"))(
        "backends", "Comma-separated backends",
        cxxopts::value<std::string>()->default_value("bits-sq,air-topk,grid-select,block-select"))(
        "degree", "bits split degree", cxxopts::value<std::string>()->default_value("32"))(
        "bits-block-size", "bits block size", cxxopts::value<std::string>()->default_value("512"))(
        "items-per-thread",
        "bits batch override: 1,4,7,8,13,16 (default: bits/bits-prefetch=7, bits-sq=4)",
        cxxopts::value<std::string>())("repeat", "Measured repetitions",
                                       cxxopts::value<std::string>()->default_value("20"))(
        "warmup", "Untimed warmups", cxxopts::value<std::string>()->default_value("3"))(
        "dataset-id", "Manifest SHA-256",
        cxxopts::value<std::string>()->default_value("unmanifested"))(
        "output", "Final verified outputs as CSV", cxxopts::value<std::string>())("help",
                                                                                  "Show help");
    if constexpr (sampling)
        options.add_options()("logits", "Little-endian FP32 logits", cxxopts::value<std::string>())(
            "batch-size", "Number of sequences", cxxopts::value<std::string>())(
            "vocabulary-size", "Candidates per sequence",
            cxxopts::value<std::string>())("temperature", "Positive FP32 temperature",
                                           cxxopts::value<std::string>()->default_value("1"))(
            "seed", "Sampling seed", cxxopts::value<std::string>()->default_value("42"));
    else
        options.add_options()("gradient", "Little-endian FP32 gradient",
                              cxxopts::value<std::string>())(
            "elements", "Complete tensor element count", cxxopts::value<std::string>());
    const auto params = options.parse(argc, argv);
    if (params.count("help"))
    {
        std::cout << options.help() << '\n';
        return 0;
    }
    const auto k = parse_number(params["k"].as<std::string>());
    const auto rows =
        parse_number(params[sampling ? "vocabulary-size" : "elements"].as<std::string>());
    const auto batch = sampling ? parse_number(params["batch-size"].as<std::string>()) : 1;
    const auto max_index = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if (rows == 0 || rows > max_index || batch == 0 || batch > max_index / rows || k == 0 ||
        k > std::min(rows, std::size_t{2048}))
        throw std::invalid_argument{
            "Require positive int32 tensor size and 0 < k <= min(candidates,2048)"};
    auto values = read_column<float>(params[sampling ? "logits" : "gradient"].as<std::string>(),
                                     rows * batch);
    if constexpr (sampling)
    {
        const auto temperature = parse_temperature(params["temperature"].as<std::string>());
        const auto seed = parse_number(params["seed"].as<std::string>());
        sampling_pipeline pipeline(std::move(values), batch, rows, k, temperature, seed);
        return run_benchmark(pipeline, params, rows, batch, k, temperature, seed);
    }
    else
    {
        gradient_pipeline pipeline(std::move(values), rows, k);
        return run_benchmark(pipeline, params, rows, batch, k, 0, 0);
    }
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
