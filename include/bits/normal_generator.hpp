#ifndef NORMAL_GENERATOR_HPP_
#define NORMAL_GENERATOR_HPP_

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bits/data_generator.hpp"

/** Generate a random dataset.
 */
class normal_generator : public data_generator
{
public:
    /** Initialize the RNG.
     *
     * @param seed seed for the RNG.
     */
    explicit normal_generator(std::size_t seed, float mean, float std_dev)
        : data_generator{seed}, mean_{mean}, std_dev_{std_dev}
    {
        set_seed(seed);
    }

    normal_generator(std::size_t seed, float std_dev) : normal_generator{seed, 0, std_dev} {}

    normal_generator(std::size_t seed) : normal_generator{seed, 0, 1} {}

    normal_generator() : normal_generator{0} {}

    std::string id() const override { return "normal"; }

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    std::vector<float> generate(std::size_t count, std::size_t dim) override
    {
        std::normal_distribution<float> data_dist{mean_, std_dev_};

        std::vector<float> data(dim * count);
        for (auto& element : data)
        {
            element = data_dist(engine_);
        }
        return data;
    }

    void set_seed(std::size_t seed) override
    {
        std::random_device dev;
        engine_ = std::default_random_engine{seed == 0 ? dev() : seed};
        engine_.discard(1 << 12);
    }

    void set_params(float mean, float std_dev)
    {
        if (std_dev <= 0)
        {
            throw std::invalid_argument("Invalid standard deviation");
        }

        mean_ = mean;
        std_dev_ = std_dev;
    }

    void set_params(std::string_view params) override
    {
        const auto floats = parse_floats(params);
        if (floats.size() > 2)
        {
            throw std::invalid_argument("Invalid number of parameters");
        }

        switch (floats.size())
        {
        case 2:
            set_params(floats[0], floats[1]);
            break;
        case 1:
            set_params(0, floats[0]);
            break;
        default:
            break;
        }
    }

private:
    float mean_, std_dev_;
    std::default_random_engine engine_;
};

#endif // NORMAL_GENERATOR_HPP_
