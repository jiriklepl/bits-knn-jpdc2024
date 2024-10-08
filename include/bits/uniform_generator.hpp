#ifndef UNIFORM_GENERATOR_HPP_
#define UNIFORM_GENERATOR_HPP_

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

#include "bits/data_generator.hpp"

/** Generate a random dataset.
 */
class uniform_generator : public data_generator
{
public:
    /** Initialize the RNG.
     *
     * @param seed seed for the RNG.
     */
    uniform_generator(std::size_t seed, float min, float max)
        : data_generator{seed}, min_{min}, max_{max}
    {
        uniform_generator::set_seed(seed);
    }

    uniform_generator(std::size_t seed, float max) : uniform_generator{seed, 0, max} {}

    uniform_generator(std::size_t seed) : uniform_generator{seed, 0, 1} {}

    uniform_generator() : uniform_generator{0} {}

    std::string id() const override { return "uniform"; }

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    std::vector<float> generate(std::size_t count, std::size_t dim) override
    {
        std::uniform_real_distribution<float> data_dist{min_, max_};

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

    void set_params(float min, float max)
    {
        if (min >= max)
        {
            throw std::invalid_argument("Invalid parameter values");
        }

        min_ = min;
        max_ = max;
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
    float min_, max_;
    std::default_random_engine engine_;
};

#endif // UNIFORM_GENERATOR_HPP_
