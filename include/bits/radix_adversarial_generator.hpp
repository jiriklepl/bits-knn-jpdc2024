#ifndef RADIX_ADVERSARIAL_GENERATOR_HPP_
#define RADIX_ADVERSARIAL_GENERATOR_HPP_

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>

#include "bits/data_generator.hpp"

/** Generate a random dataset.
 */
class radix_adversarial_generator : public data_generator
{
public:
    /** Initialize the RNG.
     *
     * @param seed seed for the RNG.
     */
    explicit radix_adversarial_generator(std::size_t seed, int bits)
        : data_generator{seed}, bits_{bits}
    {
        set_seed(seed);
    }

    radix_adversarial_generator(std::size_t seed) : radix_adversarial_generator{seed, 12} {}

    radix_adversarial_generator() : radix_adversarial_generator{0} {}

    std::string id() const override { return "radix-adversarial"; }

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    std::vector<float> generate(std::size_t count, std::size_t dim) override
    {
        std::vector<float> data(dim * count);

        // FIXME: this is evil (all of it)
        volatile union
        {
            float f;
            std::uintptr_t i;
        } representation;

        representation.i = 0;
        representation.f = 1.0f;

        const std::uintptr_t mask = (1u << bits_) - 1;

        const std::uintptr_t top_bits = representation.i & ~mask;

        representation.f = 0.0f; // 0x00000000 in IEEE 754

        for (auto& element : data)
        {
            representation.i = engine_();

            if constexpr (sizeof(std::uintptr_t) > sizeof(engine_()))
            {
                representation.i = (representation.i << (sizeof(engine_()) * 8)) | engine_();
            }

            representation.i = top_bits | (representation.i & mask);

            element = representation.f;
        }

        return data;
    }

    void set_seed(std::size_t seed) override
    {
        std::random_device dev;
        engine_ = std::default_random_engine{seed == 0 ? dev() : seed};
        engine_.discard(1 << 12);
    }

    void set_params(int bits)
    {
        if (bits < 0)
        {
            throw std::invalid_argument("Invalid number of bits");
        }

        bits_ = bits;
    }

    void set_params(std::string_view params) override
    {
        const auto values = parse_ints(params);
        if (values.size() > 1)
        {
            throw std::invalid_argument("Invalid number of parameters");
        }

        if (values.size() == 1)
        {
            set_params(values[0]);
        }
    }

private:
    int bits_;
    std::default_random_engine engine_;
};

#endif // RADIX_ADVERSARIAL_GENERATOR_HPP_
