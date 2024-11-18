#ifndef BITS_GENERATOR_DATA_GENERATOR_HPP_
#define BITS_GENERATOR_DATA_GENERATOR_HPP_

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/** Generate a dataset.
 */
class data_generator
{
public:
    data_generator() = default;

    explicit data_generator([[maybe_unused]] std::size_t seed) {}

    // Copying data_generators is disallowed
    data_generator(const data_generator&) = delete;
    data_generator& operator=(const data_generator&) = delete;

    data_generator(data_generator&&) noexcept = default;
    data_generator& operator=(data_generator&&) noexcept = default;

    virtual ~data_generator() = default;

    virtual std::string id() const = 0;

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    virtual std::vector<float> generate(std::size_t count, std::size_t dim) = 0;
    virtual void set_seed(std::size_t seed) = 0;
    virtual void set_params(std::string_view params) = 0;

protected:
    static std::vector<float> parse_floats(std::string_view params)
    {
        std::vector<float> result;
        const char* begin = params.data();
        const char* const end = params.data() + params.size();
        char* next = nullptr;
        float value = 0;

        while (begin < end)
        {
            value = std::strtof(begin, &next);

            if (next == begin)
            {
                throw std::invalid_argument("Invalid float parameter");
            }

            if (next + 1 < end && *next != ',')
            {
                throw std::invalid_argument("Invalid parameter format");
            }

            begin = next + 1;
            result.push_back(value);
        }

        return result;
    }

    static std::vector<long> parse_ints(std::string_view params, int base = 0)
    {
        std::vector<long> result;
        const char* begin = params.data();
        const char* const end = params.data() + params.size();
        char* next = nullptr;
        long value = 0;

        while (begin < end)
        {
            value = std::strtol(begin, &next, base);

            if (next == begin)
            {
                throw std::invalid_argument("Invalid int parameter");
            }

            if (next + 1 < end && *next != ',')
            {
                throw std::invalid_argument("Invalid parameter format");
            }

            begin = next + 1;
            result.push_back(value);
        }

        return result;
    }
};

#endif // BITS_GENERATOR_DATA_GENERATOR_HPP_
