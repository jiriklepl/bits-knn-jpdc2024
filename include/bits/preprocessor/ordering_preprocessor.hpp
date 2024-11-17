#ifndef ORDERING_PREPROCESSOR_HPP_
#define ORDERING_PREPROCESSOR_HPP_

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "bits/preprocessor/data_preprocessor.hpp"

template <typename Ordering = std::less<>>
class ordering_preprocessor : public data_preprocessor
{
public:
    ordering_preprocessor() = default;

    static_assert(std::is_same_v<Ordering, std::less<>> || std::is_same_v<Ordering, std::greater<>>,
                  "Ordering must be std::less<> or std::greater<>");

    std::string id() const override;

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    void preprocess([[maybe_unused]] std::vector<float>& data,
                    [[maybe_unused]] std::size_t dim) override
    {
        throw std::runtime_error("Not implemented");
    }

    void preprocess(std::vector<float>& data, std::vector<float>& query,
                    [[maybe_unused]] std::size_t dim) override
    {
        // cheat: set all query vectors to 0 and sort the data
        std::fill(query.begin(), query.end(), 0);
        std::sort(data.begin(), data.end(), Ordering{});
    }
};

template <>
inline std::string ordering_preprocessor<std::less<>>::id() const
{
    return "ascending";
}

template <>
inline std::string ordering_preprocessor<std::greater<>>::id() const
{
    return "descending";
}

#endif // ORDERING_PREPROCESSOR_HPP_
