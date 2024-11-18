#ifndef BITS_PREPROCESSOR_IDENTITY_PREPROCESSOR_HPP_
#define BITS_PREPROCESSOR_IDENTITY_PREPROCESSOR_HPP_

#include <cstddef>
#include <string>
#include <vector>

#include "bits/preprocessor/data_preprocessor.hpp"

class identity_preprocessor : public data_preprocessor
{
public:
    identity_preprocessor() = default;

    std::string id() const override { return "identity"; }

    /** Generate @p count random vectors with dimension @p dim
     *
     * @param count number of vectors to generate
     * @param dim dimension of vectors
     * @return random vectors
     */
    void preprocess([[maybe_unused]] std::vector<float>& data,
                    [[maybe_unused]] std::size_t dim) override
    {
    }

    void preprocess([[maybe_unused]] std::vector<float>& data,
                    [[maybe_unused]] std::vector<float>& query,
                    [[maybe_unused]] std::size_t dim) override
    {
    }
};

#endif // BITS_PREPROCESSOR_IDENTITY_PREPROCESSOR_HPP_
