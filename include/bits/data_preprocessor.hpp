#ifndef DATA_PREPROCESSOR_HPP_
#define DATA_PREPROCESSOR_HPP_

#include <cstddef>
#include <string>
#include <vector>

class data_preprocessor
{
public:
    data_preprocessor() = default;

    // Copying data_generators is disallowed
    data_preprocessor(const data_preprocessor&) = delete;
    data_preprocessor& operator=(const data_preprocessor&) = delete;

    data_preprocessor(data_preprocessor&&) noexcept = default;
    data_preprocessor& operator=(data_preprocessor&&) noexcept = default;

    virtual ~data_preprocessor() = default;

    virtual std::string id() const = 0;

    virtual void preprocess(std::vector<float>& data, std::size_t dim) = 0;

    virtual void preprocess(std::vector<float>& data, std::vector<float>& query,
                            std::size_t dim) = 0;
};

#endif // DATA_PREPROCESSOR_HPP_
