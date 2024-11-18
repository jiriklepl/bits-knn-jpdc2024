#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "bits/array_view.hpp"

TEST_CASE("Get elements", "[array_view]")
{
    std::size_t rows = 3;
    std::size_t cols = 5;
    std::size_t stride = 8;

    std::vector<int> data(rows * stride);
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            data[i * stride + j] = i * cols + j;
        }
    }

    array_view<int, 2> view(data.data(), {rows, cols}, {stride, 1});
    REQUIRE(view.size() == rows * stride);
    REQUIRE(view.size(0) == rows);
    REQUIRE(view.size(1) == cols);

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            const auto value = view({i, j});
            REQUIRE((std::size_t)value == i * cols + j);
        }
    }
}
