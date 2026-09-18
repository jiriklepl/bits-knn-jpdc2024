#include <limits>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "bits/utils.hpp"

TEST_CASE("Numeric configuration rejects malformed and overflowing values", "[config]")
{
    REQUIRE(parse_number("0") == 0);
    REQUIRE(parse_number("32") == 32);
    REQUIRE(parse_number("2K") == 2048);
    REQUIRE(parse_number("3m") == 3 * (std::size_t{1} << 20));
    REQUIRE(parse_number(std::to_string(std::numeric_limits<std::size_t>::max())) ==
            std::numeric_limits<std::size_t>::max());
    for (const auto* text : {"", "-1", "12garbage", "k", "1.5", "1k2"})
    {
        CAPTURE(text);
        REQUIRE_THROWS_AS(parse_number(text), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(parse_number(std::to_string(std::numeric_limits<std::size_t>::max()) + "0"),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(parse_number(std::to_string(std::numeric_limits<std::size_t>::max()) + "k"),
                      std::invalid_argument);
}

TEST_CASE("Items-per-thread parsing retains defaults and rejects discarded input", "[config]")
{
    REQUIRE(parse_dim3("4") == (std::array<std::size_t, 3>{4, 1, 1}));
    REQUIRE(parse_dim3("4,8") == (std::array<std::size_t, 3>{4, 8, 1}));
    REQUIRE(parse_dim3("4,8,2") == (std::array<std::size_t, 3>{4, 8, 2}));
    for (const auto* text : {"", "4,", ",4", "4,,2", "4,8,2,9"})
    {
        CAPTURE(text);
        REQUIRE_THROWS_AS(parse_dim3(text), std::invalid_argument);
    }
}
