#include <limits>
#include <vector>

#include "bits/score_validation.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Application scores reject invalid values before GPU execution", "[score-validation]")
{
    const auto inf = std::numeric_limits<float>::infinity();
    std::vector<float> scores{-3, 2, inf, 1, 0, -2};
    REQUIRE_NOTHROW(validate_score_values(scores, 2, 3, 2));
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(validate_score_values(scores, 1, 3, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(validate_score_values(scores, 0, 3, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 4), std::invalid_argument);
    REQUIRE_THROWS_AS(validate_score_values(scores, std::numeric_limits<std::size_t>::max(), 3, 1),
                      std::invalid_argument);
    scores[2] = -inf;
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 2), std::invalid_argument);
    scores[2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 2), std::invalid_argument);
    scores = {inf, inf, inf, 1, 2, 3};
    REQUIRE_THROWS_AS(validate_score_values(scores, 2, 3, 1), std::invalid_argument);
    scores = {0, 0, 0, -2, -2, -2};
    REQUIRE_NOTHROW(validate_score_values(scores, 2, 3, 3));
}
