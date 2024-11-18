#ifndef BITS_UTILS_HPP_
#define BITS_UTILS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

/** Parse a number from string with modifiers.
 *
 * If the string ends with k or K, the number will be multiplied by 2^10
 * If the string ends with m or M, the number will be multiplied by 2^20
 *
 * @param number String representation of a number
 *
 * @returns parsed number
 */
inline std::size_t parse_number(std::string number)
{
    if (number.empty())
    {
        return 0;
    }

    std::size_t multiplier = 1;
    const auto last_char = number[number.size() - 1];
    if (last_char == 'k' || last_char == 'K')
    {
        multiplier = 1U << 10U;
        number.pop_back();
    }
    else if (last_char == 'm' || last_char == 'M')
    {
        multiplier = 1U << 20U;
        number.pop_back();
    }
    else if (last_char == 'g' || last_char == 'G')
    {
        multiplier = 1U << 30U;
        number.pop_back();
    }

    return std::atol(number.c_str()) * multiplier;
}

inline std::array<std::size_t, 3> parse_dim3(std::string number)
{
    std::array<std::size_t, 3> result{1, 1, 1};

    std::vector<std::string> parts;
    parts.emplace_back("");
    std::size_t part_idx = 0;
    for (std::size_t i = 0; i < number.size(); ++i)
    {
        if (number[i] == ',')
        {
            parts.emplace_back("");
            ++part_idx;
        }
        else
        {
            parts[part_idx].push_back(number[i]);
        }
    }

    for (std::size_t i = 0; i < std::min<std::size_t>(3, parts.size()); ++i)
    {
        result[i] = parse_number(parts[i]);
    }

    return result;
}

#endif // BITS_UTILS_HPP_
