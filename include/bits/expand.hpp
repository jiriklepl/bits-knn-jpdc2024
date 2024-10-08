#ifndef EXPAND_HPP_
#define EXPAND_HPP_

#include <cstdint>

template <std::int32_t... T>
struct fixed
{
};

template <std::int32_t... T>
struct choice
{
};

template <std::int32_t... Fixed, std::int32_t... Args, typename... Tail, typename Function>
inline void expand(Function f, fixed<Fixed...>, choice<Args...> /* head */, Tail... tail)
{
    (expand(f, fixed<Fixed..., Args>{}, tail...), ...);
}

template <std::int32_t... Fixed, typename Function>
inline void expand(Function f, fixed<Fixed...>)
{
    f.template operator()<Fixed...>();
}

/** Call function @p f with all combination of configuration parameters
 */
template <typename... Choices, typename Function>
inline void expand(Function f)
{
    expand(f, fixed<>{}, Choices{}...);
}

#endif // EXPAND_HPP_
