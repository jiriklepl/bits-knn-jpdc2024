#ifndef DYNAMIC_SWITCH_HPP_
#define DYNAMIC_SWITCH_HPP_

template <std::size_t... Options, typename F, typename... Args, typename Option>
inline bool dynamic_switch(Option option, F&& f, Args&&... args)
{
    return (
        (option == Options &&
         (std::forward<F>(f).template operator()<Options>(std::forward<Args>(args)...), true)) ||
        ...);
}

template <std::size_t... Options, typename F, typename... Args, typename Option>
inline bool dynamic_switch_le(Option option, F&& f, Args&&... args)
{
    return (
        (option <= Options &&
         (std::forward<F>(f).template operator()<Options>(std::forward<Args>(args)...), true)) ||
        ...);
}

#endif // DYNAMIC_SWITCH_HPP_
