#pragma once
// A minimal clone of llvm::StringSwitch (llvm/ADT/StringSwitch.h):
//
//   auto kind = StringSwitch<Kind>(name)
//                   .Case("add", Kind::Add)
//                   .Cases("sub", "subtract", Kind::Sub)
//                   .Default(Kind::Unknown);
//
// The first matching Case wins. All case values are evaluated eagerly (as in LLVM), so keep them cheap.
#include <optional>
#include <string_view>
#include <utility>

template <typename T>
class StringSwitch {
public:
    explicit StringSwitch(std::string_view str) : str_(str) {}

    StringSwitch(const StringSwitch &) = delete;
    StringSwitch &operator=(const StringSwitch &) = delete;

    StringSwitch &Case(std::string_view s, T value) {
        if (!result_ && str_ == s)
            result_.emplace(std::move(value));
        return *this;
    }

    StringSwitch &Cases(std::string_view s0, std::string_view s1, T value) {
        return CasesImpl(std::move(value), s0, s1);
    }

    StringSwitch &Cases(std::string_view s0, std::string_view s1, std::string_view s2, T value) {
        return CasesImpl(std::move(value), s0, s1, s2);
    }

    // Returns the matched value, or `value` when no Case matched.
    T Default(T value) {
        if (result_)
            return std::move(*result_);
        return value;
    }

    // Returns the matched value, if any.
    std::optional<T> Optional() { return std::move(result_); }

private:
    template <typename... Ss>
    StringSwitch &CasesImpl(T value, Ss... ss) {
        if (!result_ && ((str_ == ss) || ...))
            result_.emplace(std::move(value));
        return *this;
    }

    std::string_view str_;
    std::optional<T> result_;
};
