#include "tradereview/drawing/FibSettings.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tradereview::drawing {
namespace {

[[nodiscard]] std::string trim(std::string value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

[[nodiscard]] double parse_level(const std::string& token)
{
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(token, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid Fibonacci level: " + token);
    }
    if (consumed != token.size()) {
        throw std::invalid_argument("Invalid Fibonacci level: " + token);
    }
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument("Fibonacci levels must be finite and non-negative");
    }
    return value;
}

[[nodiscard]] std::vector<double> normalize_levels(std::vector<double> levels)
{
    for (const auto level : levels) {
        if (!std::isfinite(level) || level < 0.0) {
            throw std::invalid_argument("Fibonacci levels must be finite and non-negative");
        }
    }
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    return levels;
}

} // namespace

std::vector<double> FibLevelsConfig::effective_levels() const
{
    return merge_fib_levels(enabled_levels, custom_levels_text);
}

std::vector<double> default_retracement_presets()
{
    return {0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8};
}

std::vector<double> default_extension_presets()
{
    return {0.618, 1.0, 1.272, 1.618, 2.0};
}

std::vector<double> merge_fib_levels(
    const std::vector<double>& enabled_levels,
    const std::string& custom_levels_text)
{
    auto levels = enabled_levels;
    std::istringstream input(custom_levels_text);
    std::string raw;
    while (std::getline(input, raw, ',')) {
        auto token = trim(raw);
        if (token.empty()) {
            continue;
        }
        levels.push_back(parse_level(token));
    }
    return normalize_levels(std::move(levels));
}

FibSettings default_fib_settings()
{
    return FibSettings{
        FibLevelsConfig{default_retracement_presets(), ""},
        FibLevelsConfig{default_extension_presets(), ""},
    };
}

} // namespace tradereview::drawing
