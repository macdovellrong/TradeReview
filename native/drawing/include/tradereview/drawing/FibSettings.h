#pragma once

#include <string>
#include <vector>

namespace tradereview::drawing {

struct FibLevelsConfig {
    std::vector<double> enabled_levels;
    std::string custom_levels_text;

    [[nodiscard]] std::vector<double> effective_levels() const;
};

struct FibSettings {
    FibLevelsConfig retracement;
    FibLevelsConfig extension;
};

[[nodiscard]] std::vector<double> default_retracement_presets();
[[nodiscard]] std::vector<double> default_extension_presets();
[[nodiscard]] std::vector<double> merge_fib_levels(
    const std::vector<double>& enabled_levels,
    const std::string& custom_levels_text);
[[nodiscard]] FibSettings default_fib_settings();

} // namespace tradereview::drawing
