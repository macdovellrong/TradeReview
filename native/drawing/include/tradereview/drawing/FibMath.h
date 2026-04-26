#pragma once

#include <vector>

namespace tradereview::drawing {

struct FibLevel {
    double ratio = 0.0;
    double price = 0.0;
};

[[nodiscard]] std::vector<FibLevel> build_retracement_levels(
    double start_price,
    double end_price,
    const std::vector<double>& levels);

[[nodiscard]] std::vector<FibLevel> build_extension_levels(
    double a_price,
    double b_price,
    double c_price,
    const std::vector<double>& levels);

} // namespace tradereview::drawing
