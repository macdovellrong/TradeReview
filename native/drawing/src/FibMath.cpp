#include "tradereview/drawing/FibMath.h"

namespace tradereview::drawing {

std::vector<FibLevel> build_retracement_levels(
    double start_price,
    double end_price,
    const std::vector<double>& levels)
{
    std::vector<FibLevel> rows;
    rows.reserve(levels.size());
    for (const auto level : levels) {
        rows.push_back(FibLevel{
            level,
            end_price + ((start_price - end_price) * level),
        });
    }
    return rows;
}

std::vector<FibLevel> build_extension_levels(
    double a_price,
    double b_price,
    double c_price,
    const std::vector<double>& levels)
{
    const auto delta = b_price - a_price;
    std::vector<FibLevel> rows;
    rows.reserve(levels.size());
    for (const auto level : levels) {
        rows.push_back(FibLevel{
            level,
            c_price + (delta * level),
        });
    }
    return rows;
}

} // namespace tradereview::drawing
