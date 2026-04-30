#include "tradereview/chart/PriceRange.h"

#include <algorithm>
#include <cmath>

namespace tradereview::chart {

std::optional<PriceRange> normalize_price_range(double min_price, double max_price)
{
    if (!std::isfinite(min_price) || !std::isfinite(max_price)) {
        return std::nullopt;
    }
    if (max_price < min_price) {
        std::swap(min_price, max_price);
    }
    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }
    return PriceRange{min_price, max_price};
}

PriceRange zoom_price_range(PriceRange range, double anchor_price, double scale_factor)
{
    const auto normalized = normalize_price_range(range.first, range.second).value_or(PriceRange{0.0, 1.0});
    if (!std::isfinite(anchor_price)) {
        anchor_price = normalized.first + ((normalized.second - normalized.first) * 0.5);
    }

    anchor_price = std::clamp(anchor_price, normalized.first, normalized.second);
    if (!std::isfinite(scale_factor) || scale_factor <= 0.0) {
        scale_factor = 1.0;
    }
    scale_factor = std::clamp(scale_factor, 0.05, 20.0);

    const auto next_min = anchor_price - ((anchor_price - normalized.first) * scale_factor);
    const auto next_max = anchor_price + ((normalized.second - anchor_price) * scale_factor);
    return normalize_price_range(next_min, next_max).value_or(normalized);
}

PriceRange pan_price_range(PriceRange range, double price_delta)
{
    const auto normalized = normalize_price_range(range.first, range.second).value_or(PriceRange{0.0, 1.0});
    if (!std::isfinite(price_delta)) {
        return normalized;
    }
    return normalize_price_range(normalized.first + price_delta, normalized.second + price_delta).value_or(normalized);
}

} // namespace tradereview::chart
