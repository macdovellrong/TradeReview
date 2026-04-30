#pragma once

#include <optional>
#include <utility>

namespace tradereview::chart {

using PriceRange = std::pair<double, double>;

[[nodiscard]] std::optional<PriceRange> normalize_price_range(double min_price, double max_price);
[[nodiscard]] PriceRange zoom_price_range(PriceRange range, double anchor_price, double scale_factor);

} // namespace tradereview::chart
