#pragma once

#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"

#include <optional>
#include <string>

namespace tradereview::chart {

[[nodiscard]] double widget_x_for_dense_x(DenseRange visible_dense_range, int widget_width, double dense_x);

[[nodiscard]] double widget_y_for_normalized_device_y(float normalized_device_y, int widget_height);

[[nodiscard]] std::optional<double> widget_y_for_price(
    PaneRect pane,
    int widget_height,
    double min_price,
    double max_price,
    double price);

[[nodiscard]] std::string format_fib_ratio_label(double ratio);

} // namespace tradereview::chart
