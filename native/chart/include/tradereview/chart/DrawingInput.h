#pragma once

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/drawing/DrawingSpec.h"

#include <optional>

namespace tradereview::chart {

struct WidgetPosition {
    double x = 0.0;
    double y = 0.0;
};

[[nodiscard]] std::optional<drawing::DrawingPoint> drawing_point_from_widget_position(
    const ChartIndexMapper& mapper,
    DenseRange visible_dense_range,
    PaneRect price_pane,
    int widget_width,
    int widget_height,
    WidgetPosition position,
    double min_price,
    double max_price);

} // namespace tradereview::chart
