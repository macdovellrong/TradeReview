#pragma once

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/core/TimeRange.h"

#include <cstddef>

namespace tradereview::chart {

struct DenseRange {
    double start_x = 0.0;
    double end_x = 0.0;

    [[nodiscard]] double span() const;
};

struct ReloadDecision {
    bool requested = false;
    core::TimeRange visible_range;
};

class ChartInteractionController final {
public:
    explicit ChartInteractionController(int right_padding_bars = 20);

    void reset_for_row_count(std::size_t row_count);
    void reset_for_visible_time_range(const ChartIndexMapper& mapper, core::TimeRange visible_range);
    void set_visible_dense_range(DenseRange range);
    bool center_on_dense_x(double dense_x);
    void pan_by_pixels(double pixel_delta_x, int viewport_width);
    void zoom_at_pixel(double pixel_x, int viewport_width, double scale_factor);

    [[nodiscard]] DenseRange visible_dense_range() const;
    [[nodiscard]] int right_padding_bars() const;
    [[nodiscard]] double zoom_scale_from_wheel_delta(int angle_delta_y, int pixel_delta_y) const;
    [[nodiscard]] core::TimeRange visible_time_range(const ChartIndexMapper& mapper) const;
    [[nodiscard]] ReloadDecision reload_decision(const ChartIndexMapper& mapper, core::TimeRange loaded_range) const;

private:
    [[nodiscard]] DenseRange normalized_range(DenseRange range) const;
    [[nodiscard]] core::TimeRange padded_loaded_range(const ChartIndexMapper& mapper, core::TimeRange loaded_range) const;

    int right_padding_bars_;
    DenseRange visible_dense_range_;
};

} // namespace tradereview::chart
