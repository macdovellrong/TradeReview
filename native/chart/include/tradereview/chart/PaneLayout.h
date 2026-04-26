#pragma once

namespace tradereview::chart {

struct PaneRect {
    float left = -1.0F;
    float right = 1.0F;
    float bottom = -1.0F;
    float top = 1.0F;

    [[nodiscard]] float width() const;
    [[nodiscard]] float height() const;
};

struct PaneLayout {
    PaneRect price;
    PaneRect macd;
    PaneRect rsi;
    bool macd_visible = false;
    bool rsi_visible = false;
};

[[nodiscard]] PaneLayout build_pane_layout(bool indicator_panels_visible);

} // namespace tradereview::chart
