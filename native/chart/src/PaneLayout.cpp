#include "tradereview/chart/PaneLayout.h"

namespace tradereview::chart {

float PaneRect::width() const
{
    return right - left;
}

float PaneRect::height() const
{
    return top - bottom;
}

PaneLayout build_pane_layout(bool indicator_panels_visible)
{
    if (!indicator_panels_visible) {
        return PaneLayout{PaneRect{-1.0F, 1.0F, -1.0F, 1.0F}, {}, {}, false, false};
    }

    PaneLayout layout;
    layout.price = {-1.0F, 1.0F, -0.16F, 1.0F};
    layout.macd = {-1.0F, 1.0F, -0.62F, -0.22F};
    layout.rsi = {-1.0F, 1.0F, -1.0F, -0.68F};
    layout.macd_visible = true;
    layout.rsi_visible = true;
    return layout;
}

} // namespace tradereview::chart
