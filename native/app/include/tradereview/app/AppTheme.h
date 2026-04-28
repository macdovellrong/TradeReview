#pragma once

class QApplication;
class QColor;
class QString;

namespace tradereview::app {

namespace theme {

struct Size {
    static constexpr int MainToolbarHeight = 38;
    static constexpr int ChartToolbarHeight = 36;
    static constexpr int ControlHeight = 28;
    static constexpr int ToolRailWidth = 42;
    static constexpr int SidePanelWidth = 260;
    static constexpr int StatusStripHeight = 30;
};

void apply(QApplication& app);
[[nodiscard]] QColor chartBackground();
[[nodiscard]] QColor chartEmptyText();
[[nodiscard]] QColor loadingOverlay();
[[nodiscard]] QString styleSheet();

} // namespace theme

} // namespace tradereview::app
