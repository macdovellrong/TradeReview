#pragma once

#include "tradereview/chart/ChartWorkspaceState.h"

#include <QSettings>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::app {

struct SessionState {
    std::string dataset_path;
    std::int64_t center_time_ns = 0;
    int chart_count = 4;
    chart::ChartLayoutMode layout_mode = chart::ChartLayoutMode::Tabs;
    std::vector<std::string> periods;
};

[[nodiscard]] std::string layout_mode_to_string(chart::ChartLayoutMode mode);
[[nodiscard]] std::optional<chart::ChartLayoutMode> layout_mode_from_string(const std::string& text);

void save_session_state(QSettings& settings, const SessionState& state);
[[nodiscard]] std::optional<SessionState> load_session_state(QSettings& settings);

} // namespace tradereview::app
