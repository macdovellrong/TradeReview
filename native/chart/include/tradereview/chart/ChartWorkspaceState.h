#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tradereview::chart {

enum class ChartLayoutMode {
    Tabs,
    Vertical,
    DualVertical,
    Grid2x2,
};

struct ChartSlotState {
    std::uint64_t chart_id = 0;
    std::string requested_period = "1min";
};

class ChartWorkspaceState final {
public:
    ChartWorkspaceState();

    [[nodiscard]] std::size_t chart_count() const;
    [[nodiscard]] ChartLayoutMode layout_mode() const;
    [[nodiscard]] std::uint64_t active_chart_id() const;
    [[nodiscard]] std::vector<std::uint64_t> enabled_chart_ids() const;
    [[nodiscard]] const ChartSlotState* chart_slot(std::uint64_t chart_id) const;
    [[nodiscard]] ChartSlotState* chart_slot(std::uint64_t chart_id);
    [[nodiscard]] std::string chart_period(std::uint64_t chart_id) const;
    [[nodiscard]] bool chart_enabled(std::uint64_t chart_id) const;

    bool set_chart_count(int count);
    bool set_layout_mode(ChartLayoutMode mode);
    bool set_active_chart_id(std::uint64_t chart_id);
    bool set_chart_period(std::uint64_t chart_id, std::string period);

private:
    [[nodiscard]] static std::size_t clamped_chart_count(int count);

    std::vector<ChartSlotState> slots_;
    std::size_t chart_count_ = 4;
    ChartLayoutMode layout_mode_ = ChartLayoutMode::Tabs;
    std::uint64_t active_chart_id_ = 1;
};

} // namespace tradereview::chart
