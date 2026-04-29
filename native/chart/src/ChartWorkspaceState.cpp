#include "tradereview/chart/ChartWorkspaceState.h"

#include "tradereview/chart/ChartPeriod.h"

#include <algorithm>
#include <utility>

namespace tradereview::chart {
namespace {

constexpr std::size_t kMinCharts = 1;
constexpr std::size_t kMaxCharts = 4;

} // namespace

ChartWorkspaceState::ChartWorkspaceState()
{
    slots_.reserve(kMaxCharts);
    for (std::uint64_t chart_id = 1; chart_id <= kMaxCharts; ++chart_id) {
        slots_.push_back(ChartSlotState{chart_id, "1min"});
    }
}

std::size_t ChartWorkspaceState::chart_count() const
{
    return chart_count_;
}

ChartLayoutMode ChartWorkspaceState::layout_mode() const
{
    return layout_mode_;
}

std::uint64_t ChartWorkspaceState::active_chart_id() const
{
    return active_chart_id_;
}

std::vector<std::uint64_t> ChartWorkspaceState::enabled_chart_ids() const
{
    std::vector<std::uint64_t> ids;
    ids.reserve(chart_count_);
    for (std::size_t index = 0; index < chart_count_; ++index) {
        ids.push_back(slots_[index].chart_id);
    }
    return ids;
}

const ChartSlotState* ChartWorkspaceState::chart_slot(std::uint64_t chart_id) const
{
    const auto found = std::find_if(slots_.begin(), slots_.end(), [chart_id](const ChartSlotState& slot) {
        return slot.chart_id == chart_id;
    });
    if (found == slots_.end()) {
        return nullptr;
    }
    return &(*found);
}

ChartSlotState* ChartWorkspaceState::chart_slot(std::uint64_t chart_id)
{
    const auto found = std::find_if(slots_.begin(), slots_.end(), [chart_id](const ChartSlotState& slot) {
        return slot.chart_id == chart_id;
    });
    if (found == slots_.end()) {
        return nullptr;
    }
    return &(*found);
}

std::string ChartWorkspaceState::chart_period(std::uint64_t chart_id) const
{
    const auto* slot = chart_slot(chart_id);
    if (slot == nullptr) {
        return {};
    }
    return slot->requested_period;
}

bool ChartWorkspaceState::chart_enabled(std::uint64_t chart_id) const
{
    return chart_id >= 1 && chart_id <= chart_count_;
}

bool ChartWorkspaceState::set_chart_count(int count)
{
    const auto clamped = clamped_chart_count(count);
    if (chart_count_ == clamped) {
        return false;
    }

    chart_count_ = clamped;
    if (!chart_enabled(active_chart_id_)) {
        active_chart_id_ = static_cast<std::uint64_t>(chart_count_);
    }
    return true;
}

bool ChartWorkspaceState::set_layout_mode(ChartLayoutMode mode)
{
    if (layout_mode_ == mode) {
        return false;
    }
    layout_mode_ = mode;
    return true;
}

bool ChartWorkspaceState::set_active_chart_id(std::uint64_t chart_id)
{
    if (!chart_enabled(chart_id) || active_chart_id_ == chart_id) {
        return false;
    }
    active_chart_id_ = chart_id;
    return true;
}

bool ChartWorkspaceState::set_chart_period(std::uint64_t chart_id, std::string period)
{
    auto* slot = chart_slot(chart_id);
    auto canonical_period = canonical_chart_period(period);
    if (slot == nullptr || canonical_period.empty() || slot->requested_period == canonical_period) {
        return false;
    }
    slot->requested_period = std::move(canonical_period);
    return true;
}

std::size_t ChartWorkspaceState::clamped_chart_count(int count)
{
    return std::clamp(static_cast<std::size_t>(std::max(count, 0)), kMinCharts, kMaxCharts);
}

} // namespace tradereview::chart
