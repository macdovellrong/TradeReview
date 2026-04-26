#include "tradereview/chart/ChartSceneModel.h"

#include "tradereview/data/IndicatorColumns.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace tradereview::chart {
namespace {

[[nodiscard]] bool known_ema(std::string_view name)
{
    using data::IndicatorColumns;
    constexpr std::array<std::string_view, 7> kEmaColumns{
        IndicatorColumns::EMA20,
        IndicatorColumns::EMA30,
        IndicatorColumns::EMA40,
        IndicatorColumns::EMA50,
        IndicatorColumns::EMA60,
        IndicatorColumns::EMA100,
        IndicatorColumns::EMA240,
    };
    return std::find(kEmaColumns.begin(), kEmaColumns.end(), name) != kEmaColumns.end();
}

void append_unique(std::vector<std::string>& values, std::string value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(std::move(value));
    }
}

} // namespace

std::uint64_t ChartSceneModel::generation() const
{
    return generation_;
}

std::uint64_t ChartSceneModel::revision() const
{
    return revision_;
}

std::uint64_t ChartSceneModel::bump_generation()
{
    return ++generation_;
}

bool ChartSceneModel::apply_window(data::CandleWindow window)
{
    if (window.generation != generation_) {
        return false;
    }
    if (!window.has_consistent_columns()) {
        return false;
    }

    index_mapper_.set_timestamps(window.timestamp_ns);
    window_ = std::move(window);
    loading_ = false;
    ++revision_;
    return true;
}

bool ChartSceneModel::set_loading(bool loading)
{
    if (loading_ == loading) {
        return false;
    }
    loading_ = loading;
    ++revision_;
    return true;
}

bool ChartSceneModel::set_visible_dense_range(DenseRange range)
{
    if (range.end_x < range.start_x) {
        std::swap(range.start_x, range.end_x);
    }
    if (visible_dense_range_.start_x == range.start_x && visible_dense_range_.end_x == range.end_x) {
        return false;
    }

    visible_dense_range_ = range;
    ++revision_;
    return true;
}

std::size_t ChartSceneModel::row_count() const
{
    return window_.row_count();
}

bool ChartSceneModel::loading() const
{
    return loading_;
}

const data::CandleWindow& ChartSceneModel::window() const
{
    return window_;
}

DenseRange ChartSceneModel::visible_dense_range() const
{
    return visible_dense_range_;
}

const ChartIndexMapper& ChartSceneModel::index_mapper() const
{
    return index_mapper_;
}

std::vector<std::string> ChartSceneModel::enabled_price_indicators() const
{
    std::vector<std::string> indicators = indicator_state_.enabled_ema;
    if (indicator_state_.bollinger_bands_enabled) {
        indicators.push_back(std::string{data::IndicatorColumns::BB_Upper});
        indicators.push_back(std::string{data::IndicatorColumns::BB_Lower});
    }
    return indicators;
}

std::vector<std::string> ChartSceneModel::requested_indicators() const
{
    auto indicators = enabled_price_indicators();
    if (indicator_state_.indicator_panels_enabled) {
        append_unique(indicators, std::string{data::IndicatorColumns::MACD});
        append_unique(indicators, std::string{data::IndicatorColumns::MACD_Signal});
        append_unique(indicators, std::string{data::IndicatorColumns::MACD_Hist});
        append_unique(indicators, std::string{data::IndicatorColumns::RSI});
    }
    return indicators;
}

bool ChartSceneModel::bollinger_bands_enabled() const
{
    return indicator_state_.bollinger_bands_enabled;
}

bool ChartSceneModel::indicator_panels_enabled() const
{
    return indicator_state_.indicator_panels_enabled;
}

bool ChartSceneModel::set_indicator_enabled(const std::string& indicator_name, bool enabled)
{
    if (!known_ema(indicator_name)) {
        return false;
    }

    const auto existing = std::find(indicator_state_.enabled_ema.begin(), indicator_state_.enabled_ema.end(), indicator_name);
    if (enabled && existing == indicator_state_.enabled_ema.end()) {
        indicator_state_.enabled_ema.push_back(indicator_name);
        ++revision_;
        return true;
    }
    if (!enabled && existing != indicator_state_.enabled_ema.end()) {
        indicator_state_.enabled_ema.erase(existing);
        ++revision_;
        return true;
    }
    return false;
}

bool ChartSceneModel::set_bollinger_bands_enabled(bool enabled)
{
    if (indicator_state_.bollinger_bands_enabled == enabled) {
        return false;
    }
    indicator_state_.bollinger_bands_enabled = enabled;
    ++revision_;
    return true;
}

bool ChartSceneModel::set_indicator_panels_enabled(bool enabled)
{
    if (indicator_state_.indicator_panels_enabled == enabled) {
        return false;
    }
    indicator_state_.indicator_panels_enabled = enabled;
    ++revision_;
    return true;
}

} // namespace tradereview::chart
