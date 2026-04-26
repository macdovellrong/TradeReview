#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::chart {

class ChartSceneModel {
public:
    [[nodiscard]] std::uint64_t generation() const;
    [[nodiscard]] std::uint64_t revision() const;
    std::uint64_t bump_generation();

    [[nodiscard]] bool apply_window(data::CandleWindow window);
    bool set_loading(bool loading);
    bool set_visible_dense_range(DenseRange range);

    [[nodiscard]] std::size_t row_count() const;
    [[nodiscard]] bool loading() const;
    [[nodiscard]] const data::CandleWindow& window() const;
    [[nodiscard]] DenseRange visible_dense_range() const;
    [[nodiscard]] const ChartIndexMapper& index_mapper() const;
    [[nodiscard]] std::vector<std::string> enabled_price_indicators() const;
    [[nodiscard]] std::vector<std::string> requested_indicators() const;
    [[nodiscard]] bool bollinger_bands_enabled() const;
    [[nodiscard]] bool indicator_panels_enabled() const;
    bool set_indicator_enabled(const std::string& indicator_name, bool enabled);
    bool set_bollinger_bands_enabled(bool enabled);
    bool set_indicator_panels_enabled(bool enabled);

private:
    struct IndicatorState {
        std::vector<std::string> enabled_ema{"EMA20", "EMA30", "EMA40", "EMA50", "EMA60"};
        bool bollinger_bands_enabled = true;
        bool indicator_panels_enabled = true;
    };

    std::uint64_t generation_ = 0;
    std::uint64_t revision_ = 0;
    bool loading_ = false;
    data::CandleWindow window_;
    DenseRange visible_dense_range_;
    ChartIndexMapper index_mapper_;
    IndicatorState indicator_state_;
};

} // namespace tradereview::chart
