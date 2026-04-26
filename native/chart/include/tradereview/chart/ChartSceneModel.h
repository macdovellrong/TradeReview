#pragma once

#include <cstddef>
#include <cstdint>

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::chart {

class ChartSceneModel {
public:
    [[nodiscard]] std::uint64_t generation() const;
    [[nodiscard]] std::uint64_t revision() const;
    std::uint64_t bump_generation();

    [[nodiscard]] bool apply_window(data::CandleWindow window);

    [[nodiscard]] std::size_t row_count() const;
    [[nodiscard]] const data::CandleWindow& window() const;
    [[nodiscard]] const ChartIndexMapper& index_mapper() const;

private:
    std::uint64_t generation_ = 0;
    std::uint64_t revision_ = 0;
    data::CandleWindow window_;
    ChartIndexMapper index_mapper_;
};

} // namespace tradereview::chart
