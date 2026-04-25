#include "tradereview/chart/ChartSceneModel.h"

#include <utility>

namespace tradereview::chart {

std::uint64_t ChartSceneModel::generation() const
{
    return generation_;
}

void ChartSceneModel::bump_generation()
{
    ++generation_;
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
    return true;
}

std::size_t ChartSceneModel::row_count() const
{
    return window_.row_count();
}

const data::CandleWindow& ChartSceneModel::window() const
{
    return window_;
}

const ChartIndexMapper& ChartSceneModel::index_mapper() const
{
    return index_mapper_;
}

} // namespace tradereview::chart
