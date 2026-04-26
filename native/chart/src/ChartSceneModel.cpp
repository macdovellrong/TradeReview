#include "tradereview/chart/ChartSceneModel.h"

#include <utility>

namespace tradereview::chart {

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

} // namespace tradereview::chart
