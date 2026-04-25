#include "tradereview/data/CandleWindow.h"

namespace tradereview::data {

size_t CandleWindow::row_count() const
{
    return timestamp_ns.size();
}

bool CandleWindow::empty() const
{
    return row_count() == 0;
}

bool CandleWindow::has_consistent_columns() const
{
    return has_consistent_ohlcv() && has_consistent_indicators();
}

bool CandleWindow::has_consistent_ohlcv() const
{
    const size_t rows = row_count();
    if (open.size() != rows || high.size() != rows || low.size() != rows || close.size() != rows || volume.size() != rows) {
        return false;
    }
    return true;
}

bool CandleWindow::has_consistent_indicators() const
{
    const size_t rows = row_count();
    for (const auto& [name, values] : indicators) {
        (void)name;
        if (values.size() != rows) {
            return false;
        }
    }
    return true;
}

bool CandleWindow::has_loaded_range() const
{
    return loaded_range.end_ns > loaded_range.start_ns;
}

bool CandleWindow::has_visible_range() const
{
    return visible_range.end_ns > visible_range.start_ns;
}

} // namespace tradereview::data
