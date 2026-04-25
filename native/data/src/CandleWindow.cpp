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
    const size_t rows = row_count();
    if (open.size() != rows || high.size() != rows || low.size() != rows || close.size() != rows || volume.size() != rows) {
        return false;
    }
    for (const auto& [name, values] : indicators) {
        (void)name;
        if (values.size() != rows) {
            return false;
        }
    }
    return true;
}

} // namespace tradereview::data
