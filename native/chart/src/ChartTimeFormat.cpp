#include "tradereview/chart/ChartTimeFormat.h"

#include "tradereview/core/Period.h"

#include <QDateTime>
#include <QTimeZone>

namespace tradereview::chart {
namespace {

constexpr std::int64_t kOneMinuteSeconds = 60;
constexpr std::int64_t kOneDaySeconds = 24 * 60 * 60;

[[nodiscard]] QString timestamp_format_for_period(std::string_view period)
{
    const auto period_seconds = core::try_period_seconds(period);
    if (!period_seconds.has_value() || *period_seconds <= kOneMinuteSeconds) {
        return QStringLiteral("yyyy-MM-dd HH:mm:ss");
    }
    if (*period_seconds < kOneDaySeconds) {
        return QStringLiteral("yyyy-MM-dd HH:mm");
    }
    return QStringLiteral("yyyy-MM-dd");
}

} // namespace

QString format_axis_timestamp_label(std::int64_t timestamp_ns, std::string_view period)
{
    const auto timestamp_ms = timestamp_ns / 1'000'000LL;
    const auto date_time = QDateTime::fromMSecsSinceEpoch(timestamp_ms, QTimeZone::UTC);
    return date_time.toString(timestamp_format_for_period(period));
}

} // namespace tradereview::chart
