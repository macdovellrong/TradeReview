#include "tradereview/app/SessionState.h"

#include <QString>
#include <QStringList>
#include <QVariant>

#include <algorithm>

namespace tradereview::app {
namespace {

constexpr int kDefaultChartCount = 4;

[[nodiscard]] QString to_qstring(const std::string& value)
{
    return QString::fromUtf8(value.data(), static_cast<qsizetype>(value.size()));
}

[[nodiscard]] std::string to_string(const QString& value)
{
    const auto utf8 = value.toUtf8();
    return {utf8.constData(), static_cast<std::size_t>(utf8.size())};
}

[[nodiscard]] QStringList to_qstring_list(const std::vector<std::string>& values)
{
    QStringList result;
    for (const auto& value : values) {
        result.push_back(to_qstring(value));
    }
    return result;
}

[[nodiscard]] std::vector<std::string> to_string_vector(const QStringList& values)
{
    std::vector<std::string> result;
    result.reserve(static_cast<std::size_t>(values.size()));
    for (const auto& value : values) {
        result.push_back(to_string(value));
    }
    return result;
}

} // namespace

std::string layout_mode_to_string(chart::ChartLayoutMode mode)
{
    switch (mode) {
    case chart::ChartLayoutMode::Vertical:
        return "Vertical";
    case chart::ChartLayoutMode::DualVertical:
        return "Dual Vertical";
    case chart::ChartLayoutMode::Grid2x2:
        return "Grid 2x2";
    case chart::ChartLayoutMode::Tabs:
        return "Tabs";
    }
    return "Tabs";
}

std::optional<chart::ChartLayoutMode> layout_mode_from_string(const std::string& text)
{
    if (text == "Vertical") {
        return chart::ChartLayoutMode::Vertical;
    }
    if (text == "Dual Vertical") {
        return chart::ChartLayoutMode::DualVertical;
    }
    if (text == "Grid 2x2") {
        return chart::ChartLayoutMode::Grid2x2;
    }
    if (text == "Tabs") {
        return chart::ChartLayoutMode::Tabs;
    }
    return std::nullopt;
}

void save_session_state(QSettings& settings, const SessionState& state)
{
    settings.setValue("session/db_path", to_qstring(state.dataset_path));
    settings.setValue("session/center_time_ns", QVariant::fromValue<qlonglong>(state.center_time_ns));
    settings.setValue("session/chart_count", std::clamp(state.chart_count, 1, 4));
    settings.setValue("session/layout_mode", to_qstring(layout_mode_to_string(state.layout_mode)));
    settings.setValue("session/periods", to_qstring_list(state.periods));
    settings.sync();
}

std::optional<SessionState> load_session_state(QSettings& settings)
{
    const auto dataset_path = settings.value("session/db_path").toString();
    if (dataset_path.isEmpty()) {
        return std::nullopt;
    }

    bool center_ok = false;
    const auto center_time_ns = settings.value("session/center_time_ns").toLongLong(&center_ok);
    if (!center_ok) {
        return std::nullopt;
    }

    bool chart_count_ok = false;
    auto chart_count = settings.value("session/chart_count", kDefaultChartCount).toInt(&chart_count_ok);
    if (!chart_count_ok) {
        chart_count = kDefaultChartCount;
    }

    auto layout_mode = chart::ChartLayoutMode::Tabs;
    const auto layout_text = to_string(settings.value("session/layout_mode", "Tabs").toString());
    if (const auto parsed = layout_mode_from_string(layout_text); parsed.has_value()) {
        layout_mode = *parsed;
    }

    auto periods = to_string_vector(settings.value("session/periods").toStringList());
    if (periods.empty()) {
        periods = {"1min", "1min", "1min", "1min"};
    }
    while (periods.size() < 4) {
        periods.push_back("1min");
    }
    if (periods.size() > 4) {
        periods.resize(4);
    }

    return SessionState{
        to_string(dataset_path),
        center_time_ns,
        std::clamp(chart_count, 1, 4),
        layout_mode,
        std::move(periods),
    };
}

} // namespace tradereview::app
