#include "tradereview/chart/ChartPeriod.h"

#include <algorithm>
#include <cctype>

namespace tradereview::chart {
namespace {

[[nodiscard]] bool all_digits(std::string_view value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) {
        return std::isdigit(ch) != 0;
    });
}

} // namespace

std::string canonical_chart_period(std::string_view period)
{
    if (period.size() > 1 && period.back() == 'm') {
        const auto number = period.substr(0, period.size() - 1);
        if (all_digits(number)) {
            return std::string{number} + "min";
        }
    }
    return std::string{period};
}

std::string toolbar_chart_period(std::string_view period)
{
    constexpr std::string_view suffix = "min";
    if (period.size() > suffix.size() && period.ends_with(suffix)) {
        const auto number = period.substr(0, period.size() - suffix.size());
        if (all_digits(number)) {
            return std::string{number} + "m";
        }
    }
    return std::string{period};
}

} // namespace tradereview::chart
