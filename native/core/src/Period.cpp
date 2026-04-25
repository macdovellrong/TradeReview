#include "tradereview/core/Period.h"

#include <cctype>
#include <limits>
#include <stdexcept>

namespace tradereview::core {
namespace {

struct ParsedPeriod {
    std::int64_t value = 0;
    std::string_view unit;
};

std::optional<ParsedPeriod> parse_period(std::string_view period)
{
    if (period.empty()) {
        return std::nullopt;
    }

    std::int64_t value = 0;
    std::size_t index = 0;
    while (index < period.size() && std::isdigit(static_cast<unsigned char>(period[index])) != 0) {
        const auto digit = static_cast<std::int64_t>(period[index] - '0');
        if (value > (std::numeric_limits<std::int64_t>::max() - digit) / 10) {
            return std::nullopt;
        }
        value = value * 10 + digit;
        ++index;
    }

    if (index == 0 || index == period.size()) {
        return std::nullopt;
    }

    return ParsedPeriod{value, period.substr(index)};
}

std::optional<std::int64_t> unit_seconds(std::string_view unit)
{
    if (unit == "s") {
        return 1;
    }
    if (unit == "min") {
        return 60;
    }
    if (unit == "h") {
        return 60 * 60;
    }
    if (unit == "d") {
        return 24 * 60 * 60;
    }
    if (unit == "w") {
        return 7 * 24 * 60 * 60;
    }
    if (unit == "M") {
        return 30 * 24 * 60 * 60;
    }
    return std::nullopt;
}

std::optional<std::string_view> table_unit(std::string_view unit)
{
    if (unit == "s") {
        return std::string_view{"s"};
    }
    if (unit == "min") {
        return std::string_view{"m"};
    }
    if (unit == "h") {
        return std::string_view{"h"};
    }
    if (unit == "d") {
        return std::string_view{"d"};
    }
    if (unit == "w") {
        return std::string_view{"w"};
    }
    if (unit == "M") {
        return std::string_view{"mo"};
    }
    return std::nullopt;
}

} // namespace

std::optional<std::int64_t> try_period_seconds(std::string_view period)
{
    const auto parsed = parse_period(period);
    if (!parsed.has_value()) {
        return std::nullopt;
    }

    const auto seconds = unit_seconds(parsed->unit);
    if (!seconds.has_value()) {
        return std::nullopt;
    }
    if (parsed->value > std::numeric_limits<std::int64_t>::max() / *seconds) {
        return std::nullopt;
    }

    return parsed->value * *seconds;
}

std::int64_t period_seconds(std::string_view period)
{
    const auto seconds = try_period_seconds(period);
    if (!seconds.has_value()) {
        throw std::invalid_argument("invalid period");
    }
    return *seconds;
}

std::string duckdb_candle_table(std::string_view period)
{
    const auto parsed = parse_period(period);
    if (!parsed.has_value()) {
        throw std::invalid_argument("invalid period");
    }

    const auto unit = table_unit(parsed->unit);
    if (!unit.has_value()) {
        throw std::invalid_argument("invalid period");
    }

    return "candles_" + std::to_string(parsed->value) + std::string{*unit};
}

} // namespace tradereview::core
