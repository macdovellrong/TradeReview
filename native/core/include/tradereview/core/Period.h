#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace tradereview::core {

std::optional<std::int64_t> try_period_seconds(std::string_view period);
std::int64_t period_seconds(std::string_view period);
std::string duckdb_candle_table(std::string_view period);

} // namespace tradereview::core
