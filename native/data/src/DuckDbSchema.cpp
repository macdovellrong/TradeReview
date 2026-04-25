#include "tradereview/data/DuckDbSchema.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

#include "tradereview/core/Period.h"

namespace tradereview::data {
namespace {

[[nodiscard]] std::string ascii_lower(std::string_view value)
{
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return lowered;
}

[[nodiscard]] bool ascii_equal_case_insensitive(std::string_view left, std::string_view right)
{
    return ascii_lower(left) == ascii_lower(right);
}

[[nodiscard]] SchemaValidationResult missing_columns_result(
    const TableSchema& schema,
    std::vector<std::string> missing_columns)
{
    SchemaValidationResult result;
    result.error.code = DataErrorCode::MissingColumn;
    result.error.table = schema.name;
    result.error.message = "missing required DuckDB column";
    result.missing_columns = std::move(missing_columns);
    return result;
}

[[nodiscard]] bool has_timestamp_alias(const TableSchema& schema)
{
    return has_column(schema, "timestamp") || has_column(schema, "datetime") || has_column(schema, "time");
}

[[nodiscard]] SchemaValidationResult validate_time_and_required_columns(
    const TableSchema& schema,
    const std::vector<std::string>& required_columns)
{
    std::vector<std::string> missing_columns;
    if (!has_timestamp_alias(schema)) {
        missing_columns.push_back("timestamp");
    }
    for (const auto& column : required_columns) {
        if (!has_column(schema, column)) {
            missing_columns.push_back(column);
        }
    }
    if (!missing_columns.empty()) {
        return missing_columns_result(schema, std::move(missing_columns));
    }
    return {};
}

} // namespace

bool SchemaValidationResult::ok() const
{
    return error.code == DataErrorCode::None && missing_columns.empty();
}

std::string duckdb_candle_table_for_period(std::string_view period)
{
    if (period == "1m") {
        return core::duckdb_candle_table("1min");
    }
    return core::duckdb_candle_table(period);
}

bool has_column(const TableSchema& schema, std::string_view name)
{
    return std::any_of(
        schema.columns.begin(),
        schema.columns.end(),
        [name](const ColumnInfo& column) {
            return ascii_equal_case_insensitive(column.name, name);
        });
}

SchemaValidationResult validate_ticks_schema(const TableSchema& schema)
{
    return validate_time_and_required_columns(schema, {"price", "volume"});
}

SchemaValidationResult validate_candle_schema(
    const TableSchema& schema,
    const std::vector<std::string>& requested_indicators,
    bool allow_gap_rows)
{
    (void)allow_gap_rows;

    std::vector<std::string> required_columns{
        "open",
        "high",
        "low",
        "close",
        "volume",
    };
    required_columns.insert(required_columns.end(), requested_indicators.begin(), requested_indicators.end());
    return validate_time_and_required_columns(schema, required_columns);
}

} // namespace tradereview::data
