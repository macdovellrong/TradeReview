#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "tradereview/data/DataError.h"

namespace tradereview::data {

struct ColumnInfo {
    std::string name;
    std::string type;
};

struct TableSchema {
    std::string name;
    std::vector<ColumnInfo> columns;
};

struct SchemaValidationResult {
    DataError error;
    std::vector<std::string> missing_columns;

    [[nodiscard]] bool ok() const;
};

[[nodiscard]] std::string duckdb_candle_table_for_period(std::string_view period);
[[nodiscard]] std::string duckdb_period_for_candle_table(std::string_view table_name);
[[nodiscard]] bool has_column(const TableSchema& schema, std::string_view name);
[[nodiscard]] std::vector<std::string> canonical_indicator_columns_present(const TableSchema& schema);
[[nodiscard]] SchemaValidationResult validate_ticks_schema(const TableSchema& schema);
[[nodiscard]] SchemaValidationResult validate_candle_schema(
    const TableSchema& schema,
    const std::vector<std::string>& requested_indicators,
    bool allow_gap_rows = false);

} // namespace tradereview::data
