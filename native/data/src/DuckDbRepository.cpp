#include "tradereview/data/DuckDbRepository.h"

#include "tradereview/data/DataError.h"
#include "tradereview/data/DuckDbSchema.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
#include <duckdb.h>
#endif

namespace tradereview::data {
namespace {

[[noreturn]] void throw_duckdb_unavailable()
{
    throw DataException(DataError{
        DataErrorCode::OpenFailed,
        "DuckDB repository is unavailable because TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF",
        {},
        {},
    });
}

[[nodiscard]] DataError make_data_error(
    DataErrorCode code,
    std::string message,
    std::string path = {},
    std::string table = {})
{
    return DataError{
        code,
        std::move(message),
        std::move(path),
        std::move(table),
    };
}

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
class QueryResult {
public:
    QueryResult() = default;
    ~QueryResult()
    {
        duckdb_destroy_result(&result_);
    }

    QueryResult(const QueryResult&) = delete;
    QueryResult& operator=(const QueryResult&) = delete;
    QueryResult(QueryResult&& other) noexcept
        : result_(std::exchange(other.result_, duckdb_result{}))
    {
    }

    QueryResult& operator=(QueryResult&& other) noexcept
    {
        if (this != &other) {
            duckdb_destroy_result(&result_);
            result_ = std::exchange(other.result_, duckdb_result{});
        }
        return *this;
    }

    duckdb_result* get()
    {
        return &result_;
    }

private:
    duckdb_result result_{};
};

[[nodiscard]] std::string value_string(duckdb_result* result, idx_t column, idx_t row)
{
    char* raw = duckdb_value_varchar(result, column, row);
    if (raw == nullptr) {
        return {};
    }
    std::string value{raw};
    duckdb_free(raw);
    return value;
}

[[nodiscard]] std::string sql_literal(std::string_view value)
{
    std::string escaped{"'"};
    for (const char ch : value) {
        escaped.push_back(ch);
        if (ch == '\'') {
            escaped.push_back('\'');
        }
    }
    escaped.push_back('\'');
    return escaped;
}

[[nodiscard]] std::string quoted_identifier(std::string_view value)
{
    std::string escaped{"\""};
    for (const char ch : value) {
        escaped.push_back(ch);
        if (ch == '"') {
            escaped.push_back('"');
        }
    }
    escaped.push_back('"');
    return escaped;
}

[[nodiscard]] std::int64_t ns_to_us(std::int64_t ns)
{
    return ns / 1000;
}

[[nodiscard]] std::int64_t us_to_ns(std::int64_t us)
{
    return us * 1000;
}

[[nodiscard]] bool contains_string(const std::vector<std::string>& values, std::string_view target)
{
    return std::any_of(
        values.begin(),
        values.end(),
        [target](const std::string& value) {
            return value == target;
        });
}

[[nodiscard]] bool ascii_equal_case_insensitive(std::string_view left, std::string_view right)
{
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto left_ch = static_cast<unsigned char>(left[index]);
        const auto right_ch = static_cast<unsigned char>(right[index]);
        if (std::tolower(left_ch) != std::tolower(right_ch)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool is_timestamp_alias(std::string_view name)
{
    return ascii_equal_case_insensitive(name, "timestamp") ||
        ascii_equal_case_insensitive(name, "datetime") ||
        ascii_equal_case_insensitive(name, "time");
}

[[nodiscard]] std::string timestamp_column(const TableSchema& schema)
{
    for (const auto& column : schema.columns) {
        if (is_timestamp_alias(column.name)) {
            return column.name;
        }
    }
    throw DataException(make_data_error(
        DataErrorCode::MissingColumn,
        "DuckDB table is missing a timestamp column: " + schema.name,
        {},
        schema.name));
}

void throw_if_schema_invalid(const SchemaValidationResult& result, std::string_view label)
{
    if (result.ok()) {
        return;
    }

    std::ostringstream message;
    message << "DuckDB " << label << " schema mismatch";
    if (!result.error.table.empty()) {
        message << " in " << result.error.table;
    }
    if (!result.missing_columns.empty()) {
        message << "; missing columns:";
        for (const auto& column : result.missing_columns) {
            message << ' ' << column;
        }
    }
    auto code = result.error.code;
    if (code == DataErrorCode::None) {
        code = DataErrorCode::SchemaMismatch;
    }
    throw DataException(make_data_error(code, message.str(), {}, result.error.table));
}

void throw_if_table_missing(const TableSchema& schema)
{
    if (!schema.columns.empty()) {
        return;
    }

    throw DataException(make_data_error(
        DataErrorCode::MissingTable,
        "DuckDB table is missing: " + schema.name,
        {},
        schema.name));
}
#endif

} // namespace

class DuckDbRepository::Impl {
public:
    Impl() = default;
    ~Impl()
    {
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        close();
#endif
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    QueryResult query(const std::string& sql)
    {
        QueryResult result;
        if (connection_ == nullptr) {
            throw DataException(make_data_error(
                DataErrorCode::InvalidRequest,
                "DuckDB database is not open",
                current_path_));
        }
        if (duckdb_query(connection_, sql.c_str(), result.get()) == DuckDBError) {
            const std::string error = duckdb_result_error(result.get());
            throw DataException(make_data_error(
                DataErrorCode::QueryFailed,
                "DuckDB query failed: " + error,
                current_path_));
        }
        return result;
    }

    void close()
    {
        if (connection_ != nullptr) {
            duckdb_disconnect(&connection_);
            connection_ = nullptr;
        }
        if (database_ != nullptr) {
            duckdb_close(&database_);
            database_ = nullptr;
        }
    }

    TableSchema table_schema(const std::string& table_name)
    {
        auto result = query(
            "SELECT column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_name = " +
            sql_literal(table_name) + " ORDER BY ordinal_position");

        TableSchema schema;
        schema.name = table_name;
        const auto rows = duckdb_row_count(result.get());
        schema.columns.reserve(static_cast<std::size_t>(rows));
        for (idx_t row = 0; row < rows; ++row) {
            schema.columns.push_back({value_string(result.get(), 0, row), value_string(result.get(), 1, row)});
        }
        return schema;
    }
#endif

    DataSetInfo open_readonly(const std::string& path)
    {
        std::lock_guard lock(mutex_);
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        close();
        current_path_ = path;

        if (path.empty() || !std::filesystem::exists(path)) {
            throw DataException(make_data_error(
                DataErrorCode::FileNotFound,
                "DuckDB database file was not found",
                path));
        }

        duckdb_config config = nullptr;
        const auto config_result = duckdb_create_config(&config);
        if (config_result == DuckDBError) {
            throw DataException(make_data_error(
                DataErrorCode::OpenFailed,
                "failed to create DuckDB config",
                path));
        }
        const auto readonly_result = duckdb_set_config(config, "access_mode", "READ_ONLY");
        if (readonly_result == DuckDBError) {
            duckdb_destroy_config(&config);
            throw DataException(make_data_error(
                DataErrorCode::OpenFailed,
                "failed to set DuckDB read-only access mode",
                path));
        }
        const auto open_result = duckdb_open_ext(path.c_str(), &database_, config, nullptr);
        duckdb_destroy_config(&config);
        if (open_result == DuckDBError) {
            throw DataException(make_data_error(
                DataErrorCode::OpenFailed,
                "failed to open DuckDB database read-only",
                path));
        }
        if (duckdb_connect(database_, &connection_) == DuckDBError) {
            duckdb_close(&database_);
            database_ = nullptr;
            throw DataException(make_data_error(
                DataErrorCode::OpenFailed,
                "failed to connect DuckDB database",
                path));
        }

        DataSetInfo info;
        info.dataset_path = path;
        info.metadata_only = true;
        load_tick_metadata(info);
        load_table_metadata(info);
        return info;
#else
        (void)path;
        throw_duckdb_unavailable();
#endif
    }

    CandleWindow query_candles(const CandleWindowRequest& request)
    {
        std::lock_guard lock(mutex_);
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        const auto table_name = duckdb_candle_table_for_period(request.requested_period);
        const auto schema = table_schema(table_name);
        throw_if_table_missing(schema);
        throw_if_schema_invalid(validate_candle_schema(schema, {}), "candle");
        const auto timestamp_name = timestamp_column(schema);
        const auto available_indicators = canonical_indicator_columns_present(schema);

        std::vector<std::string> requested_indicators;
        if (request.include_indicators) {
            for (const auto& indicator : request.requested_indicators) {
                if (contains_string(available_indicators, indicator)) {
                    requested_indicators.push_back(indicator);
                }
            }
        }

        std::ostringstream sql;
        sql << "SELECT epoch_us(" << quoted_identifier(timestamp_name) << "), open, high, low, close, volume";
        for (const auto& indicator : requested_indicators) {
            sql << ", " << quoted_identifier(indicator);
        }
        sql << " FROM " << quoted_identifier(table_name)
            << " WHERE epoch_us(" << quoted_identifier(timestamp_name) << ") BETWEEN " << ns_to_us(request.visible_range.start_ns)
            << " AND " << ns_to_us(request.visible_range.end_ns)
            << " ORDER BY " << quoted_identifier(timestamp_name);

        auto result = query(sql.str());
        const auto rows = duckdb_row_count(result.get());

        CandleWindow window;
        window.chart_id = request.chart_id;
        window.generation = request.generation;
        window.requested_period = request.requested_period;
        window.actual_period = duckdb_period_for_candle_table(table_name);
        window.visible_range = request.visible_range;
        window.from_cache = false;

        const auto row_count = static_cast<std::size_t>(rows);
        window.timestamp_ns.reserve(row_count);
        window.open.reserve(row_count);
        window.high.reserve(row_count);
        window.low.reserve(row_count);
        window.close.reserve(row_count);
        window.volume.reserve(row_count);
        for (const auto& indicator : requested_indicators) {
            window.indicators[indicator].reserve(row_count);
        }

        for (idx_t row = 0; row < rows; ++row) {
            window.timestamp_ns.push_back(us_to_ns(duckdb_value_int64(result.get(), 0, row)));
            window.open.push_back(duckdb_value_double(result.get(), 1, row));
            window.high.push_back(duckdb_value_double(result.get(), 2, row));
            window.low.push_back(duckdb_value_double(result.get(), 3, row));
            window.close.push_back(duckdb_value_double(result.get(), 4, row));
            window.volume.push_back(duckdb_value_double(result.get(), 5, row));
            for (idx_t indicator_index = 0; indicator_index < requested_indicators.size(); ++indicator_index) {
                window.indicators[requested_indicators[static_cast<std::size_t>(indicator_index)]].push_back(
                    duckdb_value_double(result.get(), 6 + indicator_index, row));
            }
        }

        if (!window.timestamp_ns.empty()) {
            window.loaded_range.start_ns = window.timestamp_ns.front();
            window.loaded_range.end_ns = window.timestamp_ns.back();
        }
        return window;
#else
        (void)request;
        throw_duckdb_unavailable();
#endif
    }

    TickSlice query_ticks(core::TimeRange range, size_t max_rows)
    {
        std::lock_guard lock(mutex_);
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        if (max_rows == 0) {
            return {};
        }

        const auto schema = table_schema("ticks");
        throw_if_table_missing(schema);
        throw_if_schema_invalid(validate_ticks_schema(schema), "ticks");
        const auto timestamp_name = timestamp_column(schema);

        std::ostringstream sql;
        sql << "SELECT epoch_us(" << quoted_identifier(timestamp_name) << "), price, volume"
            << " FROM ticks"
            << " WHERE epoch_us(" << quoted_identifier(timestamp_name) << ") BETWEEN " << ns_to_us(range.start_ns)
            << " AND " << ns_to_us(range.end_ns)
            << " ORDER BY " << quoted_identifier(timestamp_name)
            << " LIMIT " << max_rows;

        return tick_slice_from_query(sql.str());
#else
        (void)range;
        (void)max_rows;
        throw_duckdb_unavailable();
#endif
    }

    ReplayChunk query_replay_ticks(std::int64_t from_ns, std::int64_t to_ns, std::size_t max_ticks)
    {
        std::lock_guard lock(mutex_);
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        ReplayChunk chunk;
        if (max_ticks == 0 || to_ns <= from_ns) {
            chunk.reached_end = !has_tick_after(from_ns);
            return chunk;
        }

        const auto schema = table_schema("ticks");
        throw_if_table_missing(schema);
        throw_if_schema_invalid(validate_ticks_schema(schema), "ticks");
        const auto timestamp_name = timestamp_column(schema);

        std::ostringstream sql;
        sql << "SELECT epoch_us(" << quoted_identifier(timestamp_name) << "), price, volume"
            << " FROM ticks"
            << " WHERE epoch_us(" << quoted_identifier(timestamp_name) << ") > " << ns_to_us(from_ns)
            << " AND epoch_us(" << quoted_identifier(timestamp_name) << ") <= " << ns_to_us(to_ns)
            << " ORDER BY " << quoted_identifier(timestamp_name)
            << " LIMIT " << max_ticks;

        chunk.ticks = tick_slice_from_query(sql.str());
        const auto cursor_ns = chunk.ticks.timestamp_ns.empty() ? to_ns : chunk.ticks.timestamp_ns.back();
        chunk.reached_end = !has_tick_after(cursor_ns);
        return chunk;
#else
        (void)from_ns;
        (void)to_ns;
        (void)max_ticks;
        throw_duckdb_unavailable();
#endif
    }

private:
    std::mutex mutex_;

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
    TickSlice tick_slice_from_query(const std::string& sql)
    {
        auto result = query(sql);
        const auto rows = duckdb_row_count(result.get());

        TickSlice slice;
        const auto row_count = static_cast<std::size_t>(rows);
        slice.timestamp_ns.reserve(row_count);
        slice.price.reserve(row_count);
        slice.volume.reserve(row_count);
        for (idx_t row = 0; row < rows; ++row) {
            slice.timestamp_ns.push_back(us_to_ns(duckdb_value_int64(result.get(), 0, row)));
            slice.price.push_back(duckdb_value_double(result.get(), 1, row));
            slice.volume.push_back(duckdb_value_double(result.get(), 2, row));
        }
        return slice;
    }

    bool has_tick_after(std::int64_t timestamp_ns)
    {
        const auto schema = table_schema("ticks");
        throw_if_table_missing(schema);
        throw_if_schema_invalid(validate_ticks_schema(schema), "ticks");
        const auto timestamp_name = timestamp_column(schema);

        auto result = query(
            "SELECT 1 FROM ticks WHERE epoch_us(" + quoted_identifier(timestamp_name) + ") > " +
            std::to_string(ns_to_us(timestamp_ns)) + " LIMIT 1");
        return duckdb_row_count(result.get()) > 0;
    }

    void load_tick_metadata(DataSetInfo& info)
    {
        const auto schema = table_schema("ticks");
        throw_if_table_missing(schema);
        throw_if_schema_invalid(validate_ticks_schema(schema), "ticks");
        const auto timestamp_name = quoted_identifier(timestamp_column(schema));
        auto result = query(
            "SELECT count(*), min(epoch_us(" + timestamp_name + ")), max(epoch_us(" + timestamp_name + ")) FROM ticks");
        if (duckdb_row_count(result.get()) == 0) {
            return;
        }
        info.tick_count = duckdb_value_int64(result.get(), 0, 0);
        if (info.tick_count > 0 && !duckdb_value_is_null(result.get(), 1, 0) && !duckdb_value_is_null(result.get(), 2, 0)) {
            info.tick_range.start_ns = us_to_ns(duckdb_value_int64(result.get(), 1, 0));
            info.tick_range.end_ns = us_to_ns(duckdb_value_int64(result.get(), 2, 0));
        }
    }

    void load_table_metadata(DataSetInfo& info)
    {
        auto tables = query("SHOW TABLES");
        const auto rows = duckdb_row_count(tables.get());
        for (idx_t row = 0; row < rows; ++row) {
            const auto table_name = value_string(tables.get(), 0, row);
            if (!table_name.starts_with("candles_")) {
                continue;
            }
            const auto schema = table_schema(table_name);
            if (!validate_candle_schema(schema, {}).ok()) {
                continue;
            }
            info.available_periods.push_back(duckdb_period_for_candle_table(table_name));
            for (const auto& indicator : canonical_indicator_columns_present(schema)) {
                if (!contains_string(info.available_indicators, indicator)) {
                    info.available_indicators.push_back(indicator);
                }
            }
        }
    }

    duckdb_database database_ = nullptr;
    duckdb_connection connection_ = nullptr;
    std::string current_path_;
#endif
};

DuckDbRepository::DuckDbRepository()
    : impl_(std::make_unique<Impl>())
{
}

DuckDbRepository::~DuckDbRepository() = default;

DuckDbRepository::DuckDbRepository(DuckDbRepository&&) noexcept = default;

DuckDbRepository& DuckDbRepository::operator=(DuckDbRepository&&) noexcept = default;

DataSetInfo DuckDbRepository::open_readonly(const std::string& path)
{
    return impl_->open_readonly(path);
}

CandleWindow DuckDbRepository::query_candles(const CandleWindowRequest& request)
{
    return impl_->query_candles(request);
}

TickSlice DuckDbRepository::query_ticks(core::TimeRange range, size_t max_rows)
{
    return impl_->query_ticks(range, max_rows);
}

ReplayChunk DuckDbRepository::query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks)
{
    return impl_->query_replay_ticks(from_ns, to_ns, max_ticks);
}

} // namespace tradereview::data
