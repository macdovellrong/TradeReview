#include "tradereview/core/Assertions.h"
#include "tradereview/data/DuckDbRepository.h"
#include "tradereview/data/DuckDbSchema.h"
#include "tradereview/data/IndicatorColumns.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::TableSchema ticks_schema()
{
    return tradereview::data::TableSchema{
        "ticks",
        {
            {"timestamp", "TIMESTAMP"},
            {"price", "DOUBLE"},
            {"volume", "DOUBLE"},
        }};
}

tradereview::data::TableSchema candle_schema()
{
    return tradereview::data::TableSchema{
        "candles_1m",
        {
            {"timestamp", "TIMESTAMP"},
            {"open", "DOUBLE"},
            {"high", "DOUBLE"},
            {"low", "DOUBLE"},
            {"close", "DOUBLE"},
            {"volume", "DOUBLE"},
        }};
}

void test_duckdb_schema_maps_period_aliases_to_existing_tables()
{
    tradereview::core::assert_equal(
        tradereview::data::duckdb_candle_table_for_period("1min"),
        std::string{"candles_1m"},
        "1min candle table");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_candle_table_for_period("1m"),
        std::string{"candles_1m"},
        "1m candle table alias");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_candle_table_for_period("1M"),
        std::string{"candles_1mo"},
        "month candle table");
}

void test_duckdb_schema_maps_candle_tables_to_python_periods()
{
    tradereview::core::assert_equal(
        tradereview::data::duckdb_period_for_candle_table("candles_1m"),
        std::string{"1min"},
        "1m table period");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_period_for_candle_table("candles_5m"),
        std::string{"5min"},
        "5m table period");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_period_for_candle_table("candles_1mo"),
        std::string{"1M"},
        "1mo table period");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_period_for_candle_table("candles_1h"),
        std::string{"1h"},
        "1h table period");
    tradereview::core::assert_equal(
        tradereview::data::duckdb_period_for_candle_table("candles_1D"),
        std::string{"1D"},
        "1D table period");
}

void test_duckdb_schema_validates_ticks_columns()
{
    const auto result = tradereview::data::validate_ticks_schema(ticks_schema());

    tradereview::core::assert_true(result.ok(), "ticks schema is valid");
    tradereview::core::assert_equal(
        static_cast<int>(result.error.code),
        static_cast<int>(tradereview::data::DataErrorCode::None),
        "no ticks error");
}

void test_duckdb_schema_reports_missing_ticks_column()
{
    auto schema = ticks_schema();
    schema.columns.pop_back();

    const auto result = tradereview::data::validate_ticks_schema(schema);

    tradereview::core::assert_true(!result.ok(), "ticks schema is invalid");
    tradereview::core::assert_equal(
        static_cast<int>(result.error.code),
        static_cast<int>(tradereview::data::DataErrorCode::MissingColumn),
        "missing ticks column error");
    tradereview::core::assert_equal(result.error.table, std::string{"ticks"}, "missing ticks table");
    tradereview::core::assert_equal(result.missing_columns.front(), std::string{"volume"}, "missing ticks volume");
}

void test_duckdb_schema_validates_candle_ohlcv_columns()
{
    const auto result = tradereview::data::validate_candle_schema(candle_schema(), {});

    tradereview::core::assert_true(result.ok(), "candle schema is valid");
}

void test_duckdb_schema_accepts_timestamp_aliases_case_insensitive()
{
    auto datetime_schema = candle_schema();
    datetime_schema.columns.front().name = "Datetime";
    const auto datetime_result = tradereview::data::validate_candle_schema(datetime_schema, {});
    tradereview::core::assert_true(datetime_result.ok(), "Datetime timestamp alias is valid");

    auto time_schema = candle_schema();
    time_schema.columns.front().name = "time";
    const auto time_result = tradereview::data::validate_candle_schema(time_schema, {});
    tradereview::core::assert_true(time_result.ok(), "time timestamp alias is valid");
}

void test_duckdb_schema_reports_missing_timestamp_alias()
{
    auto schema = candle_schema();
    schema.columns.erase(schema.columns.begin());

    const auto result = tradereview::data::validate_candle_schema(schema, {});

    tradereview::core::assert_true(!result.ok(), "missing timestamp alias is invalid");
    tradereview::core::assert_equal(
        static_cast<int>(result.error.code),
        static_cast<int>(tradereview::data::DataErrorCode::MissingColumn),
        "missing timestamp alias error");
    tradereview::core::assert_equal(result.missing_columns.front(), std::string{"timestamp"}, "missing timestamp");
}

void test_duckdb_schema_allows_optional_indicator_columns()
{
    auto schema = candle_schema();
    schema.columns.push_back({std::string{tradereview::data::IndicatorColumns::EMA20}, "DOUBLE"});
    schema.columns.push_back({std::string{tradereview::data::IndicatorColumns::RSI}, "DOUBLE"});

    const auto result = tradereview::data::validate_candle_schema(
        schema,
        {std::string{tradereview::data::IndicatorColumns::EMA20}, std::string{tradereview::data::IndicatorColumns::RSI}});

    tradereview::core::assert_true(result.ok(), "requested indicator columns are valid");
}

void test_duckdb_schema_reports_missing_indicator_column()
{
    const auto result = tradereview::data::validate_candle_schema(
        candle_schema(),
        {std::string{tradereview::data::IndicatorColumns::MACD}});

    tradereview::core::assert_true(!result.ok(), "missing indicator column is invalid");
    tradereview::core::assert_equal(
        static_cast<int>(result.error.code),
        static_cast<int>(tradereview::data::DataErrorCode::MissingColumn),
        "missing indicator error");
    tradereview::core::assert_equal(
        result.missing_columns.front(),
        std::string{tradereview::data::IndicatorColumns::MACD},
        "missing MACD column");
}

void test_duckdb_schema_allows_gap_row_columns_when_requested()
{
    auto schema = candle_schema();
    schema.columns.push_back({"is_gap", "BOOLEAN"});

    const auto result = tradereview::data::validate_candle_schema(schema, {}, true);

    tradereview::core::assert_true(result.ok(), "gap row marker is allowed");
}

void test_duckdb_repository_reports_unavailable_when_duckdb_is_disabled()
{
    tradereview::data::DuckDbRepository repository;

    bool open_threw = false;
    try {
        static_cast<void>(repository.open_readonly("missing.duckdb"));
    } catch (const std::runtime_error& ex) {
        open_threw = std::string{ex.what()}.find("TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF") != std::string::npos;
    }
    tradereview::core::assert_true(open_threw, "open reports unavailable DuckDB support");

    bool candles_threw = false;
    try {
        static_cast<void>(repository.query_candles({}));
    } catch (const std::runtime_error& ex) {
        candles_threw = std::string{ex.what()}.find("TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF") != std::string::npos;
    }
    tradereview::core::assert_true(candles_threw, "candle query reports unavailable DuckDB support");

    bool ticks_threw = false;
    try {
        static_cast<void>(repository.query_ticks({}, 10));
    } catch (const std::runtime_error& ex) {
        ticks_threw = std::string{ex.what()}.find("TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF") != std::string::npos;
    }
    tradereview::core::assert_true(ticks_threw, "tick query reports unavailable DuckDB support");

    bool replay_threw = false;
    try {
        static_cast<void>(repository.query_replay_ticks(0, 100, 10));
    } catch (const std::runtime_error& ex) {
        replay_threw = std::string{ex.what()}.find("TRADEREVIEW_NATIVE_WITH_DUCKDB is OFF") != std::string::npos;
    }
    tradereview::core::assert_true(replay_threw, "replay query reports unavailable DuckDB support");
}

struct RegisterDuckDbSchemaTests {
    RegisterDuckDbSchemaTests()
    {
        tradereview::tests::register_test(
            "duckdb schema maps period aliases to existing tables",
            test_duckdb_schema_maps_period_aliases_to_existing_tables);
        tradereview::tests::register_test(
            "duckdb schema maps candle tables to python periods",
            test_duckdb_schema_maps_candle_tables_to_python_periods);
        tradereview::tests::register_test(
            "duckdb schema validates ticks columns",
            test_duckdb_schema_validates_ticks_columns);
        tradereview::tests::register_test(
            "duckdb schema reports missing ticks column",
            test_duckdb_schema_reports_missing_ticks_column);
        tradereview::tests::register_test(
            "duckdb schema validates candle ohlcv columns",
            test_duckdb_schema_validates_candle_ohlcv_columns);
        tradereview::tests::register_test(
            "duckdb schema accepts timestamp aliases case insensitive",
            test_duckdb_schema_accepts_timestamp_aliases_case_insensitive);
        tradereview::tests::register_test(
            "duckdb schema reports missing timestamp alias",
            test_duckdb_schema_reports_missing_timestamp_alias);
        tradereview::tests::register_test(
            "duckdb schema allows optional indicator columns",
            test_duckdb_schema_allows_optional_indicator_columns);
        tradereview::tests::register_test(
            "duckdb schema reports missing indicator column",
            test_duckdb_schema_reports_missing_indicator_column);
        tradereview::tests::register_test(
            "duckdb schema allows gap row columns when requested",
            test_duckdb_schema_allows_gap_row_columns_when_requested);
#if !defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        tradereview::tests::register_test(
            "duckdb repository reports unavailable when duckdb is disabled",
            test_duckdb_repository_reports_unavailable_when_duckdb_is_disabled);
#endif
    }
};

const RegisterDuckDbSchemaTests register_duckdb_schema_tests;

} // namespace
