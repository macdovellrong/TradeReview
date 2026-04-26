#include "tradereview/core/Assertions.h"
#include "tradereview/data/DuckDbRepository.h"
#include "tradereview/data/IndicatorColumns.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string>

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
#include <duckdb.h>
#endif

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
class WritableDuckDb {
public:
    explicit WritableDuckDb(const std::filesystem::path& path)
    {
        if (duckdb_open(path.string().c_str(), &database_) == DuckDBError) {
            throw std::runtime_error("failed to create test DuckDB database");
        }
        if (duckdb_connect(database_, &connection_) == DuckDBError) {
            duckdb_close(&database_);
            throw std::runtime_error("failed to connect test DuckDB database");
        }
    }

    ~WritableDuckDb()
    {
        if (connection_ != nullptr) {
            duckdb_disconnect(&connection_);
        }
        if (database_ != nullptr) {
            duckdb_close(&database_);
        }
    }

    WritableDuckDb(const WritableDuckDb&) = delete;
    WritableDuckDb& operator=(const WritableDuckDb&) = delete;

    void exec(const std::string& sql)
    {
        duckdb_result result;
        const auto state = duckdb_query(connection_, sql.c_str(), &result);
        if (state == DuckDBError) {
            std::string error = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            throw std::runtime_error(error);
        }
        duckdb_destroy_result(&result);
    }

private:
    duckdb_database database_ = nullptr;
    duckdb_connection connection_ = nullptr;
};

std::filesystem::path test_database_path()
{
    auto path = std::filesystem::temp_directory_path() / "tradereview_duckdb_repository_smoke.duckdb";
    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".wal");
    return path;
}

void create_repository_smoke_database(const std::filesystem::path& path)
{
    WritableDuckDb database(path);
    database.exec(
        "CREATE TABLE ticks(timestamp TIMESTAMP, price DOUBLE, volume DOUBLE);"
        "INSERT INTO ticks VALUES "
        "('2024-01-01 00:00:00.000001', 100.0, 1.0),"
        "('2024-01-01 00:01:00.000001', 101.0, 2.0),"
        "('2024-01-01 00:02:00.000001', 102.0, 3.0);");
    database.exec(
        "CREATE TABLE candles_1m("
        "timestamp TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, "
        "EMA20 DOUBLE, RSI DOUBLE);"
        "INSERT INTO candles_1m VALUES "
        "('2024-01-01 00:00:00.000001', 100.0, 101.0, 99.0, 100.5, 10.0, 100.25, 55.0),"
        "('2024-01-01 00:01:00.000001', 100.5, 102.0, 100.0, 101.5, 11.0, 100.75, 56.0),"
        "('2024-01-01 00:02:00.000001', 101.5, 103.0, 101.0, 102.5, 12.0, 101.25, 57.0);");
}

void create_repository_alias_database(const std::filesystem::path& path)
{
    WritableDuckDb database(path);
    database.exec(
        "CREATE TABLE ticks(Datetime TIMESTAMP, price DOUBLE, volume DOUBLE);"
        "INSERT INTO ticks VALUES ('2024-01-02 00:00:00.000001', 200.0, 1.0);");
    database.exec(
        "CREATE TABLE candles_1m("
        "time TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE);"
        "INSERT INTO candles_1m VALUES ('2024-01-02 00:00:00.000001', 200.0, 201.0, 199.0, 200.5, 5.0);");
}

void test_duckdb_repository_reads_metadata_and_candle_window()
{
    const auto path = test_database_path();
    create_repository_smoke_database(path);

    tradereview::data::DuckDbRepository repository;
    const auto info = repository.open_readonly(path.string());

    tradereview::core::assert_equal(info.dataset_path, path.string(), "metadata dataset path");
    tradereview::core::assert_true(info.metadata_only, "metadata-only flag");
    tradereview::core::assert_equal(info.tick_count, std::int64_t{3}, "metadata tick count");
    tradereview::core::assert_equal(info.tick_range.start_ns, std::int64_t{1704067200000001000}, "tick range start");
    tradereview::core::assert_equal(info.tick_range.end_ns, std::int64_t{1704067320000001000}, "tick range end");
    tradereview::core::assert_equal(info.available_periods.size(), std::size_t{1}, "available period count");
    tradereview::core::assert_equal(info.available_periods.front(), std::string{"1min"}, "available period");
    tradereview::core::assert_equal(info.available_indicators.size(), std::size_t{2}, "available indicator count");
    tradereview::core::assert_equal(info.available_indicators.front(), std::string{"EMA20"}, "first indicator");
    tradereview::core::assert_equal(info.available_indicators.back(), std::string{"RSI"}, "second indicator");

    tradereview::data::CandleWindowRequest request;
    request.chart_id = 42;
    request.generation = 7;
    request.requested_period = "1min";
    request.visible_range = {1704067200000001000, 1704067260000001000};
    request.requested_indicators = {
        std::string{tradereview::data::IndicatorColumns::EMA20},
        std::string{tradereview::data::IndicatorColumns::MACD},
        std::string{tradereview::data::IndicatorColumns::RSI}};

    const auto window = repository.query_candles(request);

    tradereview::core::assert_equal(window.chart_id, std::uint64_t{42}, "window chart id");
    tradereview::core::assert_equal(window.generation, std::uint64_t{7}, "window generation");
    tradereview::core::assert_equal(window.requested_period, std::string{"1min"}, "window requested period");
    tradereview::core::assert_equal(window.actual_period, std::string{"1min"}, "window actual period");
    tradereview::core::assert_equal(window.row_count(), std::size_t{2}, "window row count");
    tradereview::core::assert_equal(window.timestamp_ns.front(), std::int64_t{1704067200000001000}, "first candle time");
    tradereview::core::assert_equal(window.timestamp_ns.back(), std::int64_t{1704067260000001000}, "last candle time");
    tradereview::core::assert_near(window.close.back(), 101.5, 0.000001, "last close");
    tradereview::core::assert_true(window.indicators.contains("EMA20"), "EMA20 returned");
    tradereview::core::assert_true(!window.indicators.contains("MACD"), "missing MACD skipped");
    tradereview::core::assert_true(window.indicators.contains("RSI"), "RSI returned");
    tradereview::core::assert_near(window.indicators.at("EMA20").front(), 100.25, 0.000001, "first EMA20");
    tradereview::core::assert_equal(window.loaded_range.start_ns, std::int64_t{1704067200000001000}, "loaded start");
    tradereview::core::assert_equal(window.loaded_range.end_ns, std::int64_t{1704067260000001000}, "loaded end");
    tradereview::core::assert_true(!window.from_cache, "DuckDB query is not cached");

    request.visible_range = {1704153600000001000, 1704153660000001000};
    const auto empty_window = repository.query_candles(request);
    tradereview::core::assert_equal(empty_window.row_count(), std::size_t{0}, "empty window row count");
    tradereview::core::assert_true(!empty_window.has_loaded_range(), "empty window has no loaded range");

    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".wal");
}

void test_duckdb_repository_uses_timestamp_aliases()
{
    const auto path = test_database_path();
    create_repository_alias_database(path);

    tradereview::data::DuckDbRepository repository;
    const auto info = repository.open_readonly(path.string());

    tradereview::core::assert_equal(info.tick_count, std::int64_t{1}, "alias metadata tick count");
    tradereview::core::assert_equal(info.tick_range.start_ns, std::int64_t{1704153600000001000}, "alias tick start");

    tradereview::data::CandleWindowRequest request;
    request.chart_id = 1;
    request.generation = 1;
    request.requested_period = "1min";
    request.visible_range = {1704153600000001000, 1704153600000001000};

    const auto window = repository.query_candles(request);
    tradereview::core::assert_equal(window.row_count(), std::size_t{1}, "alias candle row count");
    tradereview::core::assert_equal(window.timestamp_ns.front(), std::int64_t{1704153600000001000}, "alias candle time");
    tradereview::core::assert_near(window.close.front(), 200.5, 0.000001, "alias candle close");

    std::filesystem::remove(path);
    std::filesystem::remove(path.string() + ".wal");
}
#endif

struct RegisterDuckDbRepositoryTests {
    RegisterDuckDbRepositoryTests()
    {
#if defined(TRADEREVIEW_NATIVE_WITH_DUCKDB)
        tradereview::tests::register_test(
            "duckdb repository reads metadata and candle window",
            test_duckdb_repository_reads_metadata_and_candle_window);
        tradereview::tests::register_test(
            "duckdb repository uses timestamp aliases",
            test_duckdb_repository_uses_timestamp_aliases);
#endif
    }
};

const RegisterDuckDbRepositoryTests register_duckdb_repository_tests;

} // namespace
