#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataError.h"
#include "tradereview/data/DataSetInfo.h"
#include "tradereview/data/IDataStore.h"
#include "tradereview/data/IndicatorColumns.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::CandleWindow consistent_window()
{
    tradereview::data::CandleWindow window;
    window.loaded_range = {100, 400};
    window.visible_range = {150, 350};
    window.timestamp_ns = {100, 200, 300};
    window.open = {1.0, 2.0, 3.0};
    window.high = {1.5, 2.5, 3.5};
    window.low = {0.5, 1.5, 2.5};
    window.close = {1.25, 2.25, 3.25};
    window.volume = {10.0, 20.0, 30.0};
    return window;
}

void test_metadata_only_source_state()
{
    tradereview::data::DataSetInfo info;
    info.dataset_path = "V:/gold/sample.duckdb";
    info.tick_count = 12;
    info.tick_range = {1000, 9000};
    info.available_periods = {"1min", "5min"};
    info.available_indicators = {
        std::string{tradereview::data::IndicatorColumns::EMA20},
        std::string{tradereview::data::IndicatorColumns::MACD}};
    info.schema_version = "schema-v1";
    info.indicator_version = "indicator-v1";
    info.metadata_only = true;

    tradereview::core::assert_equal(info.dataset_path, std::string{"V:/gold/sample.duckdb"}, "dataset path");
    tradereview::core::assert_equal(info.tick_count, std::int64_t{12}, "tick count");
    tradereview::core::assert_equal(info.tick_range.start_ns, std::int64_t{1000}, "tick range start");
    tradereview::core::assert_equal(info.available_periods.size(), std::size_t{2}, "available period count");
    tradereview::core::assert_equal(info.available_indicators.front(), std::string{"EMA20"}, "available indicator");
    tradereview::core::assert_true(info.metadata_only, "metadata-only flag");
}

void test_indicator_contracts_are_canonical()
{
    const std::vector<std::string_view> expected = {
        "EMA20",
        "EMA30",
        "EMA40",
        "EMA50",
        "EMA60",
        "EMA100",
        "EMA240",
        "BB_Upper",
        "BB_Lower",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "RSI"};

    tradereview::core::assert_equal(
        tradereview::data::IndicatorColumns::all().size(),
        expected.size(),
        "canonical indicator count");

    for (std::size_t index = 0; index < expected.size(); ++index) {
        tradereview::core::assert_true(
            tradereview::data::IndicatorColumns::all()[index] == expected[index],
            "canonical indicator name");
    }
}

void test_indicator_consistency()
{
    auto window = consistent_window();
    window.indicators[std::string{tradereview::data::IndicatorColumns::EMA20}] = {1.0, 2.0, 3.0};
    window.indicators[std::string{tradereview::data::IndicatorColumns::MACD}] = {-1.0, 0.0, 1.0};

    tradereview::core::assert_true(window.has_consistent_ohlcv(), "OHLCV columns are consistent");
    tradereview::core::assert_true(window.has_consistent_indicators(), "indicator columns are consistent");
    tradereview::core::assert_true(window.has_consistent_columns(), "all columns are consistent");
    tradereview::core::assert_true(window.has_loaded_range(), "loaded range is present");
    tradereview::core::assert_true(window.has_visible_range(), "visible range is present");
}

void test_inconsistent_columns_are_reported()
{
    auto window = consistent_window();
    window.high.pop_back();

    tradereview::core::assert_true(!window.has_consistent_ohlcv(), "inconsistent OHLCV columns are reported");
    tradereview::core::assert_true(!window.has_consistent_columns(), "inconsistent columns are reported");

    window = consistent_window();
    window.indicators[std::string{tradereview::data::IndicatorColumns::RSI}] = {50.0, 60.0};

    tradereview::core::assert_true(!window.has_consistent_indicators(), "inconsistent indicator columns are reported");
    tradereview::core::assert_true(!window.has_consistent_columns(), "inconsistent indicator columns affect all columns");
}

void test_request_defaults()
{
    tradereview::data::CandleWindowRequest request;

    tradereview::core::assert_equal(request.chart_id, std::uint64_t{0}, "default chart id");
    tradereview::core::assert_equal(request.generation, std::uint64_t{0}, "default generation");
    tradereview::core::assert_equal(request.pixel_width, 0, "default pixel width");
    tradereview::core::assert_equal(request.buffer_multiplier, 2.0, "default buffer multiplier");
    tradereview::core::assert_true(request.include_indicators, "indicators included by default");
    tradereview::core::assert_equal(request.requested_indicators.size(), std::size_t{0}, "default requested indicators");
    tradereview::core::assert_equal(request.warmup_bars, 0, "default warmup bars");
    tradereview::core::assert_equal(request.right_padding_bars, 0, "default right padding bars");
}

void test_data_error_fields_are_explicit()
{
    tradereview::data::DataError error;
    error.code = tradereview::data::DataErrorCode::MissingTable;
    error.message = "missing candles table";
    error.path = "dataset.duckdb";
    error.table = "candles_1min";

    tradereview::core::assert_true(
        error.code == tradereview::data::DataErrorCode::MissingTable,
        "stable data error code");
    tradereview::core::assert_equal(error.message, std::string{"missing candles table"}, "data error message");
    tradereview::core::assert_equal(error.path, std::string{"dataset.duckdb"}, "data error path");
    tradereview::core::assert_equal(error.table, std::string{"candles_1min"}, "data error table");
}

struct RegisterDataContractTests {
    RegisterDataContractTests()
    {
        tradereview::tests::register_test(
            "metadata-only source state is explicit",
            test_metadata_only_source_state);
        tradereview::tests::register_test(
            "indicator contracts are canonical",
            test_indicator_contracts_are_canonical);
        tradereview::tests::register_test(
            "indicator consistency is reported",
            test_indicator_consistency);
        tradereview::tests::register_test(
            "inconsistent columns are reported",
            test_inconsistent_columns_are_reported);
        tradereview::tests::register_test(
            "candle window request defaults are stable",
            test_request_defaults);
        tradereview::tests::register_test(
            "data error fields are explicit",
            test_data_error_fields_are_explicit);
    }
};

const RegisterDataContractTests register_data_contract_tests;

} // namespace
