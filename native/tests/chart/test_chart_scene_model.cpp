#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/IndicatorColumns.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::CandleWindow sample_window(std::uint64_t generation)
{
    tradereview::data::CandleWindow window;
    window.generation = generation;
    window.timestamp_ns = {100, 200};
    window.open = {1.0, 2.0};
    window.high = {1.5, 2.5};
    window.low = {0.5, 1.5};
    window.close = {1.25, 2.25};
    window.volume = {10.0, 20.0};
    return window;
}

void test_scene_model_accepts_matching_generation()
{
    tradereview::chart::ChartSceneModel model;
    tradereview::core::assert_equal(model.revision(), std::uint64_t{0}, "initial revision");
    tradereview::core::assert_equal(model.bump_generation(), std::uint64_t{1}, "first generation bump returns 1");
    tradereview::core::assert_equal(model.bump_generation(), std::uint64_t{2}, "second generation bump returns 2");
    tradereview::core::assert_equal(model.generation(), std::uint64_t{2}, "generation after two bumps");

    auto window = sample_window(model.generation());

    tradereview::core::assert_true(model.apply_window(std::move(window)), "matching generation is accepted");
    tradereview::core::assert_equal(model.row_count(), std::size_t{2}, "scene model row count");
    tradereview::core::assert_equal(model.revision(), std::uint64_t{1}, "revision increments after accepted window");
}

void test_scene_model_revisions_same_generation_replacements()
{
    tradereview::chart::ChartSceneModel model;
    model.bump_generation();

    auto first = sample_window(model.generation());
    auto second = sample_window(model.generation());
    second.close = {1.35, 2.35};

    tradereview::core::assert_true(model.apply_window(std::move(first)), "first matching window is accepted");
    tradereview::core::assert_true(model.apply_window(std::move(second)), "same generation replacement is accepted");
    tradereview::core::assert_equal(model.revision(), std::uint64_t{2}, "revision tracks same generation replacement");
}

void test_scene_model_tracks_visible_dense_range_separately()
{
    tradereview::chart::ChartSceneModel model;
    model.bump_generation();
    auto window = sample_window(model.generation());
    tradereview::core::assert_true(model.apply_window(std::move(window)), "matching generation is accepted");

    const auto revision_after_window = model.revision();
    tradereview::core::assert_true(model.set_visible_dense_range({1.0, 5.0}), "visible range change is accepted");

    const auto range = model.visible_dense_range();
    tradereview::core::assert_near(range.start_x, 1.0, 0.000001, "visible dense start");
    tradereview::core::assert_near(range.end_x, 5.0, 0.000001, "visible dense end");
    tradereview::core::assert_equal(model.row_count(), std::size_t{2}, "loaded row count stays separate");
    tradereview::core::assert_equal(model.revision(), revision_after_window + 1, "visible range increments revision");
}

void test_scene_model_tracks_manual_price_range()
{
    tradereview::chart::ChartSceneModel model;

    const auto initial_revision = model.revision();
    tradereview::core::assert_true(
        model.set_price_range_override(tradereview::chart::PriceRange{20.0, 10.0}),
        "manual price range is accepted");
    const auto range = model.price_range_override();

    tradereview::core::assert_true(range.has_value(), "manual price range is stored");
    tradereview::core::assert_near(range->first, 10.0, 0.000001, "manual price minimum is normalized");
    tradereview::core::assert_near(range->second, 20.0, 0.000001, "manual price maximum is normalized");
    tradereview::core::assert_equal(model.revision(), initial_revision + 1, "manual price range increments revision");
    tradereview::core::assert_true(
        !model.set_price_range_override(tradereview::chart::PriceRange{10.0, 20.0}),
        "same manual range is ignored");
    tradereview::core::assert_true(model.set_price_range_override(std::nullopt), "manual price range can be cleared");
    tradereview::core::assert_true(!model.price_range_override().has_value(), "manual price range is cleared");
}

void test_scene_model_rejects_stale_generation()
{
    tradereview::chart::ChartSceneModel model;
    model.bump_generation();
    model.bump_generation();

    auto window = sample_window(1);

    tradereview::core::assert_true(!model.apply_window(std::move(window)), "stale generation is rejected");
    tradereview::core::assert_equal(model.row_count(), std::size_t{0}, "scene model row count remains empty");
}

void test_scene_model_rejects_inconsistent_columns()
{
    tradereview::chart::ChartSceneModel model;
    model.bump_generation();

    auto window = sample_window(model.generation());
    window.close.pop_back();

    tradereview::core::assert_true(!model.apply_window(std::move(window)), "inconsistent columns are rejected");
    tradereview::core::assert_equal(model.row_count(), std::size_t{0}, "scene model row count remains empty");
}

void test_scene_model_tracks_loading_state()
{
    tradereview::chart::ChartSceneModel model;

    tradereview::core::assert_true(!model.loading(), "model starts not loading");
    const auto initial_revision = model.revision();
    tradereview::core::assert_true(model.set_loading(true), "loading can be enabled");
    tradereview::core::assert_true(model.loading(), "loading flag is true");
    tradereview::core::assert_equal(model.revision(), initial_revision + 1, "loading change increments revision");
    tradereview::core::assert_true(!model.set_loading(true), "same loading state is ignored");
    tradereview::core::assert_true(model.set_loading(false), "loading can be disabled");
    tradereview::core::assert_true(!model.loading(), "loading flag is false");
}

bool contains(const std::vector<std::string>& values, std::string_view target)
{
    for (const auto& value : values) {
        if (value == target) {
            return true;
        }
    }
    return false;
}

void test_scene_model_tracks_indicator_visibility_and_requested_columns()
{
    tradereview::chart::ChartSceneModel model;

    auto requested = model.requested_indicators();
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::EMA20), "EMA20 requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::BB_Upper), "BB upper requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::MACD_Hist), "MACD histogram requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::RSI6), "RSI6 requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::RSI12), "RSI12 requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::RSI24), "RSI24 requested by default");
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::RSI), "RSI requested by default");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::EMA100), "EMA100 disabled by default");

    const auto revision_before = model.revision();
    tradereview::core::assert_true(model.set_indicator_enabled("EMA100", true), "EMA100 can be enabled");
    tradereview::core::assert_true(model.set_bollinger_bands_enabled(false), "BB can be disabled");
    tradereview::core::assert_true(model.set_indicator_panels_enabled(false), "indicator panels can be disabled");

    requested = model.requested_indicators();
    tradereview::core::assert_true(contains(requested, tradereview::data::IndicatorColumns::EMA100), "EMA100 requested when enabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::BB_Upper), "BB upper not requested when disabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::MACD), "MACD not requested when panels disabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::RSI6), "RSI6 not requested when panels disabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::RSI12), "RSI12 not requested when panels disabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::RSI24), "RSI24 not requested when panels disabled");
    tradereview::core::assert_true(!contains(requested, tradereview::data::IndicatorColumns::RSI), "RSI not requested when panels disabled");
    tradereview::core::assert_equal(model.revision(), revision_before + 3, "indicator changes increment revision");
}

struct RegisterChartSceneModelTests {
    RegisterChartSceneModelTests()
    {
        tradereview::tests::register_test(
            "scene model accepts matching generation",
            test_scene_model_accepts_matching_generation);
        tradereview::tests::register_test(
            "scene model revisions same generation replacements",
            test_scene_model_revisions_same_generation_replacements);
        tradereview::tests::register_test(
            "scene model tracks visible dense range separately",
            test_scene_model_tracks_visible_dense_range_separately);
        tradereview::tests::register_test(
            "scene model tracks manual price range",
            test_scene_model_tracks_manual_price_range);
        tradereview::tests::register_test(
            "scene model rejects stale generation",
            test_scene_model_rejects_stale_generation);
        tradereview::tests::register_test(
            "scene model rejects inconsistent columns",
            test_scene_model_rejects_inconsistent_columns);
        tradereview::tests::register_test(
            "scene model tracks loading state",
            test_scene_model_tracks_loading_state);
        tradereview::tests::register_test(
            "scene model tracks indicator visibility and requested columns",
            test_scene_model_tracks_indicator_visibility_and_requested_columns);
    }
};

const RegisterChartSceneModelTests register_chart_scene_model_tests;

} // namespace
