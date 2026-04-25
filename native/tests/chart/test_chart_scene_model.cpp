#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

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
    tradereview::core::assert_equal(model.bump_generation(), std::uint64_t{1}, "first generation bump returns 1");
    tradereview::core::assert_equal(model.bump_generation(), std::uint64_t{2}, "second generation bump returns 2");
    tradereview::core::assert_equal(model.generation(), std::uint64_t{2}, "generation after two bumps");

    auto window = sample_window(model.generation());

    tradereview::core::assert_true(model.apply_window(std::move(window)), "matching generation is accepted");
    tradereview::core::assert_equal(model.row_count(), std::size_t{2}, "scene model row count");
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

struct RegisterChartSceneModelTests {
    RegisterChartSceneModelTests()
    {
        tradereview::tests::register_test(
            "scene model accepts matching generation",
            test_scene_model_accepts_matching_generation);
        tradereview::tests::register_test(
            "scene model rejects stale generation",
            test_scene_model_rejects_stale_generation);
        tradereview::tests::register_test(
            "scene model rejects inconsistent columns",
            test_scene_model_rejects_inconsistent_columns);
    }
};

const RegisterChartSceneModelTests register_chart_scene_model_tests;

} // namespace
