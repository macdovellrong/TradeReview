#include "tradereview/sync/CrosshairSyncController.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

using tradereview::sync::CenterTimeUpdate;
using tradereview::sync::CrosshairSyncController;
using tradereview::sync::CrosshairUpdate;
using tradereview::sync::YCenterUpdate;

struct RecordedCrosshair {
    std::uint64_t target_chart_id = 0;
    std::int64_t timestamp_ns = 0;
    double price = 0.0;
    double dense_x = 0.0;
};

struct RecordedCenter {
    std::uint64_t target_chart_id = 0;
    std::int64_t timestamp_ns = 0;
    double dense_x = 0.0;
    bool has_price = false;
    double price = 0.0;
};

void test_crosshair_sync_uses_canonical_timestamp_price_and_target_dense_x()
{
    CrosshairSyncController controller;
    std::vector<RecordedCrosshair> updates;

    controller.register_chart(
        1,
        [](std::int64_t timestamp_ns) -> std::optional<double> { return static_cast<double>(timestamp_ns / 10); },
        [&](const CrosshairUpdate& update) {
            updates.push_back({update.target_chart_id, update.timestamp_ns, update.price, update.dense_x});
        },
        {},
        {});
    controller.register_chart(
        2,
        [](std::int64_t timestamp_ns) -> std::optional<double> { return 100.0 + static_cast<double>(timestamp_ns / 10); },
        [&](const CrosshairUpdate& update) {
            updates.push_back({update.target_chart_id, update.timestamp_ns, update.price, update.dense_x});
        },
        {},
        {});

    tradereview::core::assert_true(controller.sync_crosshair_from(1, 250, 1936.25), "crosshair sync emits update");
    tradereview::core::assert_equal(updates.size(), std::size_t{1}, "source chart is skipped");
    tradereview::core::assert_equal(updates[0].target_chart_id, std::uint64_t{2}, "target chart id");
    tradereview::core::assert_equal(updates[0].timestamp_ns, std::int64_t{250}, "canonical timestamp");
    tradereview::core::assert_near(updates[0].price, 1936.25, 0.000001, "canonical price");
    tradereview::core::assert_near(updates[0].dense_x, 125.0, 0.000001, "target dense x");
}

void test_crosshair_sync_skips_disabled_source_and_targets()
{
    CrosshairSyncController controller;
    std::vector<RecordedCrosshair> updates;

    controller.register_chart(
        1,
        [](std::int64_t) -> std::optional<double> { return 1.0; },
        [&](const CrosshairUpdate& update) {
            updates.push_back({update.target_chart_id, update.timestamp_ns, update.price, update.dense_x});
        },
        {},
        {});
    controller.register_chart(
        2,
        [](std::int64_t) -> std::optional<double> { return 2.0; },
        [&](const CrosshairUpdate& update) {
            updates.push_back({update.target_chart_id, update.timestamp_ns, update.price, update.dense_x});
        },
        {},
        {});

    controller.set_chart_enabled(2, false);
    tradereview::core::assert_true(!controller.sync_crosshair_from(2, 100, 10.0), "disabled source is ignored");
    tradereview::core::assert_equal(updates.size(), std::size_t{0}, "disabled source emits nothing");

    tradereview::core::assert_true(!controller.sync_crosshair_from(1, 100, 10.0), "disabled target is skipped");
    tradereview::core::assert_equal(updates.size(), std::size_t{0}, "disabled target receives nothing");
}

void test_crosshair_sync_avoids_feedback_loops()
{
    CrosshairSyncController controller;
    int updates = 0;

    controller.register_chart(
        1,
        [](std::int64_t) -> std::optional<double> { return 1.0; },
        [&](const CrosshairUpdate&) {
            ++updates;
            controller.sync_crosshair_from(1, 200, 20.0);
        },
        {},
        {});
    controller.register_chart(
        2,
        [](std::int64_t) -> std::optional<double> { return 2.0; },
        [&](const CrosshairUpdate&) {
            ++updates;
            controller.sync_crosshair_from(2, 200, 20.0);
        },
        {},
        {});

    tradereview::core::assert_true(controller.sync_crosshair_from(1, 100, 10.0), "outer sync emits");
    tradereview::core::assert_equal(updates, 1, "nested sync is suppressed");
}

void test_center_sync_targets_all_enabled_charts_with_optional_price()
{
    CrosshairSyncController controller;
    std::vector<RecordedCenter> updates;

    controller.register_chart(
        1,
        [](std::int64_t timestamp_ns) -> std::optional<double> { return static_cast<double>(timestamp_ns); },
        {},
        [&](const CenterTimeUpdate& update) {
            updates.push_back({
                update.target_chart_id,
                update.timestamp_ns,
                update.dense_x,
                update.price.has_value(),
                update.price.value_or(0.0),
            });
        },
        {});
    controller.register_chart(
        2,
        [](std::int64_t timestamp_ns) -> std::optional<double> { return 0.5 * static_cast<double>(timestamp_ns); },
        {},
        [&](const CenterTimeUpdate& update) {
            updates.push_back({
                update.target_chart_id,
                update.timestamp_ns,
                update.dense_x,
                update.price.has_value(),
                update.price.value_or(0.0),
            });
        },
        {});

    tradereview::core::assert_true(controller.sync_center_from(1, 80, 1910.5), "center sync emits");
    tradereview::core::assert_equal(updates.size(), std::size_t{2}, "center includes source and target");
    tradereview::core::assert_equal(updates[0].target_chart_id, std::uint64_t{1}, "source chart also centers");
    tradereview::core::assert_near(updates[0].dense_x, 80.0, 0.000001, "source dense center");
    tradereview::core::assert_equal(updates[1].target_chart_id, std::uint64_t{2}, "target chart centers");
    tradereview::core::assert_near(updates[1].dense_x, 40.0, 0.000001, "target dense center");
    tradereview::core::assert_true(updates[1].has_price, "optional price is carried");
    tradereview::core::assert_near(updates[1].price, 1910.5, 0.000001, "center price");
}

void test_y_center_sync_targets_all_enabled_charts()
{
    CrosshairSyncController controller;
    std::vector<double> prices;

    controller.register_chart(1, [](std::int64_t) -> std::optional<double> { return 0.0; }, {}, {}, [&](const YCenterUpdate& update) {
        prices.push_back(update.price);
    });
    controller.register_chart(2, [](std::int64_t) -> std::optional<double> { return 0.0; }, {}, {}, [&](const YCenterUpdate& update) {
        prices.push_back(update.price);
    });
    controller.set_chart_enabled(2, false);

    tradereview::core::assert_true(controller.sync_y_center_from(1, 1899.75), "y-center sync emits");
    tradereview::core::assert_equal(prices.size(), std::size_t{1}, "disabled target skipped");
    tradereview::core::assert_near(prices[0], 1899.75, 0.000001, "y center price");
}

struct RegisterCrosshairSyncTests {
    RegisterCrosshairSyncTests()
    {
        tradereview::tests::register_test(
            "crosshair sync uses canonical timestamp price and target dense x",
            test_crosshair_sync_uses_canonical_timestamp_price_and_target_dense_x);
        tradereview::tests::register_test(
            "crosshair sync skips disabled source and targets",
            test_crosshair_sync_skips_disabled_source_and_targets);
        tradereview::tests::register_test(
            "crosshair sync avoids feedback loops",
            test_crosshair_sync_avoids_feedback_loops);
        tradereview::tests::register_test(
            "center sync targets all enabled charts with optional price",
            test_center_sync_targets_all_enabled_charts_with_optional_price);
        tradereview::tests::register_test(
            "y center sync targets all enabled charts",
            test_y_center_sync_targets_all_enabled_charts);
    }
};

const RegisterCrosshairSyncTests register_crosshair_sync_tests;

} // namespace
