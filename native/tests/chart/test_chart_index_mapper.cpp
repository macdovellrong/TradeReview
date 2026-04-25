#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_mapper_finds_nearest_dense_x_for_timestamp()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{100, 200, 300});

    tradereview::core::assert_equal(mapper.nearest_dense_x(240), 1, "nearest dense x");
    tradereview::core::assert_equal(
        mapper.timestamp_at_dense_x(2),
        std::int64_t{300},
        "timestamp at dense x");
}

void test_mapper_extends_beyond_right_edge_using_median_step()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{100, 200, 300});

    tradereview::core::assert_equal(
        mapper.timestamp_from_x(4.0),
        std::int64_t{500},
        "timestamp beyond right edge");
}

void test_mapper_preserves_input_row_order_contract()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{300, 100, 200});

    tradereview::core::assert_equal(
        mapper.timestamp_at_dense_x(0),
        std::int64_t{300},
        "timestamp row order is preserved");
}

void test_timestamp_from_x_uses_left_tie_at_half_step()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{100, 200});

    tradereview::core::assert_equal(
        mapper.timestamp_from_x(0.5),
        std::int64_t{100},
        "half-step tie maps to left timestamp");
}

void test_timestamp_from_x_rejects_non_finite_x()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{100, 200});

    try {
        (void)mapper.timestamp_from_x(std::numeric_limits<double>::infinity());
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("timestamp_from_x should reject non-finite x");
}

void test_mapper_uses_default_positive_step_for_duplicate_timestamps()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(std::vector<std::int64_t>{100, 100, 100});

    tradereview::core::assert_equal(
        mapper.timestamp_from_x(3.0),
        std::int64_t{100 + (60LL * 1'000'000'000LL)},
        "duplicate timestamps use default positive step");
}

void test_empty_mapper_nearest_dense_x_throws_runtime_error()
{
    tradereview::chart::ChartIndexMapper mapper;

    try {
        (void)mapper.nearest_dense_x(100);
    } catch (const std::runtime_error&) {
        return;
    }

    throw std::runtime_error("empty mapper nearest_dense_x should throw runtime_error");
}

struct RegisterChartIndexMapperTests {
    RegisterChartIndexMapperTests()
    {
        tradereview::tests::register_test(
            "mapper finds nearest dense x for timestamp",
            test_mapper_finds_nearest_dense_x_for_timestamp);
        tradereview::tests::register_test(
            "mapper extends beyond right edge using median step",
            test_mapper_extends_beyond_right_edge_using_median_step);
        tradereview::tests::register_test(
            "mapper preserves input row order contract",
            test_mapper_preserves_input_row_order_contract);
        tradereview::tests::register_test(
            "timestamp_from_x uses left tie at half step",
            test_timestamp_from_x_uses_left_tie_at_half_step);
        tradereview::tests::register_test(
            "timestamp_from_x rejects non-finite x",
            test_timestamp_from_x_rejects_non_finite_x);
        tradereview::tests::register_test(
            "mapper uses default positive step for duplicate timestamps",
            test_mapper_uses_default_positive_step_for_duplicate_timestamps);
        tradereview::tests::register_test(
            "empty mapper nearest_dense_x throws runtime_error",
            test_empty_mapper_nearest_dense_x_throws_runtime_error);
    }
};

const RegisterChartIndexMapperTests register_chart_index_mapper_tests;

} // namespace
