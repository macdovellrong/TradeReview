#include "tradereview/core/Assertions.h"
#include "tradereview/drawing/DrawingInteractionState.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::drawing::DrawingPoint point(std::int64_t timestamp_ns, double price)
{
    return tradereview::drawing::DrawingPoint{timestamp_ns, price};
}

void test_clear_drawings_resets_active_session_and_preview()
{
    tradereview::drawing::DrawingInteractionState state;

    state.set_active_tool(tradereview::drawing::DrawingType::Line);
    state.add_point(point(100, 100.0));
    tradereview::core::assert_true(state.active_tool().has_value(), "line tool is active after first point");
    tradereview::core::assert_true(
        state.update_preview(point(200, 110.0)),
        "line preview update changes state");
    tradereview::core::assert_true(state.preview().has_value(), "line preview exists");

    tradereview::core::assert_true(state.clear_drawings(), "clear changes in-progress state");

    tradereview::core::assert_true(!state.active_tool().has_value(), "clear resets active tool");
    tradereview::core::assert_true(!state.preview().has_value(), "clear resets preview");
    tradereview::core::assert_true(
        !state.add_point(point(200, 110.0)).has_value(),
        "second point after clear does not complete stale line");
    tradereview::core::assert_true(state.drawings().empty(), "clear leaves no completed drawings");
}

void test_completed_drawing_is_selected_and_delete_removes_it()
{
    tradereview::drawing::DrawingInteractionState state;

    state.set_active_tool(tradereview::drawing::DrawingType::HorizontalLine);
    const auto spec = state.add_point(point(100, 100.0));

    tradereview::core::assert_true(spec.has_value(), "hline completes");
    tradereview::core::assert_equal(state.drawings().size(), std::size_t{1}, "drawing stored");
    tradereview::core::assert_true(state.selected_drawing_id().has_value(), "completed drawing selected");
    tradereview::core::assert_equal(*state.selected_drawing_id(), spec->id, "selected drawing id");

    tradereview::core::assert_true(state.delete_selected_drawing(), "selected drawing deleted");
    tradereview::core::assert_true(state.drawings().empty(), "drawing removed");
    tradereview::core::assert_true(!state.selected_drawing_id().has_value(), "selection cleared");
}

struct RegisterDrawingInteractionStateTests {
    RegisterDrawingInteractionStateTests()
    {
        tradereview::tests::register_test(
            "clear drawings resets active session and preview",
            test_clear_drawings_resets_active_session_and_preview);
        tradereview::tests::register_test(
            "completed drawing is selected and delete removes it",
            test_completed_drawing_is_selected_and_delete_removes_it);
    }
};

const RegisterDrawingInteractionStateTests register_drawing_interaction_state_tests;

} // namespace
