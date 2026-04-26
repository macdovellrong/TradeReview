#pragma once

#include "tradereview/drawing/DrawingSession.h"
#include "tradereview/drawing/DrawingSpec.h"
#include "tradereview/drawing/FibSettings.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tradereview::drawing {

class DrawingInteractionState final {
public:
    void set_fib_settings(FibSettings settings);
    void set_active_tool(DrawingType type);
    bool clear_active_tool();
    [[nodiscard]] std::optional<DrawingType> active_tool() const;

    [[nodiscard]] std::optional<DrawingSpec> add_point(DrawingPoint point);
    bool update_preview(std::optional<DrawingPoint> point);
    bool clear_drawings();
    bool delete_selected_drawing();
    bool select_drawing(std::uint64_t drawing_id);

    [[nodiscard]] const std::vector<DrawingSpec>& drawings() const;
    [[nodiscard]] std::optional<DrawingSpec> preview() const;
    [[nodiscard]] std::optional<std::uint64_t> selected_drawing_id() const;
    [[nodiscard]] std::uint64_t revision() const;

private:
    void touch_revision();

    FibSettings fib_settings_ = default_fib_settings();
    std::optional<DrawingSession> session_;
    std::optional<DrawingSpec> preview_;
    std::vector<DrawingSpec> drawings_;
    std::optional<std::uint64_t> selected_drawing_id_;
    std::uint64_t next_drawing_id_ = 1;
    std::uint64_t revision_ = 0;
};

} // namespace tradereview::drawing
