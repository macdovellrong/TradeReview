#include "tradereview/drawing/DrawingInteractionState.h"

#include <algorithm>
#include <utility>

namespace tradereview::drawing {

void DrawingInteractionState::set_fib_settings(FibSettings settings)
{
    fib_settings_ = std::move(settings);
    if (!session_ || !is_fib_drawing_type(session_->type())) {
        return;
    }

    set_active_tool(session_->type());
}

void DrawingInteractionState::set_active_tool(DrawingType type)
{
    session_ = DrawingSession::for_type(type, fib_settings_);
    preview_.reset();
    touch_revision();
}

bool DrawingInteractionState::clear_active_tool()
{
    if (!session_ && !preview_) {
        return false;
    }

    session_.reset();
    preview_.reset();
    touch_revision();
    return true;
}

std::optional<DrawingType> DrawingInteractionState::active_tool() const
{
    if (!session_) {
        return std::nullopt;
    }
    return session_->type();
}

std::optional<DrawingSpec> DrawingInteractionState::add_point(DrawingPoint point)
{
    if (!session_) {
        return std::nullopt;
    }

    auto completed = session_->add_point(point);
    if (!completed.has_value()) {
        preview_.reset();
        touch_revision();
        return std::nullopt;
    }

    completed->id = next_drawing_id_++;
    drawings_.push_back(*completed);
    selected_drawing_id_ = completed->id;
    session_.reset();
    preview_.reset();
    touch_revision();
    return completed;
}

bool DrawingInteractionState::update_preview(std::optional<DrawingPoint> point)
{
    if (!session_) {
        if (!preview_) {
            return false;
        }
        preview_.reset();
        touch_revision();
        return true;
    }

    const auto had_preview = preview_.has_value();
    preview_ = point.has_value() ? session_->build_preview(*point) : std::nullopt;
    if (!had_preview && !preview_) {
        return false;
    }

    touch_revision();
    return true;
}

bool DrawingInteractionState::clear_drawings()
{
    if (drawings_.empty() && !preview_ && !selected_drawing_id_ && !session_) {
        return false;
    }

    drawings_.clear();
    preview_.reset();
    selected_drawing_id_.reset();
    session_.reset();
    touch_revision();
    return true;
}

bool DrawingInteractionState::delete_selected_drawing()
{
    if (!selected_drawing_id_.has_value()) {
        return false;
    }

    const auto selected_id = *selected_drawing_id_;
    const auto old_size = drawings_.size();
    drawings_.erase(
        std::remove_if(
            drawings_.begin(),
            drawings_.end(),
            [selected_id](const DrawingSpec& spec) {
                return spec.id == selected_id;
            }),
        drawings_.end());
    if (drawings_.size() == old_size) {
        selected_drawing_id_.reset();
        return false;
    }

    selected_drawing_id_.reset();
    touch_revision();
    return true;
}

bool DrawingInteractionState::select_drawing(std::uint64_t drawing_id)
{
    const auto found = std::find_if(
        drawings_.begin(),
        drawings_.end(),
        [drawing_id](const DrawingSpec& spec) {
            return spec.id == drawing_id;
        });
    if (found == drawings_.end()) {
        return false;
    }

    selected_drawing_id_ = drawing_id;
    return true;
}

const std::vector<DrawingSpec>& DrawingInteractionState::drawings() const
{
    return drawings_;
}

std::optional<DrawingSpec> DrawingInteractionState::preview() const
{
    return preview_;
}

std::optional<std::uint64_t> DrawingInteractionState::selected_drawing_id() const
{
    return selected_drawing_id_;
}

std::uint64_t DrawingInteractionState::revision() const
{
    return revision_;
}

void DrawingInteractionState::touch_revision()
{
    ++revision_;
}

} // namespace tradereview::drawing
