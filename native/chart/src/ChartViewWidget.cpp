#include "tradereview/chart/ChartViewWidget.h"

#include "tradereview/app/AppTheme.h"
#include "tradereview/chart/DrawingInput.h"
#include "tradereview/chart/PaneLayout.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QPainter>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>

namespace tradereview::chart {
namespace {

[[nodiscard]] bool finite_candle_row(const data::CandleWindow& window, std::size_t row)
{
    return std::isfinite(window.high[row]) && std::isfinite(window.low[row]);
}

[[nodiscard]] DenseRange normalized_visible_range(DenseRange range)
{
    if (!std::isfinite(range.start_x) || !std::isfinite(range.end_x)) {
        return {0.0, 1.0};
    }
    if (range.end_x < range.start_x) {
        std::swap(range.start_x, range.end_x);
    }
    if (range.end_x <= range.start_x) {
        range.end_x = range.start_x + 1.0;
    }
    return range;
}

[[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> visible_row_bounds(DenseRange range, std::size_t rows)
{
    if (rows == 0) {
        return std::nullopt;
    }

    const auto first_visible_row = std::ceil(range.start_x);
    const auto last_visible_row = std::floor(range.end_x);
    if (last_visible_row < 0.0 || first_visible_row > static_cast<double>(rows - 1)) {
        return std::nullopt;
    }

    const auto first_row = static_cast<std::size_t>(std::max(0.0, first_visible_row));
    const auto last_row = static_cast<std::size_t>(std::min(static_cast<double>(rows - 1), last_visible_row));
    if (last_row < first_row) {
        return std::nullopt;
    }
    return std::pair{first_row, last_row};
}

[[nodiscard]] std::optional<std::pair<double, double>> visible_price_range(
    const data::CandleWindow& window,
    DenseRange visible_dense_range)
{
    if (window.empty() || !window.has_consistent_ohlcv()) {
        return std::nullopt;
    }

    const auto bounds = visible_row_bounds(normalized_visible_range(visible_dense_range), window.row_count());
    if (!bounds.has_value()) {
        return std::nullopt;
    }

    auto min_price = std::numeric_limits<double>::max();
    auto max_price = std::numeric_limits<double>::lowest();
    for (auto row = bounds->first; row <= bounds->second; ++row) {
        if (!finite_candle_row(window, row)) {
            continue;
        }
        min_price = std::min(min_price, window.low[row]);
        max_price = std::max(max_price, window.high[row]);
    }

    if (min_price == std::numeric_limits<double>::max() || max_price == std::numeric_limits<double>::lowest()) {
        return std::nullopt;
    }
    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }
    return std::pair{min_price, max_price};
}

void drawCenteredOverlay(QOpenGLWidget& widget, const QString& text, QColor fill, QColor pen)
{
    QPainter painter(&widget);
    painter.fillRect(widget.rect(), fill);
    painter.setPen(pen);
    painter.drawText(widget.rect(), Qt::AlignCenter, text);
}

} // namespace

ChartViewWidget::ChartViewWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
}

ChartViewWidget::~ChartViewWidget()
{
    release_renderer();
}

void ChartViewWidget::release_renderer()
{
    if (context() == nullptr) {
        return;
    }

    makeCurrent();
    renderer_.release();
    doneCurrent();
}

std::uint64_t ChartViewWidget::bump_generation()
{
    return scene_model_.bump_generation();
}

bool ChartViewWidget::apply_window(data::CandleWindow window)
{
    if (!scene_model_.apply_window(std::move(window))) {
        return false;
    }

    if (scene_model_.window().has_visible_range()) {
        interaction_.reset_for_visible_time_range(scene_model_.index_mapper(), scene_model_.window().visible_range);
    } else {
        interaction_.reset_for_row_count(scene_model_.row_count());
    }
    scene_model_.set_visible_dense_range(interaction_.visible_dense_range());
    has_last_reload_request_ = false;
    update();
    return true;
}

bool ChartViewWidget::set_loading(bool loading)
{
    const auto changed = scene_model_.set_loading(loading);
    if (changed) {
        update();
    }
    return changed;
}

bool ChartViewWidget::loading() const
{
    return scene_model_.loading();
}

bool ChartViewWidget::set_indicator_enabled(const std::string& indicator_name, bool enabled)
{
    if (!scene_model_.set_indicator_enabled(indicator_name, enabled)) {
        return false;
    }
    update();
    if (enabled) {
        request_current_visible_window();
    }
    return true;
}

bool ChartViewWidget::set_bollinger_bands_enabled(bool enabled)
{
    if (!scene_model_.set_bollinger_bands_enabled(enabled)) {
        return false;
    }
    update();
    if (enabled) {
        request_current_visible_window();
    }
    return true;
}

bool ChartViewWidget::set_indicator_panels_enabled(bool enabled)
{
    if (!scene_model_.set_indicator_panels_enabled(enabled)) {
        return false;
    }
    update();
    if (enabled) {
        request_current_visible_window();
    }
    return true;
}

std::vector<std::string> ChartViewWidget::requested_indicators() const
{
    return scene_model_.requested_indicators();
}

void ChartViewWidget::set_reload_request_callback(ReloadRequestCallback callback)
{
    reload_request_callback_ = std::move(callback);
}

void ChartViewWidget::set_crosshair_moved_callback(CrosshairMovedCallback callback)
{
    crosshair_moved_callback_ = std::move(callback);
}

void ChartViewWidget::set_fib_settings(drawing::FibSettings settings)
{
    drawing_state_.set_fib_settings(std::move(settings));
    update();
}

void ChartViewWidget::set_active_drawing_tool(drawing::DrawingType type)
{
    drawing_state_.set_active_tool(type);
    update();
}

void ChartViewWidget::clear_active_drawing_tool()
{
    if (drawing_state_.clear_active_tool()) {
        update();
    }
}

std::optional<drawing::DrawingType> ChartViewWidget::active_drawing_tool() const
{
    return drawing_state_.active_tool();
}

std::optional<drawing::DrawingSpec> ChartViewWidget::add_drawing_point(drawing::DrawingPoint point)
{
    auto completed = drawing_state_.add_point(point);
    update();
    return completed;
}

void ChartViewWidget::clear_drawings()
{
    if (drawing_state_.clear_drawings()) {
        update();
    }
}

bool ChartViewWidget::delete_selected_drawing()
{
    const auto deleted = drawing_state_.delete_selected_drawing();
    if (deleted) {
        update();
    }
    return deleted;
}

bool ChartViewWidget::select_drawing(std::uint64_t drawing_id)
{
    const auto selected = drawing_state_.select_drawing(drawing_id);
    if (selected) {
        update();
    }
    return selected;
}

const std::vector<drawing::DrawingSpec>& ChartViewWidget::drawings() const
{
    return drawing_state_.drawings();
}

std::optional<drawing::DrawingSpec> ChartViewWidget::drawing_preview() const
{
    return drawing_state_.preview();
}

std::optional<std::uint64_t> ChartViewWidget::selected_drawing_id() const
{
    return drawing_state_.selected_drawing_id();
}

void ChartViewWidget::request_current_visible_window()
{
    if (!reload_request_callback_ || scene_model_.index_mapper().empty()) {
        return;
    }

    const auto visible_range = interaction_.visible_time_range(scene_model_.index_mapper());
    if (visible_range.end_ns <= visible_range.start_ns) {
        return;
    }

    reload_request_callback_(visible_range);
    has_last_reload_request_ = true;
    last_reload_request_ = visible_range;
}

std::optional<double> ChartViewWidget::dense_x_for_timestamp(std::int64_t timestamp_ns) const
{
    if (scene_model_.index_mapper().empty()) {
        return std::nullopt;
    }
    return scene_model_.index_mapper().dense_x_from_timestamp(timestamp_ns);
}

void ChartViewWidget::sync_crosshair(std::int64_t timestamp_ns, double price, double dense_x)
{
    if (!std::isfinite(price) || !std::isfinite(dense_x)) {
        return;
    }
    crosshair_state_ = ChartCrosshairState{timestamp_ns, price, dense_x};
    update();
}

bool ChartViewWidget::sync_center_on_timestamp(std::int64_t timestamp_ns, std::optional<double> price)
{
    const auto dense_x = dense_x_for_timestamp(timestamp_ns);
    if (!dense_x.has_value()) {
        return false;
    }

    const auto centered = interaction_.center_on_dense_x(*dense_x);
    if (centered) {
        apply_interaction_update();
    }

    auto y_centered = false;
    if (price.has_value()) {
        y_centered = sync_y_center(*price);
    }
    return centered || y_centered;
}

bool ChartViewWidget::sync_y_center(double price)
{
    if (!std::isfinite(price)) {
        return false;
    }
    if (synced_y_center_price_.has_value() && *synced_y_center_price_ == price) {
        return false;
    }
    synced_y_center_price_ = price;
    update();
    return true;
}

std::optional<ChartCrosshairState> ChartViewWidget::crosshair_state() const
{
    return crosshair_state_;
}

std::optional<double> ChartViewWidget::synced_y_center_price() const
{
    return synced_y_center_price_;
}

const ChartSceneModel& ChartViewWidget::scene_model() const
{
    return scene_model_;
}

void ChartViewWidget::initializeGL()
{
    QObject::connect(
        context(),
        &QOpenGLContext::aboutToBeDestroyed,
        this,
        &ChartViewWidget::release_renderer,
        Qt::DirectConnection);
    renderer_.initialize();
}

void ChartViewWidget::resizeGL(int width, int height)
{
    renderer_.resize(width, height);
}

void ChartViewWidget::paintGL()
{
    renderer_.render(scene_model_, drawing_state_.drawings(), drawing_state_.preview(), drawing_state_.revision());

    if (scene_model_.row_count() == 0 && !scene_model_.loading()) {
        drawCenteredOverlay(*this, "No dataset loaded", app::theme::chartBackground(), app::theme::chartEmptyText());
        return;
    }

    if (!scene_model_.loading()) {
        return;
    }

    drawCenteredOverlay(*this, "Loading...", app::theme::loadingOverlay(), QColor(235, 238, 245));
}

void ChartViewWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        setFocus(Qt::MouseFocusReason);
        if (drawing_state_.active_tool().has_value()) {
            const auto point = drawing_point_from_position(event->position());
            if (point.has_value()) {
                (void)add_drawing_point(*point);
            }
            event->accept();
            return;
        }

        is_panning_ = true;
        last_mouse_position_ = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }

    QOpenGLWidget::mousePressEvent(event);
}

void ChartViewWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (is_panning_) {
        const auto current_position = event->position();
        interaction_.pan_by_pixels(current_position.x() - last_mouse_position_.x(), width());
        last_mouse_position_ = current_position;
        apply_interaction_update();
        event->accept();
        return;
    }

    const auto crosshair = crosshair_from_position(event->position());
    if (crosshair.has_value()) {
        crosshair_state_ = *crosshair;
        if (crosshair_moved_callback_) {
            crosshair_moved_callback_(crosshair->timestamp_ns, crosshair->price);
        }
        update();
    }

    if (drawing_state_.active_tool().has_value()) {
        const auto point = drawing_point_from_position(event->position());
        if (drawing_state_.update_preview(point)) {
            update();
        }
    }

    QOpenGLWidget::mouseMoveEvent(event);
}

void ChartViewWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && is_panning_) {
        is_panning_ = false;
        unsetCursor();
        event->accept();
        return;
    }

    QOpenGLWidget::mouseReleaseEvent(event);
}

void ChartViewWidget::wheelEvent(QWheelEvent* event)
{
    const auto scale = interaction_.zoom_scale_from_wheel_delta(event->angleDelta().y(), event->pixelDelta().y());
    if (scale == 1.0) {
        QOpenGLWidget::wheelEvent(event);
        return;
    }

    interaction_.zoom_at_pixel(event->position().x(), width(), scale);
    apply_interaction_update();
    event->accept();
}

void ChartViewWidget::keyPressEvent(QKeyEvent* event)
{
    if ((event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) && delete_selected_drawing()) {
        event->accept();
        return;
    }

    QOpenGLWidget::keyPressEvent(event);
}

void ChartViewWidget::apply_interaction_update()
{
    const auto range_changed = scene_model_.set_visible_dense_range(interaction_.visible_dense_range());
    const auto decision = interaction_.reload_decision(scene_model_.index_mapper(), scene_model_.window().loaded_range);
    if (decision.requested) {
        const auto repeated_request = has_last_reload_request_
            && last_reload_request_.start_ns == decision.visible_range.start_ns
            && last_reload_request_.end_ns == decision.visible_range.end_ns;
        if (!repeated_request && reload_request_callback_) {
            reload_request_callback_(decision.visible_range);
        }
        has_last_reload_request_ = true;
        last_reload_request_ = decision.visible_range;
    } else {
        has_last_reload_request_ = false;
    }

    if (range_changed) {
        update();
    }
}

std::optional<ChartCrosshairState> ChartViewWidget::crosshair_from_position(QPointF position) const
{
    if (scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return std::nullopt;
    }

    const auto price = price_at_pixel_y(position.y());
    if (!price.has_value()) {
        return std::nullopt;
    }

    const auto dense_x = dense_x_at_pixel(position.x());
    return ChartCrosshairState{
        scene_model_.index_mapper().timestamp_from_x(dense_x),
        *price,
        dense_x,
    };
}

std::optional<drawing::DrawingPoint> ChartViewWidget::drawing_point_from_position(QPointF position) const
{
    if (scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return std::nullopt;
    }

    const auto price_range = visible_price_range(scene_model_.window(), scene_model_.visible_dense_range());
    if (!price_range.has_value()) {
        return std::nullopt;
    }

    return drawing_point_from_widget_position(
        scene_model_.index_mapper(),
        scene_model_.visible_dense_range(),
        build_pane_layout(scene_model_.indicator_panels_enabled()).price,
        width(),
        height(),
        WidgetPosition{position.x(), position.y()},
        price_range->first,
        price_range->second);
}

double ChartViewWidget::dense_x_at_pixel(double pixel_x) const
{
    const auto range = scene_model_.visible_dense_range();
    const auto span = std::max(range.span(), 1.0);
    const auto fraction = std::clamp(pixel_x / static_cast<double>(std::max(width(), 1)), 0.0, 1.0);
    return range.start_x + (span * fraction);
}

std::optional<double> ChartViewWidget::price_at_pixel_y(double pixel_y) const
{
    const auto price_range = visible_price_range(scene_model_.window(), scene_model_.visible_dense_range());
    if (!price_range.has_value() || height() <= 0) {
        return std::nullopt;
    }

    const auto pane = build_pane_layout(scene_model_.indicator_panels_enabled()).price;
    const auto normalized_device_y = 1.0 - (2.0 * (pixel_y / static_cast<double>(height())));
    const auto pane_fraction =
        (normalized_device_y - static_cast<double>(pane.bottom)) / static_cast<double>(std::max(pane.height(), 0.000001F));
    return price_range->first + ((price_range->second - price_range->first) * pane_fraction);
}

} // namespace tradereview::chart
