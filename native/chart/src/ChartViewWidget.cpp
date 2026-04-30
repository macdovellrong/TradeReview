#include "tradereview/chart/ChartViewWidget.h"

#include "tradereview/chart/ChartOverlayGeometry.h"
#include "tradereview/chart/ChartTimeFormat.h"
#include "tradereview/chart/DrawingInput.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/PriceRange.h"
#include "tradereview/drawing/FibMath.h"

#include <QColor>
#include <QFont>
#include <QFontMetrics>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QObject>
#include <QOpenGLContext>
#include <QPainter>
#include <QPen>
#include <QRectF>
#include <QString>
#include <QTimer>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tradereview::chart {
namespace {

constexpr int kPanReloadThrottleMs = 200;
constexpr double kPriceAxisHitWidth = 96.0;

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

[[nodiscard]] QColor chartOverlayBackground()
{
    return QColor(11, 16, 23);
}

[[nodiscard]] QColor chartOverlayText()
{
    return QColor(143, 160, 184);
}

[[nodiscard]] QColor loadingOverlay()
{
    return QColor(15, 18, 24, 132);
}

[[nodiscard]] QColor overlayLineColor()
{
    return QColor(196, 208, 224, 150);
}

[[nodiscard]] QColor overlaySubtleLineColor()
{
    return QColor(82, 96, 118, 150);
}

[[nodiscard]] QColor overlayBoxFill()
{
    return QColor(10, 14, 21, 220);
}

[[nodiscard]] QColor fibLabelColor()
{
    return QColor(244, 204, 96);
}

[[nodiscard]] const std::string& axis_label_period(const data::CandleWindow& window)
{
    if (!window.actual_period.empty()) {
        return window.actual_period;
    }
    return window.requested_period;
}

[[nodiscard]] QString format_price(double price)
{
    const auto abs_price = std::abs(price);
    const auto decimals = abs_price >= 100.0 ? 2 : 4;
    return QString::number(price, 'f', decimals);
}

void draw_boxed_text(
    QPainter& painter,
    QRectF rect,
    const QString& text,
    QColor text_color,
    Qt::Alignment alignment = Qt::AlignCenter)
{
    painter.save();
    painter.setPen(QPen(overlaySubtleLineColor(), 1.0));
    painter.setBrush(overlayBoxFill());
    painter.drawRoundedRect(rect, 4.0, 4.0);
    painter.setPen(text_color);
    painter.drawText(rect, alignment, text);
    painter.restore();
}

void draw_right_axis_label(
    QPainter& painter,
    const QString& text,
    double y,
    int widget_width,
    double top,
    double bottom,
    QColor text_color)
{
    if (widget_width <= 0 || bottom <= top || text.isEmpty()) {
        return;
    }

    const QFontMetrics metrics(painter.font());
    const auto label_width = static_cast<double>(metrics.horizontalAdvance(text) + 12);
    const auto label_height = static_cast<double>(metrics.height() + 4);
    const auto x = std::max(4.0, static_cast<double>(widget_width) - label_width - 6.0);
    const auto y_top = std::clamp(
        y - (label_height * 0.5),
        top + 3.0,
        std::max(top + 3.0, bottom - label_height - 3.0));

    draw_boxed_text(painter, QRectF{x, y_top, label_width, label_height}, text, text_color);
}

void draw_label_at(
    QPainter& painter,
    const QString& text,
    QPointF anchor,
    int widget_width,
    int widget_height,
    QColor text_color,
    bool center_x)
{
    const QFontMetrics metrics(painter.font());
    const auto width = static_cast<double>(metrics.horizontalAdvance(text) + 12);
    const auto height = static_cast<double>(metrics.height() + 4);
    auto x = center_x ? anchor.x() - (width * 0.5) : anchor.x();
    auto y = anchor.y() - (height * 0.5);
    x = std::clamp(x, 4.0, std::max(4.0, static_cast<double>(widget_width) - width - 4.0));
    y = std::clamp(y, 4.0, std::max(4.0, static_cast<double>(widget_height) - height - 4.0));
    draw_boxed_text(painter, QRectF{x, y, width, height}, text, text_color);
}

[[nodiscard]] std::optional<std::pair<double, double>> pane_pixel_bounds(PaneRect pane, int widget_height)
{
    if (widget_height <= 0) {
        return std::nullopt;
    }
    auto top = widget_y_for_normalized_device_y(pane.top, widget_height);
    auto bottom = widget_y_for_normalized_device_y(pane.bottom, widget_height);
    if (bottom < top) {
        std::swap(top, bottom);
    }
    return std::pair{top, bottom};
}

[[nodiscard]] std::vector<double> fib_levels_for_spec(const drawing::DrawingSpec& spec)
{
    if (spec.fib_snapshot.has_value()) {
        return spec.fib_snapshot->levels;
    }
    return {};
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

    pan_reload_timer_ = new QTimer(this);
    pan_reload_timer_->setSingleShot(true);
    pan_reload_timer_->setInterval(kPanReloadThrottleMs);
    connect(pan_reload_timer_, &QTimer::timeout, this, [this]() {
        if (is_panning_) {
            apply_interaction_update();
        }
    });
}

ChartViewWidget::~ChartViewWidget()
{
    if (pan_reload_timer_ != nullptr) {
        pan_reload_timer_->stop();
    }
    if (auto* gl_context = context(); gl_context != nullptr) {
        QObject::disconnect(gl_context, &QOpenGLContext::aboutToBeDestroyed, this, &ChartViewWidget::release_renderer);
    }
    release_renderer();
}

void ChartViewWidget::release_renderer()
{
    if (!renderer_context_ready_) {
        return;
    }

    auto* gl_context = context();
    if (gl_context == nullptr || !gl_context->isValid() || !isValid()) {
        renderer_context_ready_ = false;
        return;
    }
    makeCurrent();
    renderer_.release();
    doneCurrent();
    renderer_context_ready_ = false;
}

std::uint64_t ChartViewWidget::bump_generation()
{
    return scene_model_.bump_generation();
}

bool ChartViewWidget::apply_window(data::CandleWindow window)
{
    std::optional<core::TimeRange> panning_visible_range;
    if (is_panning_ && !scene_model_.index_mapper().empty()) {
        panning_visible_range = interaction_.visible_time_range(scene_model_.index_mapper());
    }

    if (!scene_model_.apply_window(std::move(window))) {
        return false;
    }

    if (panning_visible_range.has_value()) {
        interaction_.reset_for_visible_time_range(scene_model_.index_mapper(), *panning_visible_range);
    } else if (scene_model_.window().has_visible_range()) {
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

bool ChartViewWidget::fit_price_axis_to_visible_range()
{
    const auto price_range = visible_price_range(scene_model_.window(), scene_model_.visible_dense_range());
    if (!price_range.has_value()) {
        return false;
    }
    if (!scene_model_.set_price_range_override(*price_range)) {
        return false;
    }
    update();
    return true;
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
    renderer_context_ready_ = true;
}

void ChartViewWidget::resizeGL(int width, int height)
{
    renderer_.resize(width, height);
}

void ChartViewWidget::paintGL()
{
    renderer_.render(scene_model_, drawing_state_.drawings(), drawing_state_.preview(), drawing_state_.revision());

    if (scene_model_.row_count() == 0 && !scene_model_.loading()) {
        drawCenteredOverlay(*this, "No dataset loaded", chartOverlayBackground(), chartOverlayText());
        return;
    }

    if (scene_model_.row_count() > 0) {
        QPainter painter(this);
        draw_chart_overlays(painter);
    }

    if (!scene_model_.loading()) {
        return;
    }

    drawCenteredOverlay(*this, "Loading...", loadingOverlay(), QColor(235, 238, 245));
}

void ChartViewWidget::draw_chart_overlays(QPainter& painter) const
{
    auto font = painter.font();
    font.setPointSize(9);
    painter.setFont(font);
    painter.setRenderHint(QPainter::Antialiasing, true);

    draw_time_axis(painter);
    draw_value_axis(painter);
    draw_fib_labels(painter);
    draw_crosshair(painter);
}

void ChartViewWidget::draw_time_axis(QPainter& painter) const
{
    if (scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return;
    }

    const auto range = normalized_visible_range(scene_model_.visible_dense_range());
    const auto period = axis_label_period(scene_model_.window());
    const auto label_count = std::clamp(width() / 160, 3, 6);
    const auto axis_top = static_cast<double>(height() - 24);

    painter.save();
    painter.fillRect(QRectF{0.0, axis_top, static_cast<double>(width()), 24.0}, QColor(5, 9, 14, 170));
    painter.setPen(QPen(overlaySubtleLineColor(), 1.0));
    painter.drawLine(QPointF{0.0, axis_top}, QPointF{static_cast<double>(width()), axis_top});

    const QFontMetrics metrics(painter.font());
    painter.setPen(chartOverlayText());
    for (int index = 0; index < label_count; ++index) {
        const auto fraction = label_count == 1 ? 0.0 : static_cast<double>(index) / static_cast<double>(label_count - 1);
        const auto dense_x = range.start_x + (range.span() * fraction);
        auto x = widget_x_for_dense_x(range, width(), dense_x);
        x = std::clamp(x, 2.0, std::max(2.0, static_cast<double>(width()) - 2.0));

        painter.setPen(QPen(overlaySubtleLineColor(), 1.0));
        painter.drawLine(QPointF{x, axis_top}, QPointF{x, axis_top + 4.0});

        const auto label =
            format_axis_timestamp_label(scene_model_.index_mapper().timestamp_from_x(dense_x), period);
        const auto text_width = static_cast<double>(metrics.horizontalAdvance(label));
        auto text_x = x - (text_width * 0.5);
        text_x = std::clamp(text_x, 4.0, std::max(4.0, static_cast<double>(width()) - text_width - 4.0));
        painter.setPen(chartOverlayText());
        painter.drawText(QPointF{text_x, static_cast<double>(height() - 7)}, label);
    }
    painter.restore();
}

void ChartViewWidget::draw_value_axis(QPainter& painter) const
{
    if (scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return;
    }

    const auto price_range = current_price_range();
    if (!price_range.has_value()) {
        return;
    }

    const auto layout = build_pane_layout(scene_model_.indicator_panels_enabled());
    const auto price_bounds = pane_pixel_bounds(layout.price, height());
    if (!price_bounds.has_value()) {
        return;
    }

    constexpr double kAxisFractions[] = {0.0, 0.25, 0.5, 0.75, 1.0};
    const auto axis_bottom = std::min(price_bounds->second, static_cast<double>(height() - 28));
    if (axis_bottom <= price_bounds->first) {
        return;
    }

    painter.save();
    painter.setPen(QPen(overlaySubtleLineColor(), 1.0));
    for (const auto fraction : kAxisFractions) {
        const auto price = price_range->first + ((price_range->second - price_range->first) * fraction);
        const auto y = widget_y_for_price(layout.price, height(), price_range->first, price_range->second, price);
        if (!y.has_value()) {
            continue;
        }
        painter.drawLine(
            QPointF{static_cast<double>(width() - 6), *y},
            QPointF{static_cast<double>(width()), *y});
        draw_right_axis_label(
            painter,
            format_price(price),
            *y,
            width(),
            price_bounds->first,
            axis_bottom,
            QColor(235, 238, 245));
    }
    painter.restore();
}

void ChartViewWidget::draw_crosshair(QPainter& painter) const
{
    if (!crosshair_state_.has_value() || scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return;
    }

    const auto price_range = current_price_range();
    if (!price_range.has_value()) {
        return;
    }

    const auto layout = build_pane_layout(scene_model_.indicator_panels_enabled());
    const auto x = widget_x_for_dense_x(scene_model_.visible_dense_range(), width(), crosshair_state_->dense_x);
    const auto y = widget_y_for_price(layout.price, height(), price_range->first, price_range->second, crosshair_state_->price);
    if (x < 0.0 || x > static_cast<double>(width()) || !y.has_value()) {
        return;
    }

    painter.save();
    painter.setPen(QPen(overlayLineColor(), 1.0, Qt::DashLine));
    painter.drawLine(QPointF{x, 0.0}, QPointF{x, static_cast<double>(height())});
    if (*y >= 0.0 && *y <= static_cast<double>(height())) {
        painter.drawLine(QPointF{0.0, *y}, QPointF{static_cast<double>(width()), *y});
    }

    draw_label_at(
        painter,
        format_axis_timestamp_label(crosshair_state_->timestamp_ns, axis_label_period(scene_model_.window())),
        QPointF{x, static_cast<double>(height() - 12)},
        width(),
        height(),
        QColor(235, 238, 245),
        true);

    if (*y >= 0.0 && *y <= static_cast<double>(height())) {
        draw_label_at(
            painter,
            format_price(crosshair_state_->price),
            QPointF{static_cast<double>(width() - 74), *y},
            width(),
            height(),
            QColor(235, 238, 245),
            false);
    }
    painter.restore();
}

void ChartViewWidget::draw_fib_labels(QPainter& painter) const
{
    if (scene_model_.index_mapper().empty() || width() <= 0 || height() <= 0) {
        return;
    }

    const auto price_range = current_price_range();
    if (!price_range.has_value()) {
        return;
    }

    const auto layout = build_pane_layout(scene_model_.indicator_panels_enabled());
    const auto price_bounds = pane_pixel_bounds(layout.price, height());
    if (!price_bounds.has_value()) {
        return;
    }

    const auto draw_spec_labels = [&](const drawing::DrawingSpec& spec) {
        const auto levels = fib_levels_for_spec(spec);
        if (levels.empty()) {
            return;
        }

        if (spec.type == drawing::DrawingType::FibRetracement && spec.points.size() >= 2) {
            const auto x1 = scene_model_.index_mapper().dense_x_from_timestamp(spec.points[0].timestamp_ns);
            const auto x2 = scene_model_.index_mapper().dense_x_from_timestamp(spec.points[1].timestamp_ns);
            const auto label_x =
                std::max(
                    widget_x_for_dense_x(scene_model_.visible_dense_range(), width(), x1),
                    widget_x_for_dense_x(scene_model_.visible_dense_range(), width(), x2))
                + 8.0;

            for (const auto& level : drawing::build_retracement_levels(spec.points[0].price, spec.points[1].price, levels)) {
                const auto y = widget_y_for_price(layout.price, height(), price_range->first, price_range->second, level.price);
                if (!y.has_value() || *y < price_bounds->first || *y > price_bounds->second) {
                    continue;
                }
                draw_label_at(
                    painter,
                    QString::fromStdString(format_fib_ratio_label(level.ratio)),
                    QPointF{label_x, *y},
                    width(),
                    height(),
                    fibLabelColor(),
                    false);
            }
            return;
        }

        if (spec.type == drawing::DrawingType::FibExtension && spec.points.size() >= 3) {
            const auto x_a = scene_model_.index_mapper().dense_x_from_timestamp(spec.points[0].timestamp_ns);
            const auto x_b = scene_model_.index_mapper().dense_x_from_timestamp(spec.points[1].timestamp_ns);
            const auto x_c = scene_model_.index_mapper().dense_x_from_timestamp(spec.points[2].timestamp_ns);
            const auto projection_span = std::max(std::abs(x_b - x_a), 1.0);
            const auto label_x = widget_x_for_dense_x(scene_model_.visible_dense_range(), width(), x_c + projection_span) + 8.0;

            for (const auto& level : drawing::build_extension_levels(
                     spec.points[0].price,
                     spec.points[1].price,
                     spec.points[2].price,
                     levels)) {
                const auto y = widget_y_for_price(layout.price, height(), price_range->first, price_range->second, level.price);
                if (!y.has_value() || *y < price_bounds->first || *y > price_bounds->second) {
                    continue;
                }
                draw_label_at(
                    painter,
                    QString::fromStdString(format_fib_ratio_label(level.ratio)),
                    QPointF{label_x, *y},
                    width(),
                    height(),
                    fibLabelColor(),
                    false);
            }
        }
    };

    for (const auto& spec : drawing_state_.drawings()) {
        draw_spec_labels(spec);
    }
    if (const auto preview = drawing_state_.preview(); preview.has_value()) {
        draw_spec_labels(*preview);
    }
}

void ChartViewWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        setFocus(Qt::MouseFocusReason);
        if (price_axis_hit(event->position())) {
            event->accept();
            return;
        }

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
        const auto delta_x = current_position.x() - last_mouse_position_.x();
        const auto delta_y = current_position.y() - last_mouse_position_.y();
        interaction_.pan_by_pixels(delta_x, width());
        last_mouse_position_ = current_position;
        (void)pan_price_axis_by_pixels(delta_y);
        apply_interaction_update(false);
        schedule_pan_reload();
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
        if (pan_reload_timer_ != nullptr) {
            pan_reload_timer_->stop();
        }
        unsetCursor();
        apply_interaction_update();
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

    if (price_axis_hit(event->position()) && zoom_price_axis_at(event->position(), scale)) {
        event->accept();
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

void ChartViewWidget::apply_interaction_update(bool allow_reload)
{
    const auto range_changed = scene_model_.set_visible_dense_range(interaction_.visible_dense_range());
    if (allow_reload) {
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
    }

    if (range_changed) {
        update();
    }
}

void ChartViewWidget::schedule_pan_reload()
{
    if (pan_reload_timer_ != nullptr && !pan_reload_timer_->isActive()) {
        pan_reload_timer_->start();
    }
}

std::optional<PriceRange> ChartViewWidget::current_price_range() const
{
    if (const auto override_range = scene_model_.price_range_override(); override_range.has_value()) {
        return override_range;
    }
    return visible_price_range(scene_model_.window(), scene_model_.visible_dense_range());
}

bool ChartViewWidget::price_axis_hit(QPointF position) const
{
    if (width() <= 0 || height() <= 0 || position.x() < static_cast<double>(width()) - kPriceAxisHitWidth) {
        return false;
    }

    const auto layout = build_pane_layout(scene_model_.indicator_panels_enabled());
    const auto price_bounds = pane_pixel_bounds(layout.price, height());
    if (!price_bounds.has_value()) {
        return false;
    }

    const auto axis_bottom = std::min(price_bounds->second, static_cast<double>(height() - 28));
    return position.y() >= price_bounds->first && position.y() <= axis_bottom;
}

bool ChartViewWidget::zoom_price_axis_at(QPointF position, double scale_factor)
{
    const auto price_range = current_price_range();
    const auto anchor_price = price_at_pixel_y(position.y());
    if (!price_range.has_value() || !anchor_price.has_value()) {
        return false;
    }

    const auto next_range = zoom_price_range(*price_range, *anchor_price, scale_factor);
    if (!scene_model_.set_price_range_override(next_range)) {
        return false;
    }
    update();
    return true;
}

bool ChartViewWidget::pan_price_axis_by_pixels(double pixel_delta_y)
{
    if (!std::isfinite(pixel_delta_y) || pixel_delta_y == 0.0) {
        return false;
    }

    const auto price_range = current_price_range();
    if (!price_range.has_value()) {
        return false;
    }

    const auto layout = build_pane_layout(scene_model_.indicator_panels_enabled());
    const auto price_bounds = pane_pixel_bounds(layout.price, height());
    if (!price_bounds.has_value()) {
        return false;
    }

    const auto pane_height = std::max(std::abs(price_bounds->second - price_bounds->first), 1.0);
    const auto price_span = price_range->second - price_range->first;
    if (!std::isfinite(price_span) || price_span <= 0.0) {
        return false;
    }

    const auto price_delta = price_delta_for_pixel_pan(*price_range, pixel_delta_y, pane_height);
    const auto next_range = pan_price_range(*price_range, price_delta);
    if (!scene_model_.set_price_range_override(next_range)) {
        return false;
    }
    update();
    return true;
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

    const auto price_range = current_price_range();
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
    const auto price_range = current_price_range();
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
