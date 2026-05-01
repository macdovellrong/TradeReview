#include "tradereview/chart/ChartWorkspaceWidget.h"

#include "tradereview/chart/ChartPanelWidget.h"
#include "tradereview/chart/ChartPeriod.h"
#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/FloatingChartWindow.h"

#include <QGridLayout>
#include <QHBoxLayout>
#include <QString>
#include <QTabWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tradereview::chart {
namespace {

constexpr int kDefaultPixelWidth = 1200;

QString chart_label(std::uint64_t chart_id)
{
    return QString("Chart %1").arg(static_cast<qulonglong>(chart_id));
}

} // namespace

ChartWorkspaceWidget::ChartWorkspaceWidget(QWidget* parent)
    : QWidget(parent)
{
    root_layout_ = new QVBoxLayout(this);
    root_layout_->setContentsMargins(0, 0, 0, 0);
    root_layout_->setSpacing(0);

    panels_.reserve(4);
    for (std::uint64_t chart_id = 1; chart_id <= 4; ++chart_id) {
        auto* panel_widget = new ChartPanelWidget(chart_id, this);
        panel_widget->set_requested_period(state_.chart_period(chart_id));
        connect_panel(*panel_widget);
        panels_.push_back(panel_widget);
    }

    rebuild_layout();
}

ChartWorkspaceWidget::~ChartWorkspaceWidget()
{
    for (auto& detached : detached_windows_) {
        if (detached.window == nullptr) {
            continue;
        }
        detached.window->setCloseCallback({});
        if (auto* target_panel = detached.window->take_panel(); target_panel != nullptr) {
            target_panel->setParent(this);
            target_panel->hide();
        }
        delete detached.window;
        detached.window = nullptr;
    }
    detached_windows_.clear();
}

void ChartWorkspaceWidget::setStatusCallback(StatusCallback callback)
{
    status_callback_ = std::move(callback);
    for (auto* panel_widget : panels_) {
        panel_widget->setStatusCallback(status_callback_);
    }
}

void ChartWorkspaceWidget::setReloadRequestCallback(ReloadRequestCallback callback)
{
    reload_request_callback_ = std::move(callback);
}

bool ChartWorkspaceWidget::setChartCount(int count)
{
    if (!state_.set_chart_count(count)) {
        return false;
    }
    reattach_disabled_detached_charts();
    refresh_sync_enabled_charts();
    rebuild_layout();
    return true;
}

bool ChartWorkspaceWidget::setLayoutMode(ChartLayoutMode mode)
{
    if (!state_.set_layout_mode(mode)) {
        return false;
    }
    rebuild_layout();
    return true;
}

bool ChartWorkspaceWidget::setRequestedPeriod(std::uint64_t chart_id, std::string period)
{
    auto* target_panel = panel(chart_id);
    if (target_panel == nullptr || period.empty()) {
        return false;
    }

    auto canonical_period = canonical_chart_period(period);
    const auto changed = state_.set_chart_period(chart_id, canonical_period);
    target_panel->set_requested_period(std::move(canonical_period));
    if (auto* window = floating_window(chart_id); window != nullptr) {
        window->refresh_title();
    }
    return changed;
}

bool ChartWorkspaceWidget::setChartLoading(std::uint64_t chart_id, bool loading)
{
    auto* target_panel = panel(chart_id);
    if (target_panel == nullptr) {
        return false;
    }
    return target_panel->set_loading(loading);
}

bool ChartWorkspaceWidget::apply_window(data::CandleWindow window)
{
    if (!state_.chart_enabled(window.chart_id)) {
        return false;
    }
    auto* target_panel = panel(window.chart_id);
    if (target_panel == nullptr) {
        return false;
    }
    state_.set_active_chart_id(window.chart_id);
    return target_panel->apply_window(std::move(window));
}

bool ChartWorkspaceWidget::triggerDrawingAction(const QString& action)
{
    auto* active_panel = panel(active_chart_id());
    if (active_panel == nullptr) {
        return false;
    }
    active_panel->trigger_drawing_action(action);
    return true;
}

ChartViewWidget& ChartWorkspaceWidget::chart_view()
{
    return chart_view(state_.active_chart_id());
}

const ChartViewWidget& ChartWorkspaceWidget::chart_view() const
{
    return chart_view(state_.active_chart_id());
}

ChartViewWidget& ChartWorkspaceWidget::chart_view(std::uint64_t chart_id)
{
    auto* target_panel = panel(chart_id);
    if (target_panel == nullptr) {
        throw std::out_of_range("unknown chart id");
    }
    return target_panel->chart_view();
}

const ChartViewWidget& ChartWorkspaceWidget::chart_view(std::uint64_t chart_id) const
{
    const auto* target_panel = panel(chart_id);
    if (target_panel == nullptr) {
        throw std::out_of_range("unknown chart id");
    }
    return target_panel->chart_view();
}

ChartPanelWidget* ChartWorkspaceWidget::panel(std::uint64_t chart_id)
{
    const auto found = std::find_if(panels_.begin(), panels_.end(), [chart_id](const ChartPanelWidget* panel_widget) {
        return panel_widget->chart_id() == chart_id;
    });
    if (found == panels_.end()) {
        return nullptr;
    }
    return *found;
}

const ChartPanelWidget* ChartWorkspaceWidget::panel(std::uint64_t chart_id) const
{
    const auto found = std::find_if(panels_.begin(), panels_.end(), [chart_id](const ChartPanelWidget* panel_widget) {
        return panel_widget->chart_id() == chart_id;
    });
    if (found == panels_.end()) {
        return nullptr;
    }
    return *found;
}

std::vector<std::string> ChartWorkspaceWidget::requested_indicators() const
{
    return requested_indicators(state_.active_chart_id());
}

std::vector<std::string> ChartWorkspaceWidget::requested_indicators(std::uint64_t chart_id) const
{
    const auto* target_panel = panel(chart_id);
    if (target_panel == nullptr) {
        return {};
    }
    return target_panel->requested_indicators();
}

std::string ChartWorkspaceWidget::requested_period(std::uint64_t chart_id) const
{
    return state_.chart_period(chart_id);
}

int ChartWorkspaceWidget::chart_pixel_width(std::uint64_t chart_id) const
{
    const auto* target_panel = panel(chart_id);
    if (target_panel == nullptr) {
        return kDefaultPixelWidth;
    }
    return std::max(target_panel->chart_view().width(), kDefaultPixelWidth);
}

std::uint64_t ChartWorkspaceWidget::active_chart_id() const
{
    return state_.active_chart_id();
}

std::vector<std::uint64_t> ChartWorkspaceWidget::enabled_chart_ids() const
{
    return state_.enabled_chart_ids();
}

std::size_t ChartWorkspaceWidget::chart_count() const
{
    return state_.chart_count();
}

ChartLayoutMode ChartWorkspaceWidget::layout_mode() const
{
    return state_.layout_mode();
}

bool ChartWorkspaceWidget::detachChart(std::uint64_t chart_id)
{
    if (auto* existing = floating_window(chart_id); existing != nullptr) {
        existing->raise();
        existing->activateWindow();
        return false;
    }

    auto* target_panel = panel(chart_id);
    if (target_panel == nullptr || !state_.detach_chart(chart_id)) {
        return false;
    }

    auto* window = new FloatingChartWindow(target_panel, this);
    window->setCloseCallback([this](std::uint64_t detached_chart_id) {
        reattachChart(detached_chart_id);
    });
    detached_windows_.push_back(DetachedChartWindow{chart_id, window});
    window->show();
    rebuild_layout();
    return true;
}

bool ChartWorkspaceWidget::reattachChart(std::uint64_t chart_id)
{
    const auto found = std::find_if(
        detached_windows_.begin(),
        detached_windows_.end(),
        [chart_id](const DetachedChartWindow& detached) {
            return detached.chart_id == chart_id;
        });
    if (found == detached_windows_.end()) {
        return false;
    }

    auto* window = found->window;
    ChartPanelWidget* target_panel = nullptr;
    if (window != nullptr) {
        window->setCloseCallback({});
        target_panel = window->take_panel();
        window->hide();
        window->deleteLater();
    }
    detached_windows_.erase(found);
    state_.reattach_chart(chart_id);
    if (state_.chart_enabled(chart_id)) {
        state_.set_active_chart_id(chart_id);
    }
    if (target_panel != nullptr) {
        target_panel->setParent(this);
        target_panel->show();
    }
    rebuild_layout();
    return true;
}

bool ChartWorkspaceWidget::syncCrosshairFrom(std::uint64_t source_chart_id, std::int64_t timestamp_ns, double price)
{
    return crosshair_sync_controller_.sync_crosshair_from(source_chart_id, timestamp_ns, price);
}

bool ChartWorkspaceWidget::syncCenterFrom(
    std::uint64_t source_chart_id,
    std::int64_t timestamp_ns,
    std::optional<double> price)
{
    return crosshair_sync_controller_.sync_center_from(source_chart_id, timestamp_ns, price);
}

bool ChartWorkspaceWidget::syncYCenterFrom(std::uint64_t source_chart_id, double price)
{
    return crosshair_sync_controller_.sync_y_center_from(source_chart_id, price);
}

void ChartWorkspaceWidget::rebuild_layout()
{
    reset_content_widget();
    const auto ids = state_.visible_chart_ids();

    if (state_.layout_mode() == ChartLayoutMode::Tabs) {
        auto* layout = new QVBoxLayout(content_);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        auto* tabs = new QTabWidget(content_);
        layout->addWidget(tabs, 1);
        for (const auto chart_id : ids) {
            auto* target_panel = panel(chart_id);
            target_panel->show();
            tabs->addTab(target_panel, chart_label(chart_id));
        }
        const auto active = state_.active_chart_id();
        for (int index = 0; index < tabs->count(); ++index) {
            if (ids[static_cast<std::size_t>(index)] == active) {
                tabs->setCurrentIndex(index);
                break;
            }
        }
        connect(tabs, &QTabWidget::currentChanged, this, [this, ids](int index) {
            if (index >= 0 && static_cast<std::size_t>(index) < ids.size()) {
                state_.set_active_chart_id(ids[static_cast<std::size_t>(index)]);
            }
        });
        return;
    }

    if (state_.layout_mode() == ChartLayoutMode::Grid2x2) {
        auto* layout = new QGridLayout(content_);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(1);
        for (std::size_t index = 0; index < ids.size(); ++index) {
            auto* target_panel = panel(ids[index]);
            target_panel->show();
            layout->addWidget(target_panel, static_cast<int>(index / 2), static_cast<int>(index % 2));
        }
        return;
    }

    if (state_.layout_mode() == ChartLayoutMode::DualVertical) {
        auto* layout = new QHBoxLayout(content_);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(1);
        auto* left = new QVBoxLayout();
        auto* right = new QVBoxLayout();
        left->setContentsMargins(0, 0, 0, 0);
        right->setContentsMargins(0, 0, 0, 0);
        left->setSpacing(1);
        right->setSpacing(1);
        layout->addLayout(left, 1);
        layout->addLayout(right, 1);

        const auto split = (ids.size() + 1) / 2;
        for (std::size_t index = 0; index < ids.size(); ++index) {
            auto* target_panel = panel(ids[index]);
            target_panel->show();
            (index < split ? left : right)->addWidget(target_panel, 1);
        }
        return;
    }

    auto* layout = new QVBoxLayout(content_);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(1);
    for (const auto chart_id : ids) {
        auto* target_panel = panel(chart_id);
        target_panel->show();
        layout->addWidget(target_panel, 1);
    }
}

void ChartWorkspaceWidget::reset_content_widget()
{
    if (content_ != nullptr) {
        const auto tabs = content_->findChildren<QTabWidget*>();
        for (auto* tab_widget : tabs) {
            while (tab_widget->count() > 0) {
                tab_widget->removeTab(0);
            }
        }
    }

    for (auto* panel_widget : panels_) {
        if (panel_widget != nullptr && state_.chart_detached(panel_widget->chart_id())) {
            continue;
        }
        panel_widget->hide();
        panel_widget->setParent(this);
    }

    if (content_ != nullptr) {
        root_layout_->removeWidget(content_);
        delete content_;
    }

    content_ = new QWidget(this);
    root_layout_->addWidget(content_, 1);
}

void ChartWorkspaceWidget::connect_panel(ChartPanelWidget& panel_widget)
{
    const auto chart_id = panel_widget.chart_id();
    panel_widget.setReloadRequestCallback([this](std::uint64_t chart_id, core::TimeRange range) {
        if (reload_request_callback_) {
            reload_request_callback_(chart_id, range);
        }
    });
    panel_widget.setPeriodChangedCallback([this](std::uint64_t chart_id, const std::string& period) {
        state_.set_chart_period(chart_id, canonical_chart_period(period));
        if (auto* window = floating_window(chart_id); window != nullptr) {
            window->refresh_title();
        }
    });
    panel_widget.setPopoutCallback([this](std::uint64_t chart_id) {
        detachChart(chart_id);
    });
    panel_widget.chart_view().set_crosshair_moved_callback([this, chart_id](std::int64_t timestamp_ns, double price) {
        crosshair_sync_controller_.sync_crosshair_from(chart_id, timestamp_ns, price);
    });

    crosshair_sync_controller_.register_chart(
        chart_id,
        [&panel_widget](std::int64_t timestamp_ns) {
            return panel_widget.chart_view().dense_x_for_timestamp(timestamp_ns);
        },
        [&panel_widget](const sync::CrosshairUpdate& update) {
            panel_widget.chart_view().sync_crosshair(update.timestamp_ns, update.price, update.dense_x);
        },
        [&panel_widget](const sync::CenterTimeUpdate& update) {
            panel_widget.chart_view().sync_center_on_timestamp(update.timestamp_ns, update.price);
        },
        [&panel_widget](const sync::YCenterUpdate& update) {
            panel_widget.chart_view().sync_y_center(update.price);
        });
}

FloatingChartWindow* ChartWorkspaceWidget::floating_window(std::uint64_t chart_id) const
{
    const auto found = std::find_if(
        detached_windows_.begin(),
        detached_windows_.end(),
        [chart_id](const DetachedChartWindow& detached) {
            return detached.chart_id == chart_id;
        });
    if (found == detached_windows_.end()) {
        return nullptr;
    }
    return found->window;
}

void ChartWorkspaceWidget::reattach_disabled_detached_charts()
{
    std::vector<std::uint64_t> disabled_detached_ids;
    for (const auto& detached : detached_windows_) {
        if (!state_.chart_enabled(detached.chart_id)) {
            disabled_detached_ids.push_back(detached.chart_id);
        }
    }
    for (const auto chart_id : disabled_detached_ids) {
        reattachChart(chart_id);
    }
}

void ChartWorkspaceWidget::refresh_sync_enabled_charts()
{
    const auto enabled_ids = state_.enabled_chart_ids();
    const auto registered_ids = crosshair_sync_controller_.registered_chart_ids();
    for (const auto chart_id : registered_ids) {
        const auto enabled = std::find(enabled_ids.begin(), enabled_ids.end(), chart_id) != enabled_ids.end();
        crosshair_sync_controller_.set_chart_enabled(chart_id, enabled);
    }
}

} // namespace tradereview::chart
