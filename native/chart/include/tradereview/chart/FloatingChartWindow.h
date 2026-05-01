#pragma once

#include <QMainWindow>

#include <cstdint>
#include <functional>

class QCloseEvent;
class QWidget;

namespace tradereview::chart {

class ChartPanelWidget;

class FloatingChartWindow final : public QMainWindow {
public:
    using CloseCallback = std::function<void(std::uint64_t)>;

    explicit FloatingChartWindow(ChartPanelWidget* panel, QWidget* parent = nullptr);

    [[nodiscard]] std::uint64_t chart_id() const;
    [[nodiscard]] ChartPanelWidget* take_panel();
    void setCloseCallback(CloseCallback callback);
    void refresh_title();

private:
    void closeEvent(QCloseEvent* event) override;

    ChartPanelWidget* panel_ = nullptr;
    CloseCallback close_callback_;
};

} // namespace tradereview::chart
