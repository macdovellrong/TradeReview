#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::app {

class MainControlsBar final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using LoadDataCallback = std::function<void()>;
    using LayoutModeCallback = std::function<void(const QString&)>;
    using ChartCountCallback = std::function<void(int)>;

    explicit MainControlsBar(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setLoadDataCallback(LoadDataCallback callback);
    void setLayoutModeCallback(LayoutModeCallback callback);
    void setChartCountCallback(ChartCountCallback callback);

private:
    void loadData() const;
    void selectLayoutMode(const QString& mode) const;
    void selectChartCount(const QString& count) const;
    void notify(const QString& action) const;

    StatusCallback status_callback_;
    LoadDataCallback load_data_callback_;
    LayoutModeCallback layout_mode_callback_;
    ChartCountCallback chart_count_callback_;
};

} // namespace tradereview::app
