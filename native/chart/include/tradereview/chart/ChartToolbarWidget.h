#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::chart {

class ChartToolbarWidget final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using IndicatorToggleCallback = std::function<void(const QString&, bool)>;

    explicit ChartToolbarWidget(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setIndicatorToggleCallback(IndicatorToggleCallback callback);

private:
    void notify(const QString& action) const;
    void notify_indicator_toggle(const QString& indicator, bool enabled) const;

    StatusCallback status_callback_;
    IndicatorToggleCallback indicator_toggle_callback_;
};

} // namespace tradereview::chart
