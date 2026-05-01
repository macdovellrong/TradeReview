#pragma once

#include <QWidget>

#include <functional>
#include <vector>

class QPushButton;
class QString;

namespace tradereview::chart {

class ChartToolbarWidget final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using IndicatorToggleCallback = std::function<void(const QString&, bool)>;
    using PeriodSelectedCallback = std::function<void(const QString&)>;
    using PriceAxisFitCallback = std::function<void()>;
    using PopoutCallback = std::function<void()>;

    explicit ChartToolbarWidget(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setIndicatorToggleCallback(IndicatorToggleCallback callback);
    void setPeriodSelectedCallback(PeriodSelectedCallback callback);
    void setPriceAxisFitCallback(PriceAxisFitCallback callback);
    void setPopoutCallback(PopoutCallback callback);
    void setSelectedPeriod(const QString& period);

private:
    void notify_indicator_toggle(const QString& indicator, bool enabled) const;
    void notify_period_selected(const QString& period) const;

    StatusCallback status_callback_;
    IndicatorToggleCallback indicator_toggle_callback_;
    PeriodSelectedCallback period_selected_callback_;
    PriceAxisFitCallback price_axis_fit_callback_;
    PopoutCallback popout_callback_;
    std::vector<QPushButton*> period_buttons_;
};

} // namespace tradereview::chart
