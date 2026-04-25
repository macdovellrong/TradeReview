#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::chart {

class ChartToolbarWidget final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;

    explicit ChartToolbarWidget(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);

private:
    void notify(const QString& action) const;

    StatusCallback status_callback_;
};

} // namespace tradereview::chart
