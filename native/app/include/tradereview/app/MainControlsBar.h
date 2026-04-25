#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::app {

class MainControlsBar final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;

    explicit MainControlsBar(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);

private:
    void notify(const QString& action) const;

    StatusCallback status_callback_;
};

} // namespace tradereview::app
