#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::app {

class MainControlsBar final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using LoadDataCallback = std::function<void()>;

    explicit MainControlsBar(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setLoadDataCallback(LoadDataCallback callback);

private:
    void loadData() const;
    void notify(const QString& action) const;

    StatusCallback status_callback_;
    LoadDataCallback load_data_callback_;
};

} // namespace tradereview::app
