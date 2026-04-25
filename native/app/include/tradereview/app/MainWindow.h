#pragma once

#include <QMainWindow>

namespace tradereview::app {

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);
};

} // namespace tradereview::app
