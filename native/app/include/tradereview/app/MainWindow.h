#pragma once

#include "tradereview/app/DataLoadController.h"

#include <QMainWindow>

#include <memory>

namespace tradereview::app {

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    std::unique_ptr<DataLoadController> data_load_controller_;
};

} // namespace tradereview::app
