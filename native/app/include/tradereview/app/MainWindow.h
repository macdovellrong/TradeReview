#pragma once

#include "tradereview/app/DataLoadController.h"

#include <QMainWindow>
#include <QSettings>

#include <memory>

class QString;

namespace tradereview::chart {
class ChartWorkspaceWidget;
}

namespace tradereview::app {

class MainControlsBar;
class SideInfoPanelWidget;
class StatusStripWidget;

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void showStatusMessage(const QString& message);
    void setLoadingStatus(const QString& message);
    void setReadyStatus(const QString& message);

    QSettings settings_;
    std::unique_ptr<DataLoadController> data_load_controller_;
    MainControlsBar* main_controls_ = nullptr;
    chart::ChartWorkspaceWidget* chart_workspace_ = nullptr;
    SideInfoPanelWidget* side_info_panel_ = nullptr;
    StatusStripWidget* status_strip_ = nullptr;
};

} // namespace tradereview::app
