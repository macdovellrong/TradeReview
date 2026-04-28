#include "tradereview/app/AppTheme.h"

#include <QApplication>
#include <QColor>
#include <QString>

namespace tradereview::app::theme {

QColor chartBackground()
{
    return QColor(11, 16, 23);
}

QColor chartEmptyText()
{
    return QColor(143, 160, 184);
}

QColor loadingOverlay()
{
    return QColor(15, 18, 24, 132);
}

QString styleSheet()
{
    return R"(
        QMainWindow {
            background-color: #0d1118;
            color: #d7dde6;
        }
        QMenuBar {
            background-color: #151b24;
            color: #d7dde6;
            border-bottom: 1px solid #2a3545;
            padding: 3px 8px;
        }
        QMenuBar::item:selected {
            background-color: #222d3b;
        }
        QStatusBar {
            background-color: #101720;
            color: #8fa0b8;
            border-top: 1px solid #2a3545;
        }
        #MainControlsBar,
        #ChartToolbarWidget,
        #SideInfoPanelWidget,
        #StatusStripWidget,
        #DrawingToolRailWidget {
            background-color: #131a23;
            color: #d7dde6;
        }
        #MainControlsBar {
            border-bottom: 1px solid #2a3545;
        }
        #ChartToolbarWidget {
            background-color: #151d27;
            border-bottom: 1px solid #2a3545;
        }
        QWidget#ToolbarGroup {
            background-color: transparent;
        }
        QScrollArea#PeriodScrollArea {
            background-color: #151d27;
            border: none;
        }
        QWidget#PeriodScrollViewport,
        QWidget#PeriodScrollContent {
            background-color: #151d27;
        }
        #SideInfoPanelWidget {
            border-left: 1px solid #2a3545;
        }
        #StatusStripWidget {
            border-top: 1px solid #2a3545;
        }
        #DrawingToolRailWidget {
            border-right: 1px solid #2a3545;
        }
        QLabel,
        QCheckBox {
            color: #cbd5e1;
        }
        QPushButton,
        QComboBox,
        QDateTimeEdit {
            min-height: 24px;
            background-color: #222d3b;
            border: 1px solid #354256;
            border-radius: 6px;
            color: #d7dde6;
            padding: 3px 8px;
        }
        QPushButton:hover,
        QComboBox:hover,
        QDateTimeEdit:hover {
            background-color: #293548;
            border-color: #42516a;
            color: #f8fafc;
        }
        QPushButton:checked,
        QPushButton[primary="true"] {
            background-color: #0f766e;
            border-color: #13a39b;
            color: #f8fafc;
        }
        QPushButton:disabled,
        QComboBox:disabled,
        QDateTimeEdit:disabled {
            background-color: #17202b;
            border-color: #273142;
            color: #58677d;
        }
        QTabWidget::pane {
            border: 1px solid #2a3545;
        }
        QTabBar::tab {
            background-color: #18202b;
            color: #8fa0b8;
            padding: 6px 12px;
            border: 1px solid #2a3545;
        }
        QTabBar::tab:selected {
            background-color: #222d3b;
            color: #d7dde6;
        }
    )";
}

void apply(QApplication& app)
{
    app.setStyleSheet(styleSheet());
}

} // namespace tradereview::app::theme
