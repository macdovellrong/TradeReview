from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class FloatingChartWindow(QWidget):
    sig_window_closed = pyqtSignal(object)

    def __init__(self, chart_widget, parent=None):
        super().__init__(parent)
        self.chart_widget = chart_widget
        self.setWindowTitle(f"Chart - {chart_widget.current_period}")
        self.resize(800, 600)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        layout.addWidget(self.chart_widget)
        self.chart_widget.show()

        self.chart_widget.sig_period_changed.connect(self.update_title)

    def update_title(self, period_display):
        self.setWindowTitle(f"Chart - {period_display}")

    def closeEvent(self, event):
        self.sig_window_closed.emit(self.chart_widget)
        event.accept()
