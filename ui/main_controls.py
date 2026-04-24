import datetime

from PyQt6.QtCore import QDateTime, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class MainControls(QWidget):
    load_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    save_view_requested = pyqtSignal()
    layout_changed = pyqtSignal(str)
    pop_layout_requested = pyqtSignal()
    chart_count_changed = pyqtSignal(str)
    replay_mode_changed = pyqtSignal(int)
    play_requested = pyqtSignal()
    step_back_requested = pyqtSignal()
    step_forward_requested = pyqtSignal()
    speed_changed = pyqtSignal(int)
    date_edit_finished = pyqtSignal()

    def __init__(self, current_time=None, replay_speed=60, parent=None):
        super().__init__(parent)

        panel = QHBoxLayout()
        panel.setContentsMargins(0, 0, 0, 0)
        self.setLayout(panel)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(lambda checked=False: self.load_requested.emit())
        panel.addWidget(self.btn_load)

        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(lambda checked=False: self.reset_requested.emit())
        panel.addWidget(self.btn_reset)

        self.btn_save_view = QPushButton("Save View")
        self.btn_save_view.clicked.connect(lambda checked=False: self.save_view_requested.emit())
        panel.addWidget(self.btn_save_view)

        panel.addWidget(QLabel("Layout:"))
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["Tabs", "Dual Vertical", "Grid 2x2", "Vertical"])
        self.combo_layout.currentTextChanged.connect(self.layout_changed.emit)
        panel.addWidget(self.combo_layout)

        self.btn_detach_layout = QPushButton("Pop Layout")
        self.btn_detach_layout.clicked.connect(lambda checked=False: self.pop_layout_requested.emit())
        panel.addWidget(self.btn_detach_layout)

        panel.addWidget(QLabel("Charts:"))
        self.combo_chart_count = QComboBox()
        self.combo_chart_count.addItems(["1", "2", "3", "4"])
        self.combo_chart_count.setCurrentText("4")
        self.combo_chart_count.currentTextChanged.connect(self.chart_count_changed.emit)
        panel.addWidget(self.combo_chart_count)

        self.chk_replay = QCheckBox("Replay Mode")
        self.chk_replay.setChecked(False)
        self.chk_replay.stateChanged.connect(self.replay_mode_changed.emit)
        panel.addWidget(self.chk_replay)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(lambda checked=False: self.play_requested.emit())
        self.btn_play.setEnabled(False)
        panel.addWidget(self.btn_play)

        self.btn_step_back = QPushButton("Back")
        self.btn_step_back.clicked.connect(lambda checked=False: self.step_back_requested.emit())
        panel.addWidget(self.btn_step_back)

        self.btn_step_forward = QPushButton("Forward")
        self.btn_step_forward.clicked.connect(lambda checked=False: self.step_forward_requested.emit())
        panel.addWidget(self.btn_step_forward)

        self.combo_step = QComboBox()
        self.combo_step.addItems(["30s", "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D"])
        self.combo_step.setCurrentText("1h")
        panel.addWidget(self.combo_step)

        panel.addWidget(QLabel("Speed:"))
        self.speed_btn_group = QButtonGroup(self)
        self.speed_btn_group.setExclusive(True)

        for speed in [1, 10, 60, 120, 300, 600]:
            btn = QPushButton(f"{speed}x")
            btn.setCheckable(True)
            btn.setFixedSize(40, 25)
            if speed == replay_speed:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, val=speed: self.speed_changed.emit(val))
            self.speed_btn_group.addButton(btn)
            panel.addWidget(btn)

        self.date_edit = QDateTimeEdit()
        self.date_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setKeyboardTracking(False)
        dt = current_time or datetime.datetime.now()
        self.date_edit.setDateTime(QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0))
        self.date_edit.editingFinished.connect(self.date_edit_finished.emit)
        panel.addWidget(self.date_edit)

        panel.addStretch()
