import finplot as fplt
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QComboBox, QLabel, QSlider, QDateTimeEdit, QSplitter)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from engine.data_engine import DataEngine
import datetime

# 封装单个图表窗口
class ChartWidget(QWidget):
    def __init__(self, name="Chart", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        
        # 顶部工具条 (周期选择)
        self.toolbar_layout = QHBoxLayout()
        self.lbl_name = QLabel(name)
        self.combo_period = QComboBox()
        self.combo_period.addItems(["1min", "5min", "15min", "30min", "1h", "4h", "1D"])
        
        self.toolbar_layout.addWidget(self.lbl_name)
        self.toolbar_layout.addWidget(self.combo_period)
        self.toolbar_layout.addStretch()
        
        self.layout.addLayout(self.toolbar_layout)
        
        # Finplot 画布
        # 使用 create_plot_widget 可以嵌入到 PyQt 布局中
        self.ax = fplt.create_plot_widget(master=self)
        self.layout.addWidget(self.ax.vb.win)
        
        # 数据缓存
        self.current_period = "1min"
        self.plot_item = None # K线图对象
        self.df_data = None
        
        # 信号连接
        self.combo_period.currentTextChanged.connect(self.on_period_change)

    def on_period_change(self, text):
        self.current_period = text
        # 这里只是更新了标记，主循环 update 时会拉取新数据

    def update_chart(self, df):
        if df is None or df.empty:
            return

        # 第一次绘制
        if self.plot_item is None:
            self.plot_item = fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=self.ax)
            # 设置一些默认交互
            self.ax.showGrid(True, True)
        else:
            # 更新数据
            self.plot_item.update_data(df[['open', 'close', 'high', 'low']])
            # 保持 X 轴跟随最新数据 (可选，如果用户在回看历史可以不强制)
            # self.ax.set_x_range(...) 


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Trade Review")
        self.resize(1400, 900)
        
        # 核心引擎
        self.engine = DataEngine() # 默认加载 data/ticks.parquet
        if self.engine.df_ticks is not None:
             self.current_time = self.engine.df_ticks.index[1000]
        else:
             self.current_time = datetime.datetime.now() # Fallback

        self.is_playing = False
        self.replay_speed = 60 # 1秒走60秒数据

        # 主布局
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # 1. 顶部全局控制栏
        self.create_control_panel()
        
        # 2. 图表区域 (使用 Splitter 支持拖动调整大小)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # 初始化两个图表
        self.chart1 = ChartWidget("Chart 1")
        self.chart1.combo_period.setCurrentText("1h") # 默认 1小时
        
        self.chart2 = ChartWidget("Chart 2")
        self.chart2.combo_period.setCurrentText("5min") # 默认 5分钟

        self.splitter.addWidget(self.chart1)
        self.splitter.addWidget(self.chart2)
        
        # 3. 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(100) # 100ms 刷新

    def create_control_panel(self):
        panel = QHBoxLayout()
        
        # 播放/暂停
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        panel.addWidget(self.btn_play)
        
        # 速度控制
        panel.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 1000) # 1x 到 1000x
        self.slider_speed.setValue(60)
        self.slider_speed.valueChanged.connect(self.change_speed)
        panel.addWidget(self.slider_speed)
        self.lbl_speed_val = QLabel("60x")
        panel.addWidget(self.lbl_speed_val)

        # 时间跳转
        self.date_edit = QDateTimeEdit()
        self.date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        # 将当前时间转为 QDateTime
        # 注意：pandas timestamp -> python datetime -> qdatetime
        self.date_edit.setDateTime(QDateTime(self.current_time.year, self.current_time.month, self.current_time.day,
                                           self.current_time.hour, self.current_time.minute, self.current_time.second))
        self.date_edit.dateTimeChanged.connect(self.on_date_change)
        panel.addWidget(self.date_edit)
        
        panel.addStretch()
        self.main_layout.addLayout(panel)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def change_speed(self, val):
        self.replay_speed = val
        self.lbl_speed_val.setText(f"{val}x")

    def on_date_change(self, qdt):
        # 用户手动修改时间时触发
        # 这里需要注意，如果是 timer 自动更新导致的变动，不要死循环
        # 简单处理：只有在暂停时允许手动精确跳转，或者加个 flag
        pass 

    def on_timer_tick(self):
        if not self.is_playing:
            return

        # 1. 推进时间
        self.current_time += datetime.timedelta(seconds=self.replay_speed)
        
        # 更新时间显示 (不触发 signal)
        self.date_edit.blockSignals(True)
        self.date_edit.setDateTime(QDateTime(self.current_time.year, self.current_time.month, self.current_time.day,
                                           self.current_time.hour, self.current_time.minute, self.current_time.second))
        self.date_edit.blockSignals(False)

        # 2. 更新所有图表
        # Chart 1
        df1 = self.engine.get_candles_by_time(self.chart1.current_period, self.current_time, count=300)
        self.chart1.update_chart(df1)
        
        # Chart 2
        df2 = self.engine.get_candles_by_time(self.chart2.current_period, self.current_time, count=300)
        self.chart2.update_chart(df2)

    def run(self):
        # Finplot 全局配置
        fplt.foreground = '#FFFFFF'
        fplt.background = '#000000'
        fplt.show(qt_exec=True) 

