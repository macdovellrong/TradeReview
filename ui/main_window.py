import finplot as fplt
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QComboBox, QLabel, QSlider, QDateTimeEdit, QSplitter, QCheckBox, QFileDialog, QGridLayout, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, QDateTime
from engine.data_engine import DataEngine
import datetime
import os

class MockYScale:
    def __init__(self):
        self.scalef = 1
        self.scaletype = 'linear'

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
        
        # Finplot 画布嵌入逻辑
        self.glw = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.glw)
        
        self.ax = self.glw.addPlot(axisItems={'bottom': pg.DateAxisItem()})
        self.ax.significant_decimals = 4 
        self.ax.significant_eps = 1e-4
        
        # 3. Monkeypatch: 欺骗 Finplot
        self.ax.vb.yscale = MockYScale()
        self.ax.vb.datasrc = None 
        self.ax.vb.v_zoom_scale = 0.9
        self.ax.vb.x_zoom_scale = 1.0 
        self.ax.vb.x_indexed = True
        self.ax.vb.win = self.glw 
        self.glw._isMouseLeftDrag = False 
        self.ax.vb.master_viewbox = None # 修复 AttributeError
        
        def set_datasrc(ds):
            # 手动初始化这些属性，防止 finplot 添加指标时报错
            if not hasattr(ds, 'init_x0'):
                ds.init_x0 = 0
            if not hasattr(ds, 'init_x1'):
                ds.init_x1 = len(ds.df) if hasattr(ds, 'df') else 0
            self.ax.vb.datasrc = ds
        self.ax.vb.set_datasrc = set_datasrc

        def update_y_zoom(x0, x1):
            pass
        self.ax.vb.update_y_zoom = update_y_zoom

        # 自定义滚轮逻辑
        def custom_wheel_event(ev, axis=None):
            # 强制接管事件，不检查 isAccepted
            
            # 计算缩放比例
            # PyqtGraph 的 QGraphicsSceneWheelEvent 可能没有 angleDelta
            # 使用 delta() 方法
            delta = ev.delta()
            if delta == 0:
                return
                
            s = 0.85 ** (delta / 120.0)
            
            # 获取缩放中心
            try:
                pos = ev.scenePosition()
                center = self.ax.vb.mapSceneToView(pos)
            except:
                center = None

            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl 按下：缩放 Y 轴
                self.ax.vb.scaleBy(x=1, y=s, center=center)
            else:
                # 普通滚动：缩放 X 轴
                self.ax.vb.scaleBy(x=s, y=1, center=center)
            
            ev.accept()
        
        # 保存这个函数引用，以便后续重新绑定
        self._custom_wheel_event = custom_wheel_event
        self.ax.vb.wheelEvent = custom_wheel_event
        
        # 属性设置
        self.ax.showGrid(x=True, y=True)
        self.ax.showAxis('right', show=True)
        self.ax.showAxis('left', show=False)
        self.ax.getAxis('right').setWidth(60)

        # 缩放交互
        self.ax.setMouseEnabled(x=True, y=True)
        self.ax.getAxis('right').enableAutoSIPrefix(False)

        # 还原黑色背景
        pg.setConfigOptions(foreground='#FFFFFF', background='#000000')
        self.glw.setBackground('k')
        
        self.current_period = "1min"
        self.plot_item = None 
        self.indicator_items = {} # 存储指标对象 {name: item}
        self.combo_period.currentTextChanged.connect(self.on_period_change)

    def on_period_change(self, text):
        self.current_period = text

    def update_chart(self, df):
        if df is None or df.empty:
            return

        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)

        if self.plot_item is None:
            self.plot_item = fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']], ax=self.ax)
            self.ax.showGrid(True, True)
            self.ax.setLogMode(y=False)
            self.plot_item.setZValue(10)
        else:
            self.plot_item.update_data(df[['open', 'close', 'high', 'low']])
        
        # 2. 绘制指标
        ema_colors = {
            'EMA20': '#FF0000', 'EMA30': '#FF8800', 'EMA40': '#FFFF00', 
            'EMA50': '#00FF00', 'EMA60': '#0000FF'
        }
        
        # 准备 X 轴数据 (确保是 np.array)
        if self.ax.vb.x_indexed:
             x_data = np.arange(len(df))
        else:
             x_data = df.index.astype(np.int64) // 10**9 
             
        # 绘制 EMA
        for name, color in ema_colors.items():
            if name in df.columns:
                # 强制转换为 float64 的 numpy array
                y_data = df[name].to_numpy(dtype=np.float64)
                
                if name not in self.indicator_items:
                    curve = pg.PlotCurveItem(x=x_data, y=y_data, pen=pg.mkPen(color, width=1.5), name=name)
                    self.ax.addItem(curve)
                    self.indicator_items[name] = curve
                else:
                    self.indicator_items[name].setData(x=x_data, y=y_data)

        # 绘制布林带
        bb_color = '#FFFFFF'
        for name in ['BB_Upper', 'BB_Lower']:
            if name in df.columns:
                y_data = df[name].to_numpy(dtype=np.float64)
                
                if name not in self.indicator_items:
                    curve = pg.PlotCurveItem(x=x_data, y=y_data, pen=pg.mkPen(bb_color, width=1), name=name)
                    self.ax.addItem(curve)
                    self.indicator_items[name] = curve
                else:
                    self.indicator_items[name].setData(x=x_data, y=y_data)

        if len(df) > 0:
            y_min = df['low'].min()
            y_max = df['high'].max()
            y_pad = (y_max - y_min) * 0.05
            self.ax.setYRange(y_min - y_pad, y_max + y_pad, padding=0)
            self.ax.setXRange(0, len(df), padding=0.02)
            
        # 重新绑定滚轮事件，防止被覆盖
        if hasattr(self, '_custom_wheel_event'):
            self.ax.vb.wheelEvent = self._custom_wheel_event


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Trade Review")
        self.resize(1400, 950)
        
        self.engine = DataEngine(parquet_file=None) 
        self.current_time = datetime.datetime.now()
        self.is_playing = False
        self.replay_speed = 60 

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        self.create_control_panel()
        
        # 图表容器区域
        self.chart_container_layout = QVBoxLayout() # 这一层用于承载变化的布局
        self.main_layout.addLayout(self.chart_container_layout)
        
        self.charts = []
        # 初始化4个图表对象 (但不先加到界面)
        self.init_charts()
        
        # 默认使用田字格布局
        self.switch_layout("Grid 2x2")
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(100) 

    def init_charts(self):
        configs = [("H1", "1h"), ("M15", "15min"), ("M5", "5min"), ("M1", "1min")]
        for name, period in configs:
            chart = ChartWidget(name)
            chart.combo_period.setCurrentText(period)
            chart.combo_period.currentTextChanged.connect(lambda _, c=chart: self.refresh_single_chart(c))
            self.charts.append(chart)

    def create_control_panel(self):
        panel = QHBoxLayout()
        
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.open_file_dialog)
        panel.addWidget(btn_load)
        
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.reset_charts_view)
        panel.addWidget(btn_reset)
        
        # 布局选择
        panel.addWidget(QLabel("Layout:"))
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["Grid 2x2", "Vertical", "Tabs"])
        self.combo_layout.currentTextChanged.connect(self.switch_layout)
        panel.addWidget(self.combo_layout)

        self.chk_replay = QCheckBox("Replay Mode")
        self.chk_replay.setChecked(False)
        self.chk_replay.stateChanged.connect(self.on_mode_change)
        panel.addWidget(self.chk_replay)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        panel.addWidget(self.btn_play)
        
        panel.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 1000)
        self.slider_speed.setValue(60)
        self.slider_speed.valueChanged.connect(self.change_speed)
        panel.addWidget(self.slider_speed)
        self.lbl_speed_val = QLabel("60x")
        panel.addWidget(self.lbl_speed_val)

        self.date_edit = QDateTimeEdit()
        self.date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.date_edit.setDateTime(QDateTime(self.current_time.year, self.current_time.month, self.current_time.day,
                                           self.current_time.hour, self.current_time.minute, self.current_time.second))
        panel.addWidget(self.date_edit)
        
        panel.addStretch()
        self.main_layout.addLayout(panel)

    def switch_layout(self, layout_name):
        # 1. 清空当前容器
        # 注意：这里不能删除 widget，只能把它们从 layout 里移除（setParent(None)）
        # 或者直接删除旧的 layout item
        while self.chart_container_layout.count():
            item = self.chart_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None) # 把旧的容器拿掉
        
        # 2. 根据模式创建新容器并添加图表
        if layout_name == "Vertical":
            splitter = QSplitter(Qt.Orientation.Vertical)
            for chart in self.charts:
                splitter.addWidget(chart)
            self.chart_container_layout.addWidget(splitter)
            
        elif layout_name == "Grid 2x2":
            # Grid 放在一个 Widget 里，方便统一管理
            grid_widget = QWidget()
            grid = QGridLayout()
            grid_widget.setLayout(grid)
            
            grid.addWidget(self.charts[0], 0, 0)
            grid.addWidget(self.charts[1], 0, 1)
            grid.addWidget(self.charts[2], 1, 0)
            grid.addWidget(self.charts[3], 1, 1)
            
            self.chart_container_layout.addWidget(grid_widget)
            
        elif layout_name == "Tabs":
            tabs = QTabWidget()
            for chart in self.charts:
                tabs.addTab(chart, chart.lbl_name.text())
            self.chart_container_layout.addWidget(tabs)
            
        # 切换布局后可能需要重新触发一下 resize 或刷新
        self.refresh_all_charts()

    # ... (Rest of the methods remain same) ...
    def load_data_file(self, file_path):
        self.engine.parquet_file = file_path
        self.engine.load_data()
        if self.engine.df_ticks is not None:
             if len(self.engine.df_ticks) > 100000:
                self.current_time = self.engine.df_ticks.index[100000]
             else:
                self.current_time = self.engine.df_ticks.index[0]
             if hasattr(self, 'date_edit'):
                 self.date_edit.setDateTime(QDateTime(self.current_time.year, self.current_time.month, self.current_time.day,
                                                    self.current_time.hour, self.current_time.minute, self.current_time.second))
             self.refresh_all_charts()

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Parquet Data", "", "Parquet Files (*.parquet);;All Files (*)")
        if file_name:
            self.load_data_file(file_name)

    def on_mode_change(self, state):
        is_replay = self.chk_replay.isChecked()
        self.btn_play.setEnabled(is_replay)
        self.refresh_all_charts()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def change_speed(self, val):
        self.replay_speed = val
        self.lbl_speed_val.setText(f"{val}x")

    def reset_charts_view(self):
        self.refresh_all_charts()
        for chart in self.charts:
            chart.ax.autoRange()

    def refresh_single_chart(self, chart):
        if self.engine.df_ticks is None:
            return
        if self.chk_replay.isChecked():
            df = self.engine.get_candles_by_time(chart.current_period, self.current_time, count=300)
        else:
            df = self.engine.get_candles(chart.current_period) 
        chart.update_chart(df)

    def refresh_all_charts(self):
        for chart in self.charts:
            self.refresh_single_chart(chart)

    def on_timer_tick(self):
        if not self.chk_replay.isChecked() or not self.is_playing:
            return
        self.current_time += datetime.timedelta(seconds=self.replay_speed)
        self.date_edit.blockSignals(True)
        self.date_edit.setDateTime(QDateTime(self.current_time.year, self.current_time.month, self.current_time.day,
                                           self.current_time.hour, self.current_time.minute, self.current_time.second))
        self.date_edit.blockSignals(False)
        self.refresh_all_charts()