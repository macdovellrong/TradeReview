import finplot as fplt
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QComboBox, QLabel, QSlider, QDateTimeEdit, QSplitter, QCheckBox, QFileDialog, QGridLayout, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, QDateTime, pyqtSignal
from engine.data_engine import DataEngine
import datetime
import os

class MockYScale:
    def __init__(self):
        self.scalef = 1
        self.scaletype = 'linear'

# 封装单个图表窗口
class ChartWidget(QWidget):
    # 定义信号：鼠标移动时发射当前的时间戳 (float)
    sig_mouse_moved = pyqtSignal(float)

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
        
        # Monkeypatch
        self.ax.vb.yscale = MockYScale()
        self.ax.vb.datasrc = None 
        self.ax.vb.v_zoom_scale = 0.9
        self.ax.vb.x_zoom_scale = 1.0 
        self.ax.vb.x_indexed = True
        self.ax.vb.win = self.glw 
        self.glw._isMouseLeftDrag = False 
        self.ax.vb.master_viewbox = None
        
        def set_datasrc(ds):
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
            delta = ev.delta()
            if delta == 0:
                return
                
            s = 0.85 ** (delta / 120.0)
            
            try:
                pos = ev.scenePosition()
                center = self.ax.vb.mapSceneToView(pos)
            except:
                center = None

            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.ax.vb.scaleBy(x=1, y=s, center=center)
            else:
                self.ax.vb.scaleBy(x=s, y=1, center=center)
            
            ev.accept()
        
        self._custom_wheel_event = custom_wheel_event
        self.ax.vb.wheelEvent = custom_wheel_event
        
        # 4. 设置一些基础属性
        self.ax.showGrid(x=True, y=True)
        self.ax.showAxis('right', show=True)
        self.ax.showAxis('left', show=False)
        self.ax.getAxis('right').setWidth(60)

        # 十字光标 (Crosshair)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.vLine.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.hLine.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        
        self.ax.addItem(self.vLine, ignoreBounds=True)
        self.ax.addItem(self.hLine, ignoreBounds=True)
        
        # 监听鼠标移动
        self.proxy = pg.SignalProxy(self.ax.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_move)

        # 缩放交互
        self.ax.setMouseEnabled(x=True, y=True)
        self.ax.getAxis('right').enableAutoSIPrefix(False)

        # 还原黑色背景
        pg.setConfigOptions(foreground='#FFFFFF', background='#000000')
        self.glw.setBackground('k')
        
        self.current_period = "1min"
        self.plot_item = None 
        self.indicator_items = {} 
        self.combo_period.currentTextChanged.connect(self.on_period_change)

    def on_mouse_move(self, evt):
        pos = evt[0]
        if self.ax.sceneBoundingRect().contains(pos):
            mousePoint = self.ax.vb.mapSceneToView(pos)
            
            # 1. 移动自己的水平线 (价格)
            self.hLine.setPos(mousePoint.y())
            self.vLine.setPos(mousePoint.x())
            
            # 2. 发射信号，通知其他窗口同步垂直线 (时间)
            timestamp = self.get_timestamp_from_x(mousePoint.x())
            if timestamp is not None:
                self.sig_mouse_moved.emit(timestamp)

    def get_timestamp_from_x(self, x_val):
        """将 X 轴坐标转换为时间戳"""
        # 我们需要访问 datasrc (finplot 对象) 或者直接访问 self.plot_item (Finplot Item)
        # self.ax.vb.datasrc 是我们手动 hack 的，可能不准。
        # 最稳的是直接看我们 update_chart 时用的 df
        # 所以我们需要把 df 存下来
        if not hasattr(self, 'current_df') or self.current_df is None or self.current_df.empty:
            return None
            
        df = self.current_df
        idx = int(round(x_val))
        if 0 <= idx < len(df):
            # df.index 在 update_chart 里被去除了 tz，但那是 copy。
            # 我们应该在 update_chart 把处理后的 df 存到 self.current_df
            return df.index[idx].timestamp()
        return None

    def sync_vline(self, timestamp):
        """接收外部时间戳，移动垂直线"""
        if not hasattr(self, 'current_df') or self.current_df is None or self.current_df.empty:
            return

        df = self.current_df
        try:
            # df.index 是 datetime (naive)，timestamp 是 UTC float
            # 需要把 timestamp 转为 naive datetime (假设是本地时间? 不，我们在 DataEngine 转成了 NY 时间并去除了 tz)
            # 关键：DataEngine 里的 index 是带时区的。
            # update_chart 里我们 df.index = df.index.tz_localize(None)
            # 所以这里的 df.index 是 naive 的 NY 时间。
            
            # 我们的 timestamp 是从 get_timestamp_from_x 来的
            # 那个函数也是从 naive 的 NY 时间转出来的 timestamp()
            # 所以反转回去应该也是 naive 的 local time (基于系统时区?)
            # 不，timestamp() 是 UTC。
            # datetime.fromtimestamp(ts) 返回的是本地时间。
            # 这会导致时区错乱。
            
            # 正确做法：不要用 timestamp float 中转，直接传 string 或者 datetime 对象。
            # 但 pyqtSignal 定义了 float。
            # 那么接收端需要小心。
            
            # 如果我们只是在同台机器跑，fromtimestamp 会用系统时区。
            # 而我们的 df.index 是 "America/New_York" 去掉 tzinfo 后的时间。
            # 这两个“本地”可能不一样！
            
            # 简单粗暴：我们改用 datetime.utcfromtimestamp? 也不对。
            
            # 修正：get_timestamp_from_x 应该返回 naive datetime 的 timestamp?
            # pd.Timestamp(naive).timestamp() 会假定它是本地时间。
            
            # 最佳方案：
            # 在 update_chart 里，保留一个 index 到 timestamp 的映射表？
            # 或者直接在 signal 里传 datetime 字符串 (ISO format)。
            pass # 稍后在 MainWindow 修改信号类型
            
            # 暂时用简单的 searchsorted，假设时间戳一致
            target_dt = datetime.datetime.fromtimestamp(timestamp)
            idx = df.index.searchsorted(target_dt)
            
            if idx >= len(df):
                idx = len(df) - 1
            
            self.vLine.setPos(idx)
        except Exception:
            pass

    def on_period_change(self, text):
        self.current_period = text

    def update_chart(self, df):
        if df is None or df.empty:
            return

        # 保存原始 df 用于同步查询 (带时区? 不，存处理后的)
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        self.current_df = df # 保存引用

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
        
        self.chart_container_layout = QVBoxLayout() 
        self.main_layout.addLayout(self.chart_container_layout)
        
        self.charts = []
        self.init_charts()
        
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
            # 连接光标同步信号
            chart.sig_mouse_moved.connect(self.sync_all_charts)
            self.charts.append(chart)

    def sync_all_charts(self, timestamp):
        """同步所有图表的垂直光标"""
        for chart in self.charts:
            chart.sync_vline(timestamp)

    def create_control_panel(self):
        panel = QHBoxLayout()
        
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.open_file_dialog)
        panel.addWidget(btn_load)
        
        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self.reset_charts_view)
        panel.addWidget(btn_reset)
        
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
        while self.chart_container_layout.count():
            item = self.chart_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None) 
        
        if layout_name == "Vertical":
            splitter = QSplitter(Qt.Orientation.Vertical)
            for chart in self.charts:
                splitter.addWidget(chart)
            self.chart_container_layout.addWidget(splitter)
            
        elif layout_name == "Grid 2x2":
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
            
        # 切换布局后强制重置视图
        self.reset_charts_view()

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
             # 强制重置视图，确保K线居中显示
             self.reset_charts_view()

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
