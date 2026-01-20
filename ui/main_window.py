import finplot as fplt
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QComboBox, QLabel, QDateTimeEdit, QSplitter, QCheckBox, QFileDialog, QGridLayout, QTabWidget, QScrollArea, QButtonGroup, QApplication)
from PyQt6.QtGui import QAction, QPainter, QPicture
from PyQt6.QtCore import Qt, QTimer, QDateTime, pyqtSignal, QSize
from engine.data_engine import DataEngine
from engine.replay_engine import ReplayEngine
import datetime
import os

# ... (Keep MockYScale and TimeAxisItem as is) ...

class MockYScale:
    def __init__(self):
        self.scalef = 1
        self.scaletype = 'linear'

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dt_index = None
        self._delta = None

    def set_datetime_index(self, dt_index):
        self._dt_index = dt_index
        if len(dt_index) > 1:
            # 计算步长：取前 10 个点的差值的中位数，以防数据开头有缺口
            count = min(10, len(dt_index) - 1)
            deltas = []
            for i in range(count):
                deltas.append(dt_index[i+1] - dt_index[i])
            # 简单的取中位数
            deltas.sort()
            self._delta = deltas[len(deltas) // 2]
        else:
            self._delta = datetime.timedelta(minutes=1)

    def tickValues(self, minVal, maxVal, size):
        visible_range = maxVal - minVal
        if visible_range <= 0:
            return []
        target_ticks = max(2, int(size / 100))
        step = visible_range / target_ticks
        if step < 1:
            step = 1
        else:
            power_of_10 = 10 ** int(np.log10(step))
            rel_step = step / power_of_10
            if rel_step < 1.5:
                step = 1 * power_of_10
            elif rel_step < 3.5:
                step = 2 * power_of_10
            elif rel_step < 7.5:
                step = 5 * power_of_10
            else:
                step = 10 * power_of_10
        step = int(step)
        start = (int(minVal) // step) * step
        if start < minVal:
            start += step
        values = []
        val = start
        while val <= maxVal:
            values.append(val)
            val += step
        return [(step, values)]

    def tickStrings(self, values, scale, spacing):
        if self._dt_index is None or len(self._dt_index) == 0:
            return [""] * len(values)
        last_idx = len(self._dt_index) - 1
        out = []
        for x in values:
            idx = int(round(x))
            if 0 <= idx <= last_idx:
                dt = self._dt_index[idx]
            elif idx > last_idx:
                diff = idx - last_idx
                dt = self._dt_index[last_idx] + self._delta * diff
            else:
                diff = idx
                dt = self._dt_index[0] + self._delta * diff
            out.append(dt.strftime('%m-%d %H:%M'))
        
        # 调试：打印前几个刻度的映射情况 (仅当 values 包含较小索引时)
        if any(v < 5 for v in values) and len(out) > 0:
            print(f"Tick Debug: Values={values[:3]}, Strings={out[:3]}")
            
        return out

class CandlestickItem(pg.GraphicsObject):
    def __init__(self, x_data, open_data, close_data, high_data, low_data):
        super().__init__()
        self._x = x_data
        self._open = open_data
        self._close = close_data
        self._high = high_data
        self._low = low_data
        self._picture = None
        self._generate_picture()

    def set_data(self, x_data, open_data, close_data, high_data, low_data):
        self._x = x_data
        self._open = open_data
        self._close = close_data
        self._high = high_data
        self._low = low_data
        self._generate_picture()
        self.update()

    def _generate_picture(self):
        pic = QPicture()
        p = QPainter(pic)
        width = 0.6
        up_pen = pg.mkPen('#FFFFFF')
        down_pen = pg.mkPen('#FF4444')
        up_brush = pg.mkBrush('#FFFFFF')
        down_brush = pg.mkBrush('#FF4444')

        for x, o, c, h, l in zip(self._x, self._open, self._close, self._high, self._low):
            if np.isnan(o) or np.isnan(c) or np.isnan(h) or np.isnan(l):
                continue
            is_up = c >= o
            p.setPen(up_pen if is_up else down_pen)
            p.setBrush(up_brush if is_up else down_brush)
            p.drawLine(pg.Point(x, l), pg.Point(x, h))
            if c == o:
                p.drawLine(pg.Point(x - width / 2, o), pg.Point(x + width / 2, o))
            else:
                rect = pg.QtCore.QRectF(x - width / 2, o, width, c - o)
                p.drawRect(rect.normalized())

        p.end()
        self._picture = pic

    def paint(self, p, *args):
        if self._picture is not None:
            p.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        if self._picture is None:
            return pg.QtCore.QRectF()
        return pg.QtCore.QRectF(self._picture.boundingRect())

# 封装单个图表窗口
class ChartWidget(QWidget):
    # 定义信号：鼠标移动时发射当前的时间戳 (float)
    sig_mouse_moved = pyqtSignal(float)
    # 定义信号：鼠标移动时发射时间戳与价格
    sig_mouse_moved_with_price = pyqtSignal(float, float)
    # 信号：周期改变时发射 (str)
    sig_period_changed = pyqtSignal(str)
    # 信号：请求分离/还原
    sig_detach_requested = pyqtSignal(object)
    # 信号：请求同步所有图表中心点 (datetime)
    sig_sync_center_requested = pyqtSignal(object)
    # 信号：设置回放开始时间 (datetime)
    sig_set_replay_start = pyqtSignal(object)

    def __init__(self, name="Chart", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        
        # 顶部工具条 (周期选择)
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(0)
        
        # 使用 ScrollArea 来容纳众多按钮
        scroll = QScrollArea()
        scroll.setFixedHeight(40) # 固定高度
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QHBoxLayout()
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(2)
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        
        self.btn_group = QButtonGroup()
        self.btn_group.setExclusive(True)
        self.btn_group.buttonClicked.connect(self.on_btn_period_clicked)
        
        # 定义周期选项及其显示文本
        periods = [
            ("1min", "1m"), ("2min", "2m"), ("3min", "3m"), ("5min", "5m"), 
            ("10min", "10m"), ("15min", "15m"), ("30min", "30m"), ("45min", "45m"),
            ("1h", "1h"), ("2h", "2h"), ("3h", "3h"), ("4h", "4h"), 
            ("6h", "6h"), ("8h", "8h"), ("12h", "12h"),
            ("1D", "1D"), ("1W", "1W"), ("1M", "1M")
        ]
        
        self.period_map = {p[1]: p[0] for p in periods} # display -> actual
        self.display_map = {p[0]: p[1] for p in periods} # actual -> display
        
        for actual, display in periods:
            btn = QPushButton(display)
            btn.setCheckable(True)
            btn.setFixedSize(40, 30) # 小按钮
            btn.setStyleSheet("""
                QPushButton {
                    border: 1px solid #444;
                    background-color: #222;
                    color: #AAA;
                    border-radius: 2px;
                }
                QPushButton:checked {
                    background-color: #007ACC;
                    color: white;
                    border: 1px solid #007ACC;
                }
                QPushButton:hover {
                    background-color: #333;
                }
            """)
            self.btn_group.addButton(btn)
            scroll_layout.addWidget(btn)
            
            # 存储实际周期值
            btn.setProperty("period", actual)
            
        scroll_layout.addStretch()
        self.toolbar_layout.addWidget(scroll)

        # 分离按钮
        self.btn_detach = QPushButton("Pop")
        self.btn_detach.setFixedSize(40, 30)
        self.btn_detach.setStyleSheet("""
            QPushButton {
                border: 1px solid #444;
                background-color: #222;
                color: #AAA;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #333;
                color: white;
            }
        """)
        self.btn_detach.clicked.connect(self.on_detach_clicked)
        self.toolbar_layout.addWidget(self.btn_detach)
        
        self.layout.addLayout(self.toolbar_layout)
        
        self.is_detached = False
        
        # Finplot 画布嵌入逻辑
        self.glw = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.glw)
        
        self.time_axis = TimeAxisItem(orientation='bottom')
        self.ax = self.glw.addPlot(axisItems={'bottom': self.time_axis})
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
            # 1. 获取滚动增量
            if hasattr(ev, 'angleDelta'):
                delta = ev.angleDelta().y()
            else:
                delta = ev.delta()

            if delta == 0:
                return
                
            # 2. 计算缩放系数 (值越接近 1.0，缩放越慢)
            s_x = 0.92 ** (delta / 120.0)
            s_y = 0.97 ** (delta / 120.0)
            
            try:
                # 3. 获取鼠标位置 (Scene坐标)
                pos = ev.scenePos()
                
                # 4. 判定鼠标所在区域
                rect_x = self.ax.getAxis('bottom').sceneBoundingRect()
                rect_y = self.ax.getAxis('right').sceneBoundingRect()
                rect_plot = self.ax.vb.sceneBoundingRect()
                
                # 5. 计算缩放中心 (数据坐标系 - 必须使用实际价格/时间值)
                center = self.ax.vb.mapSceneToView(pos)

                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    self.ax.vb.scaleBy(x=1, y=s_y, center=center)
                elif rect_x.contains(pos):
                    self.ax.vb.scaleBy(x=s_x, y=1, center=center)
                elif rect_y.contains(pos):
                    self.ax.vb.scaleBy(x=1, y=s_y, center=center)
                elif rect_plot.contains(pos):
                    self.ax.vb.scaleBy(x=s_x, y=s_y, center=center)
                else:
                    self.ax.vb.scaleBy(x=s_x, y=1, center=center)
            
            except Exception as e:
                print(f"Zoom interaction error: {e}")
            
            ev.accept()
        
        self._custom_wheel_event = custom_wheel_event
        self.ax.vb.wheelEvent = custom_wheel_event

        orig_mouse_drag_event = self.ax.vb.mouseDragEvent
        def custom_mouse_drag_event(ev, axis=None):
            if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                ev.ignore()
                return
            return orig_mouse_drag_event(ev, axis)
        self.ax.vb.mouseDragEvent = custom_mouse_drag_event
        
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
        
        # 标签 (Labels) - 改用 TextItem 以便精确定位
        self.txt_price = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(1, 1))
        self.txt_price.setZValue(20) # 确保在最上层
        self.ax.addItem(self.txt_price, ignoreBounds=True)

        self.txt_time = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(0.5, 1))
        self.txt_time.setZValue(20)
        self.ax.addItem(self.txt_time, ignoreBounds=True)

        # 价差测算提示
        self.txt_measure = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(0, 1))
        self.txt_measure.setZValue(20)
        self.txt_measure.hide()
        self.ax.addItem(self.txt_measure, ignoreBounds=True)
        
        self.ax.addItem(self.vLine, ignoreBounds=True)
        self.ax.addItem(self.hLine, ignoreBounds=True)
        
        # 监听鼠标移动
        self.proxy = pg.SignalProxy(self.ax.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_move)
        
        # 监听鼠标点击 (用于右键菜单定位)
        self.last_click_scene_pos = None
        self.proxy_click = pg.SignalProxy(self.ax.scene().sigMouseClicked, slot=self.on_mouse_clicked)

        # 添加右键菜单动作
        self.sync_action = self.ax.vb.menu.addAction("Sync Time Center")
        self.sync_action.triggered.connect(self.on_sync_action_triggered)
        
        self.replay_start_action = self.ax.vb.menu.addAction("Set Replay Start")
        self.replay_start_action.triggered.connect(self.on_replay_start_action_triggered)

        # 缩放交互
        self.ax.setMouseEnabled(x=True, y=True)
        self.ax.getAxis('right').enableAutoSIPrefix(False)
        
        # 解除缩放和平移限制，允许无限拖动
        self.ax.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        
        # 增强网格可见性
        self.ax.getAxis('bottom').setGrid(100) # 0-255
        self.ax.getAxis('right').setGrid(100)

        # 还原黑色背景
        pg.setConfigOptions(foreground='#FFFFFF', background='#000000')
        self.glw.setBackground('k')
        
        self.current_period = "1min"
        self.plot_item = None 
        self.indicator_items = {} 
        self.full_df = None # 全量数据引用
        self.current_df = None # 当前切片数据 (View Slice)
        self.current_x = None
        self.current_time_values = None
        self.measure_active = False
        self.measure_start_y = None
        
        # 监听 Range 变化，用于动态切片加载
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.refresh_visible_view)
        self.ax.sigXRangeChanged.connect(self.on_range_changed)
        self._last_slice_start = -1
        self._last_slice_end = -1

    def on_range_changed(self):
        # 只有当用户拖动时才触发（Replay 也会触发，但我们需要它触发）
        # 延迟 20ms 更新，合并多次信号
        self.update_timer.start(20)

    def refresh_visible_view(self):
        if self.full_df is None or self.full_df.empty:
            return
            
        # 1. 获取当前视图范围
        view_range = self.ax.vb.viewRange()[0]
        min_x, max_x = view_range
        
        # 2. 计算需要加载的数据范围 (Padding)
        # 预读左右各 1000 根 (Buffer)
        padding = 1000
        slice_start = max(0, int(min_x) - padding)
        slice_end = min(len(self.full_df), int(max_x) + padding)
        
        # 3. 检查是否需要更新
        if self._last_slice_start != -1:
            if slice_start >= self._last_slice_start and slice_end <= self._last_slice_end:
                return

        # 4. 执行切片
        df_slice = self.full_df.iloc[slice_start:slice_end]
        if df_slice.empty:
            return
            
        self._last_slice_start = slice_start
        self._last_slice_end = slice_end
        
        # 5. 更新图表
        self.update_plot_items(df_slice, offset_x=slice_start)

    
    def update_plot_items(self, df, offset_x=0):
        # Candles
        x_data = np.arange(len(df), dtype=np.float64) + offset_x
        o_data = df['open'].to_numpy(dtype=np.float64)
        c_data = df['close'].to_numpy(dtype=np.float64)
        h_data = df['high'].to_numpy(dtype=np.float64)
        l_data = df['low'].to_numpy(dtype=np.float64)

        if self.plot_item is None:
            self.plot_item = CandlestickItem(x_data, o_data, c_data, h_data, l_data)
            self.ax.addItem(self.plot_item)
            self.ax.showGrid(True, True)
            self.ax.setLogMode(y=False)
            self.plot_item.setZValue(10)
        else:
            self.plot_item.set_data(x_data, o_data, c_data, h_data, l_data)

        # Indicators
        ema_colors = {
            'EMA20': '#FF0000', 'EMA30': '#FF8800', 'EMA40': '#FFFF00',
            'EMA50': '#00FF00', 'EMA60': '#0000FF'
        }

        for name, color in ema_colors.items():
            if name in df.columns:
                y_data = df[name].to_numpy(dtype=np.float64)

                if name not in self.indicator_items:
                    curve = pg.PlotCurveItem(
                        x=x_data, y=y_data,
                        pen=pg.mkPen(color, width=1.5),
                        name=name,
                        clipToView=True,
                        autoDownsample=True
                    )
                    self.ax.addItem(curve)
                    self.indicator_items[name] = curve
                else:
                    self.indicator_items[name].setData(x=x_data, y=y_data)

        # Bollinger Bands
        bb_color = '#FFFFFF'
        for name in ['BB_Upper', 'BB_Lower']:
            if name in df.columns:
                y_data = df[name].to_numpy(dtype=np.float64)

                if name not in self.indicator_items:
                    curve = pg.PlotCurveItem(
                        x=x_data, y=y_data,
                        pen=pg.mkPen(bb_color, width=1),
                        name=name,
                        clipToView=True,
                        autoDownsample=True
                    )
                    self.ax.addItem(curve)
                    self.indicator_items[name] = curve
                else:
                    self.indicator_items[name].setData(x=x_data, y=y_data)

        # View limits
        self.ax.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

    def on_mouse_move(self, evt):
        pos = evt[0]
        if self.ax.sceneBoundingRect().contains(pos):
            mousePoint = self.ax.vb.mapSceneToView(pos)

            mods = QApplication.keyboardModifiers()
            buttons = QApplication.mouseButtons()
            ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
            left_down = bool(buttons & Qt.MouseButton.LeftButton)
            if ctrl_down and left_down:
                if not self.measure_active:
                    self.measure_active = True
                    self.measure_start_y = mousePoint.y()
                diff = abs(mousePoint.y() - (self.measure_start_y or mousePoint.y()))
                self.txt_measure.setText(f"Δ {diff:.3f}")
                self.txt_measure.setPos(mousePoint.x(), mousePoint.y())
                self.txt_measure.show()
            else:
                if self.measure_active:
                    self.measure_active = False
                    self.measure_start_y = None
                    self.txt_measure.hide()
            
            # 1. 移动自己的水平线 (价格)
            self.hLine.setPos(mousePoint.y())
            # 移动垂直线 (时间) - 恢复平滑跟随鼠标，不强制吸附
            self.vLine.setPos(mousePoint.x())
            
            # 获取当前视图范围
            view_range = self.ax.vb.viewRange()
            x_min, x_max = view_range[0]
            y_min, y_max = view_range[1]
            
            # 更新价格标签
            self.txt_price.setText(f"{mousePoint.y():.4f}")
            self.txt_price.setPos(x_max, mousePoint.y())
            self.txt_price.setAnchor((1, 0.5))
            
            # 2. 时间标签处理
            if self.current_df is not None and self.current_x is not None and len(self.current_x) > 0:
                idx = int(round(mousePoint.x()))
                last_idx = len(self.current_x) - 1
                
                if 0 <= idx <= last_idx:
                    dt = self.current_df.index[idx]
                else:
                    # 外推时间
                    if idx > last_idx:
                        diff = idx - last_idx
                        base_dt = self.current_df.index[last_idx]
                    else: # idx < 0
                        diff = idx
                        base_dt = self.current_df.index[0]
                    
                    # 使用 time_axis 计算出的步长
                    delta = self.time_axis._delta if self.time_axis._delta else datetime.timedelta(minutes=1)
                    dt = base_dt + delta * diff

                dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                self.txt_time.setText(dt_str)
                
                # 位置跟随鼠标 (平滑)，而不是吸附到 idx
                self.txt_time.setPos(mousePoint.x(), y_min)
                self.txt_time.setAnchor((0.5, 1)) 
                
                # 发射信号 (如果是未来时间，也发射时间戳，以便其他窗口同步)
                # 使用 dt.value (纳秒) 转为秒，避免 naive datetime 的 timestamp() 时区问题
                ts_seconds = dt.value / 1e9
                self.sig_mouse_moved.emit(ts_seconds)
                self.sig_mouse_moved_with_price.emit(ts_seconds, mousePoint.y())
            else:
                self.txt_time.setText("")

    def on_mouse_clicked(self, evt):
        # 记录点击位置，供右键菜单使用
        event = evt[0]
        self.last_click_scene_pos = event.scenePos()

    def on_sync_action_triggered(self):
        if self.last_click_scene_pos is None or self.current_df is None:
            return
        
        # 将 Scene 坐标转换为 View 坐标 (X轴为 Index)
        mousePoint = self.ax.vb.mapSceneToView(self.last_click_scene_pos)
        dt = self.get_datetime_from_x(mousePoint.x())
        
        if dt:
            self.sig_sync_center_requested.emit(dt)

    def on_replay_start_action_triggered(self):
        if self.last_click_scene_pos is None or self.current_df is None:
            return
        
        mousePoint = self.ax.vb.mapSceneToView(self.last_click_scene_pos)
        dt = self.get_datetime_from_x(mousePoint.x())
        
        if dt:
            self.sig_set_replay_start.emit(dt)

    def get_datetime_from_x(self, x_val):
        """根据 X 轴坐标 (Index) 获取时间，支持外推"""
        if self.current_df is None or self.current_df.empty:
            return None
            
        idx = int(round(x_val))
        last_idx = len(self.current_df) - 1
        
        if 0 <= idx <= last_idx:
            return self.current_df.index[idx]
        
        # 外推
        if idx > last_idx:
            diff = idx - last_idx
            base_dt = self.current_df.index[last_idx]
        else:
            diff = idx
            base_dt = self.current_df.index[0]
            
        delta = self.time_axis._delta if self.time_axis._delta else datetime.timedelta(minutes=1)
        return base_dt + delta * diff

    def get_timestamp_from_x(self, x_val):
        """将 X 轴坐标转换为时间戳"""
        if self.current_x is None or len(self.current_x) == 0:
            return None
        return float(x_val)

    def sync_vline(self, timestamp):
        """接收外部时间戳，移动垂直线"""
        if self.current_time_values is None or len(self.current_time_values) == 0:
            return
        
        # 1. 尝试在范围内查找
        # current_time_values 是纳秒级 int64 (view) 转成的 float
        # timestamp 是 float (秒)
        # 需要统一单位。pandas view('int64') 是纳秒。timestamp 是秒。
        # 等等，之前的 current_time_values 已经是 float64 吗？
        # self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        # timestamp() 返回的是秒。
        # 这是一个严重的单位不匹配隐患，之前可能因为碰巧数值大没报错或者逻辑被掩盖。
        # 让我们检查 update_chart 中的赋值。
        
        # 在 update_chart: self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64) // 10**9 
        # 必须除以 10^9 才是秒。原代码漏了除法吗？
        # 原代码：self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        # 这就是纳秒。
        # 而 sig_mouse_moved 发射的是 dt.timestamp() (秒)。
        # searchsorted 会完全失效（总是返回 0 或 len）。
        
        # 既然现在我要重写 sync_vline，我必须确保存储的是秒，或者转换一下。
        # 考虑到性能，最好存秒。
        
        # 为了最小化改动风险，我会在 searchsorted 前转换 timestamp 为纳秒，或者...
        # 不，最好在 update_chart 里修正 current_time_values 的单位。
        # 但 update_chart 还没改。
        
        # 让我们先假定 current_time_values 是纳秒。
        ts_ns = timestamp * 1e9
        
        if len(self.current_time_values) > 0:
            first_ts = self.current_time_values[0]
            last_ts = self.current_time_values[-1]
            
            if first_ts <= ts_ns <= last_ts:
                # 范围内
                idx = int(np.searchsorted(self.current_time_values, ts_ns))
                self.vLine.setPos(idx)
            else:
                # 范围外，进行反算
                delta_ns = self.time_axis._delta.total_seconds() * 1e9
                if delta_ns > 0:
                    if ts_ns > last_ts:
                        diff = (ts_ns - last_ts) / delta_ns
                        idx = (len(self.current_time_values) - 1) + diff
                    else:
                        diff = (ts_ns - first_ts) / delta_ns
                        idx = diff # 负数
                    self.vLine.setPos(idx)

    def sync_crosshair(self, timestamp, price):
        self.sync_vline(timestamp)
        self.hLine.setPos(price)

    def on_btn_period_clicked(self, btn):
        period = btn.property("period")
        self.set_period(period)

    def set_period(self, period):
        self.current_period = period
        # 更新按钮状态
        display = self.display_map.get(period, period)
        for btn in self.btn_group.buttons():
            if btn.text() == display:
                btn.setChecked(True)
                break
        
        self.sig_period_changed.emit(display)

    def update_chart(self, df, auto_scale=False, highlight_idx=None):
        if df is None or df.empty:
            return

        # 性能优化：检查数据是否真的更新了
        current_last_ts = df.index[-1].timestamp()
        current_last_close = df['close'].iloc[-1]
        
        if hasattr(self, '_last_update_ts') and hasattr(self, '_last_update_len') and hasattr(self, '_last_update_close'):
            if (self._last_update_ts == current_last_ts and 
                self._last_update_len == len(df) and 
                self._last_update_close == current_last_close):
                if not auto_scale:
                    return
        
        self._last_update_ts = current_last_ts
        self._last_update_len = len(df)
        self._last_update_close = current_last_close

        # 保存全量数据引用
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        self.full_df = df
        self.current_df = df # 兼容旧逻辑
        
        # 数据源变更，重置切片缓存，确保 refresh_visible_view 能触发更新
        self._last_slice_start = -1
        self._last_slice_end = -1
        
        # 记录更新前的视图状态
        last_len = len(self.current_x) if self.current_x is not None else 0
        self.current_x = np.arange(len(df), dtype=np.float64)
        self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        
        # 调试：UI 层收到的数据检查
        print(f"UI update_chart received {len(df)} rows. First 3: {df.index[:3].tolist()}")
        
        self.time_axis.set_datetime_index(df.index)

        # 处理视图范围
        view_range = self.ax.vb.viewRange()
        view_right = view_range[0][1]
        is_following = (view_right >= last_len - 0.5)

        if len(df) > 0:
            if auto_scale:
                # 类似 TradingView：定位到目标索引（或末尾），显示约 150 根 K 线
                idx = highlight_idx if highlight_idx is not None else len(df) - 1
                x_start = max(0, idx - 150)
                x_end = idx + 20 # 预留右侧空白
                
                # 计算该范围内的 Y 轴
                visible_slice = df.iloc[int(x_start):int(x_end)]
                if not visible_slice.empty:
                    y_min = visible_slice['low'].min()
                    y_max = visible_slice['high'].max()
                    y_pad = (y_max - y_min) * 0.1
                    self.ax.setYRange(y_min - y_pad, y_max + y_pad, padding=0)
                
                self.ax.setXRange(x_start, x_end, padding=0)
            elif is_following and len(df) > last_len:
                diff = len(df) - last_len
                self.ax.vb.translateBy(x=diff, y=0)
                
                # 跟随模式下也自动调整 Y 轴 (最近 150 根)
                idx = len(df) - 1
                visible_slice = df.iloc[max(0, idx-150):idx+1]
                y_min = visible_slice['low'].min()
                y_max = visible_slice['high'].max()
                y_pad = (y_max - y_min) * 0.1
                self.ax.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

        # 触发首次渲染
        self.refresh_visible_view()
        
        if hasattr(self, '_custom_wheel_event'):
            self.ax.vb.wheelEvent = self._custom_wheel_event

    def on_detach_clicked(self):
        self.sig_detach_requested.emit(self)

    def set_detached_state(self, detached: bool):
        self.is_detached = detached
        self.btn_detach.setText("Dock" if detached else "Pop")


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
        
        # 更新标题
        self.chart_widget.sig_period_changed.connect(self.update_title)

    def update_title(self, period_display):
        self.setWindowTitle(f"Chart - {period_display}")

    def closeEvent(self, event):
        self.sig_window_closed.emit(self.chart_widget)
        event.accept()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Trade Review")
        self.resize(1400, 950)
        
        self.engine = DataEngine(parquet_file=None) 
        self.replay_engine = ReplayEngine(self.engine)
        self.current_time = datetime.datetime.now()
        self.is_playing = False
        self.replay_speed = 60 

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        self.create_control_panel()
        
        self.chart_container_layout = QVBoxLayout() 
        self.main_layout.addLayout(self.chart_container_layout)
        
        self.floating_windows = []
        self.charts = []
        self.tabs = None # 引用 tabs 组件
        self.init_charts()
        
        self.switch_layout("Tabs") # 默认 Tabs
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(100) 

    def _get_ticks_tz(self):
        if self.engine.df_ticks is None:
            return None
        return self.engine.df_ticks.index.tz

    def _normalize_time(self, dt):
        tz = self._get_ticks_tz()
        if tz is None:
            return pd.Timestamp(dt).tz_localize(None)
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts.tz_localize(tz)
        return ts.tz_convert(tz)

    def _set_date_edit(self, dt):
        if dt is None:
            return
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        self.date_edit.setDateTime(QDateTime(dt.year, dt.month, dt.day,
                                            dt.hour, dt.minute, dt.second))

    def _get_replay_periods(self):
        return [chart.current_period for chart in self.charts]

    def _ensure_replay_engine(self):
        if self.engine.df_ticks is None:
            return
        if not self.replay_engine.states:
            self.replay_engine.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )

    def _get_replay_max_count_map(self):
        max_count_map = {}
        for chart in self.charts:
            period = chart.current_period
            max_count_map[period] = self._get_max_count_for_period(period)
        return max_count_map

    def _get_max_count_for_period(self, period):
        tf = str(period).strip().lower()
        if tf.endswith("min"):
            try:
                minutes = int(tf[:-3])
            except ValueError:
                return 1000
            if minutes <= 5:
                return 4000
            if minutes <= 15:
                return 2500
            if minutes <= 60:
                return 1800
            return 1200
        if tf.endswith("h"):
            try:
                hours = int(tf[:-1])
            except ValueError:
                return 1000
            if hours <= 1:
                return 1500
            if hours <= 4:
                return 1000
            return 800
        return 800

    def _get_view_count_for_period(self, period):
        tf = str(period).strip().lower()
        if tf.endswith("min"):
            try:
                minutes = int(tf[:-3])
            except ValueError:
                return 300
            if minutes <= 5:
                return 800
            if minutes <= 15:
                return 600
            if minutes <= 60:
                return 400
            return 300
        if tf.endswith("h"):
            try:
                hours = int(tf[:-1])
            except ValueError:
                return 300
            if hours <= 1:
                return 400
            if hours <= 4:
                return 300
            return 200
        return 200

    def init_charts(self):
        # 初始只创建4个，默认给不同周期
        configs = [("1h", "1h"), ("15min", "15min"), ("5min", "5min"), ("1min", "1min")]
        for i, (display, period) in enumerate(configs):
            chart = ChartWidget(display)
            # 设置初始周期并触发加载
            chart.set_period(period)
            
            # 监听周期改变，触发重绘 + 更新 Tab 标题
            chart.sig_period_changed.connect(lambda p, c=chart: self.on_chart_period_changed(c, p))
            
            # 监听分离请求
            chart.sig_detach_requested.connect(self.toggle_chart_detach)
            
            # 监听同步请求
            chart.sig_sync_center_requested.connect(self.sync_charts_center)
            
            # 监听回放跳转请求
            chart.sig_set_replay_start.connect(self.set_replay_start_time)
            
            # 连接光标同步信号
            chart.sig_mouse_moved_with_price.connect(self.sync_all_charts_crosshair)
            self.charts.append(chart)

    def set_replay_start_time(self, target_dt):
        """跳转回放时间到指定点"""
        if self.engine.df_ticks is None:
            return

        # 如果当前不在回放模式，自动开启
        if not self.chk_replay.isChecked():
            self.chk_replay.setChecked(True)
            
        # 更新内部时间
        self.current_time = self._normalize_time(target_dt)
        if self.replay_engine is not None:
            self.replay_engine.reset(self.current_time)
        
        # 更新时间显示框
        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)
        
        # 强制刷新图表
        self.refresh_all_charts(auto_scale=True) # 跳转后通常希望自动聚焦

    def sync_charts_center(self, target_dt):
        """将所有图表视图中心对齐到指定时间"""
        for chart in self.charts:
            if chart.full_df is None or chart.full_df.empty:
                continue
                
            # 1. 找到目标时间对应的索引 (支持近似查找)
            # 由于可能存在数据缺口，使用 searchsorted 找到最近的插入点
            # 如果 target_dt 超出范围，需要特殊处理吗？ searchsorted 会返回 0 或 len
            
            # 为了更精确，使用 index 的 searchsorted
            idx = self.engine.df_ticks.index.searchsorted(self.current_time)
            
            # 2. 获取当前视图跨度 (保持缩放比例不变)
            view_range = chart.ax.vb.viewRange()[0]
            span = view_range[1] - view_range[0]
            
            # 3. 计算新的范围，使 idx 居中
            new_min = idx - span / 2
            new_max = idx + span / 2
            
            chart.ax.setXRange(new_min, new_max, padding=0)

    def on_chart_period_changed(self, chart, period_display):
        if self.chk_replay.isChecked() and self.engine.df_ticks is not None:
            self.replay_engine.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )

        # 1. 刷新数据
        self.refresh_single_chart(chart, auto_scale=True)
        
        # 2. 如果在 Tabs 模式，更新标题
        if self.combo_layout.currentText() == "Tabs" and self.tabs is not None:
            idx = self.tabs.indexOf(chart)
            if idx != -1:
                self.tabs.setTabText(idx, period_display)

    def sync_all_charts(self, timestamp):
        """同步所有图表的垂直光标"""
        for chart in self.charts:
            chart.sync_vline(timestamp)

    def sync_all_charts_crosshair(self, timestamp, price):
        for chart in self.charts:
            chart.sync_crosshair(timestamp, price)

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
        self.combo_layout.addItems(["Tabs", "Grid 2x2", "Vertical"]) # Tabs 放前面
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

        self.btn_step_back = QPushButton("Back")
        self.btn_step_back.clicked.connect(self.on_step_back)
        panel.addWidget(self.btn_step_back)

        self.btn_step_forward = QPushButton("Forward")
        self.btn_step_forward.clicked.connect(self.on_step_forward)
        panel.addWidget(self.btn_step_forward)

        self.combo_step = QComboBox()
        self.combo_step.addItems(["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D"])
        self.combo_step.setCurrentText("1h")
        panel.addWidget(self.combo_step)
        
        panel.addWidget(QLabel("Speed:"))
        
        self.speed_btn_group = QButtonGroup()
        self.speed_btn_group.setExclusive(True)
        
        # 定义常用倍速
        speeds = [1, 10, 60, 120, 300, 600]
        
        for s in speeds:
            btn = QPushButton(f"{s}x")
            btn.setCheckable(True)
            btn.setFixedSize(40, 25)
            
            # 默认选中 60x
            if s == 60:
                btn.setChecked(True)
                self.replay_speed = s
            
            btn.clicked.connect(lambda checked, val=s: self.set_speed(val))
            self.speed_btn_group.addButton(btn)
            panel.addWidget(btn)

        self.date_edit = QDateTimeEdit()
        self.date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._set_date_edit(self.current_time)
        panel.addWidget(self.date_edit)
        
        panel.addStretch()
        self.main_layout.addLayout(panel)

    def toggle_chart_detach(self, chart):
        if chart.is_detached:
            self.attach_chart(chart)
        else:
            self.detach_chart(chart)

    def detach_chart(self, chart):
        chart.set_detached_state(True)
        
        fw = FloatingChartWindow(chart)
        fw.sig_window_closed.connect(self.on_floating_window_closed)
        fw.show()
        self.floating_windows.append(fw)
        
        # Update main layout
        self.switch_layout(self.combo_layout.currentText())

    def attach_chart(self, chart):
        # Find window and close it. Logic handled in on_floating_window_closed
        for fw in self.floating_windows:
            if fw.chart_widget == chart:
                fw.close()
                return

    def on_floating_window_closed(self, chart):
        # Remove from list
        self.floating_windows = [fw for fw in self.floating_windows if fw.chart_widget != chart]
        
        # Ensure chart is marked attached
        chart.set_detached_state(False)
        
        # Re-integrate
        self.switch_layout(self.combo_layout.currentText())

    def switch_layout(self, layout_name):
        active_charts = [c for c in self.charts if not c.is_detached]
        for chart in active_charts:
            if chart.parent() is not None:
                chart.setParent(None)

        while self.chart_container_layout.count():
            item = self.chart_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None) 
        
        self.tabs = None
        
        if layout_name == "Vertical":
            splitter = QSplitter(Qt.Orientation.Vertical)
            for chart in active_charts:
                splitter.addWidget(chart)
                chart.show()
            self.chart_container_layout.addWidget(splitter)
            
        elif layout_name == "Grid 2x2":
            grid_widget = QWidget()
            grid = QGridLayout()
            grid_widget.setLayout(grid)
            
            # Simple grid logic for dynamic number of charts
            for i, chart in enumerate(active_charts):
                row = i // 2
                col = i % 2
                grid.addWidget(chart, row, col)
                chart.show()
            
            self.chart_container_layout.addWidget(grid_widget)
            
        elif layout_name == "Tabs":
            self.tabs = QTabWidget()
            for chart in active_charts:
                # 获取当前显示的周期名作为标题
                title = chart.display_map.get(chart.current_period, chart.current_period)
                self.tabs.addTab(chart, title)
                chart.show()
            self.chart_container_layout.addWidget(self.tabs)
            
        # 切换布局仅重排容器，不强制重采样/重置视图

    def load_data_file(self, file_path):
        self.engine.parquet_file = file_path
        self.engine.load_data()
        if self.engine.df_ticks is not None:
            total_ticks = len(self.engine.df_ticks)
            if total_ticks > 100000:
                self.current_time = self.engine.df_ticks.index[100000]
            else:
                self.current_time = self.engine.df_ticks.index[0]

            if hasattr(self, 'date_edit'):
                self._set_date_edit(self.current_time)
            # 强制重置视图，确保K线居中显示
            self.reset_charts_view()
            if hasattr(self, 'replay_engine'):
                self.replay_engine.initialize(
                    self._get_replay_periods(),
                    self.current_time,
                    max_count_map=self._get_replay_max_count_map(),
                )

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Parquet Data", "", "Parquet Files (*.parquet);;All Files (*)")
        if file_name:
            self.load_data_file(file_name)

    def _get_step_delta(self):
        if not hasattr(self, "combo_step"):
            return None
        text = self.combo_step.currentText().strip()
        try:
            return pd.Timedelta(text)
        except Exception:
            return None

    def _apply_time_jump(self, delta):
        if delta is None:
            return
        new_time = self._normalize_time(self.current_time) + delta
        if self.engine.df_ticks is not None and not self.engine.df_ticks.empty:
            start = self.engine.df_ticks.index[0]
            end = self.engine.df_ticks.index[-1]
            if new_time < start:
                new_time = start
            if new_time > end:
                new_time = end

        self.current_time = new_time
        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)
        if self.chk_replay.isChecked():
            self.replay_engine.reset(self.current_time)
        self.refresh_all_charts(auto_scale=True)

    def on_step_back(self):
        delta = self._get_step_delta()
        if delta is not None:
            self._apply_time_jump(-delta)

    def on_step_forward(self):
        delta = self._get_step_delta()
        if delta is not None:
            self._apply_time_jump(delta)

    def on_mode_change(self, state):
        is_replay = self.chk_replay.isChecked()
        self.btn_play.setEnabled(is_replay)
        if is_replay:
            self._ensure_replay_engine()
        self.refresh_all_charts(auto_scale=True)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def set_speed(self, speed):
        self.replay_speed = speed

    def reset_charts_view(self):
        self.refresh_all_charts(auto_scale=True)

    def refresh_single_chart(self, chart, auto_scale=False):
        if self.engine.df_ticks is None:
            return
        
        target_idx = None
        if self.chk_replay.isChecked():
            self._ensure_replay_engine()
            view_count = self._get_view_count_for_period(chart.current_period)
            df = self.replay_engine.get_view(chart.current_period, count=view_count, with_indicators=True)
            if df is not None:
                target_idx = len(df) - 1
        else:
            df = self.engine.get_candles(chart.current_period) 
            # 全量模式下，根据当前时间找到对应的 index
            if df is not None and not df.empty:
                search_time = self.current_time
                # 如果 K 线索引是 Naive (因为 NY Close 转换)，而 current_time 是 Aware，则去除时区
                if df.index.tz is None and search_time.tzinfo is not None:
                    search_time = search_time.replace(tzinfo=None)
                
                target_idx = df.index.searchsorted(search_time)
                
        chart.update_chart(df, auto_scale=auto_scale, highlight_idx=target_idx)

    def refresh_all_charts(self, auto_scale=False):
        for chart in self.charts:
            self.refresh_single_chart(chart, auto_scale=auto_scale)

    def on_timer_tick(self):
        if not self.chk_replay.isChecked() or not self.is_playing:
            return
        
        current_ts = self._normalize_time(self.current_time)
        target_time = current_ts + pd.Timedelta(seconds=self.replay_speed)

        self._ensure_replay_engine()
        actual_time = self.replay_engine.advance_to(target_time)
        if actual_time is None:
            return
        self.current_time = actual_time

        if self.engine.df_ticks is not None:
            idx = min(self.replay_engine.tick_pos, len(self.engine.df_ticks) - 1)
            if idx >= len(self.engine.df_ticks) - 1:
                self.toggle_play()

        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)
        self.refresh_all_charts()
