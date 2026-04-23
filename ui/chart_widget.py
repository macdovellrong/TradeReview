import datetime
import os
from functools import partial

import finplot as fplt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
from ui.chart_performance import build_visible_slice_window, should_refresh_visible_slice
from ui.drawings.dialogs import FibConfigDialog
from ui.drawings.fib_config import default_fib_settings, load_fib_settings, save_fib_settings
from ui.drawings.renderers import render_spec_items
from ui.drawings.specs import normalize_drawing_spec
from ui.drawings.tools import DrawingSession, TOOL_DEFINITIONS
from ui.time_navigation import clamp_timestamp, normalize_jump_timestamp, resolve_chart_target

class ChartWidget(QWidget):
    # 瀹氫箟淇″彿锛氶紶鏍囩Щ鍔ㄦ椂鍙戝皠褰撳墠鐨勬椂闂存埑 (float)
    sig_mouse_moved = pyqtSignal(float)
    # 瀹氫箟淇″彿锛氶紶鏍囩Щ鍔ㄦ椂鍙戝皠鏃堕棿鎴充笌浠锋牸
    sig_mouse_moved_with_price = pyqtSignal(float, float)
    # 缁樺浘璇锋眰/鍒犻櫎/娓呯┖
    sig_drawing_request = pyqtSignal(object)
    sig_drawing_delete_request = pyqtSignal(int)
    sig_drawing_clear_request = pyqtSignal()
    sig_fib_config_requested = pyqtSignal()
    # 淇″彿锛氬懆鏈熸敼鍙樻椂鍙戝皠 (str)
    sig_period_changed = pyqtSignal(str)
    # 淇″彿锛氳姹傚垎绂?杩樺師
    sig_detach_requested = pyqtSignal(object)
    # 淇″彿锛氳姹傚悓姝ユ墍鏈夊浘琛ㄤ腑蹇冪偣 (datetime)
    sig_sync_center_requested = pyqtSignal(object)
    sig_sync_y_center_requested = pyqtSignal(float)
    # 淇″彿锛氳缃洖鏀惧紑濮嬫椂闂?(datetime)
    sig_set_replay_start = pyqtSignal(object)

    def __init__(self, name="Chart", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        
        # 椤堕儴宸ュ叿鏉?(鍛ㄦ湡閫夋嫨)
        self.toolbar_layout = QHBoxLayout()
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.toolbar_layout.setSpacing(0)
        
        # 浣跨敤 ScrollArea 鏉ュ绾充紬澶氭寜閽?
        scroll = QScrollArea()
        scroll.setFixedHeight(40) # 鍥哄畾楂樺害
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
        
        # 瀹氫箟鍛ㄦ湡閫夐」鍙婂叾鏄剧ず鏂囨湰
        periods = [
            ("30s", "30s"),
            ("1min", "1m"), ("2min", "2m"), ("3min", "3m"), ("5min", "5m"),
            ("10min", "10m"), ("15min", "15m"), ("20min", "20m"), ("30min", "30m"), ("45min", "45m"), ("90min", "90m"),
            ("1h", "1h"), ("2h", "2h"), ("3h", "3h"), ("4h", "4h"),
            ("6h", "6h"), ("8h", "8h"), ("12h", "12h"),
            ("1D", "1D"), ("1W", "1W"), ("1M", "1M")
        ]
        
        self.period_map = {p[1]: p[0] for p in periods} # display -> actual
        self.display_map = {p[0]: p[1] for p in periods} # actual -> display
        
        for actual, display in periods:
            btn = QPushButton(display)
            btn.setCheckable(True)
            btn.setFixedSize(40, 30) # 灏忔寜閽?
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
            
            # 瀛樺偍瀹為檯鍛ㄦ湡鍊?
            btn.setProperty("period", actual)
            
        scroll_layout.addStretch()
        self.toolbar_layout.addWidget(scroll)

        # EMA鏄剧ず鍒嗙粍锛堝彲澶氶€夛級
        self.ema_toggle_buttons = {}
        self.btn_toggle_bb = None
        self.btn_toggle_macd_rsi = None
        for span in [20, 30, 40, 50, 60, 100, 240]:
            name = f"EMA{span}"
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setFixedSize(68, 30)
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
            btn.toggled.connect(self.on_indicator_toggle_changed)
            self.toolbar_layout.addWidget(btn)
            self.ema_toggle_buttons[name] = btn
        # 淇濇寔鏃ц涓猴細榛樿鏄剧ずEMA20-60
        for name in ["EMA20", "EMA30", "EMA40", "EMA50", "EMA60"]:
            btn = self.ema_toggle_buttons[name]
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)

        self.btn_toggle_bb = QPushButton("BB")
        self.btn_toggle_bb.setCheckable(True)
        self.btn_toggle_bb.setChecked(True)
        self.btn_toggle_bb.setFixedSize(44, 30)
        self.btn_toggle_bb.setStyleSheet("""
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
        self.btn_toggle_bb.toggled.connect(self.on_indicator_toggle_changed)
        self.toolbar_layout.addWidget(self.btn_toggle_bb)

        self.btn_toggle_macd_rsi = QPushButton("MACD/RSI")
        self.btn_toggle_macd_rsi.setCheckable(True)
        self.btn_toggle_macd_rsi.setChecked(True)
        self.btn_toggle_macd_rsi.setFixedSize(86, 30)
        self.btn_toggle_macd_rsi.setStyleSheet("""
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
        self.btn_toggle_macd_rsi.toggled.connect(self.on_indicator_panel_toggle_changed)
        self.toolbar_layout.addWidget(self.btn_toggle_macd_rsi)

        # 缁樺浘宸ュ叿鎸夐挳
        self.btn_draw_select = QPushButton("Sel")
        self.btn_draw_hline = QPushButton("H")
        self.btn_draw_vline = QPushButton("V")
        self.btn_draw_line = QPushButton("Line")
        self.btn_draw_fib = QPushButton("Fib")
        self.btn_draw_fib_ext = QPushButton("Fib Ext")
        self.btn_draw_fib_config = QPushButton("Fib Config")
        self.btn_draw_clear = QPushButton("Clear")
        for btn in [self.btn_draw_select, self.btn_draw_hline, self.btn_draw_vline,
                    self.btn_draw_line, self.btn_draw_fib, self.btn_draw_fib_ext,
                    self.btn_draw_fib_config, self.btn_draw_clear]:
            width = 44
            if btn in (self.btn_draw_fib_ext, self.btn_draw_fib_config):
                width = 76
            btn.setFixedSize(width, 30)
            btn.setStyleSheet("""
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

        self.btn_draw_select.clicked.connect(lambda: self.set_draw_mode(None))
        self.btn_draw_hline.clicked.connect(lambda: self.set_draw_mode("hline"))
        self.btn_draw_vline.clicked.connect(lambda: self.set_draw_mode("vline"))
        self.btn_draw_line.clicked.connect(lambda: self.set_draw_mode("line"))
        self.btn_draw_fib.clicked.connect(lambda: self.set_draw_mode("fib"))
        self.btn_draw_fib_ext.clicked.connect(lambda: self.set_draw_mode("fib_ext"))
        self.btn_draw_fib_config.clicked.connect(self.on_open_fib_config)
        self.btn_draw_clear.clicked.connect(self.on_clear_drawings)

        self.toolbar_layout.addWidget(self.btn_draw_select)
        self.toolbar_layout.addWidget(self.btn_draw_hline)
        self.toolbar_layout.addWidget(self.btn_draw_vline)
        self.toolbar_layout.addWidget(self.btn_draw_line)
        self.toolbar_layout.addWidget(self.btn_draw_fib)
        self.toolbar_layout.addWidget(self.btn_draw_fib_ext)
        self.toolbar_layout.addWidget(self.btn_draw_fib_config)
        self.toolbar_layout.addWidget(self.btn_draw_clear)

        # 鍒嗙鎸夐挳
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
        
        # Finplot 鐢诲竷宓屽叆閫昏緫
        self.glw = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.glw)
        
        self.time_axis = TimeAxisItem(orientation='bottom')
        self.ax = self.glw.addPlot(axisItems={'bottom': self.time_axis})
        self.glw.nextRow()
        self.ax_macd = self.glw.addPlot()
        self.glw.nextRow()
        self.ax_rsi = self.glw.addPlot()
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

        # 鑷畾涔夋粴杞€昏緫
        def custom_wheel_event(ev, axis=None):
            # 1. 鑾峰彇婊氬姩澧為噺
            if hasattr(ev, 'angleDelta'):
                delta = ev.angleDelta().y()
            else:
                delta = ev.delta()

            if delta == 0:
                return
                
            # 2. 璁＄畻缂╂斁绯绘暟 (鍊艰秺鎺ヨ繎 1.0锛岀缉鏀捐秺鎱?
            s_x = 0.92 ** (delta / 120.0)
            s_y = 0.97 ** (delta / 120.0)
            
            try:
                # 3. 鑾峰彇榧犳爣浣嶇疆 (Scene鍧愭爣)
                pos = ev.scenePos()
                
                # 4. 鍒ゅ畾榧犳爣鎵€鍦ㄥ尯鍩?
                rect_x = self.ax.getAxis('bottom').sceneBoundingRect()
                rect_y = self.ax.getAxis('right').sceneBoundingRect()
                rect_plot = self.ax.vb.sceneBoundingRect()
                
                # 5. 璁＄畻缂╂斁涓績 (鏁版嵁鍧愭爣绯?- 蹇呴』浣跨敤瀹為檯浠锋牸/鏃堕棿鍊?
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
        
        # 4. 璁剧疆涓€浜涘熀纭€灞炴€?
        self.ax.showGrid(x=True, y=True)
        self.ax.showAxis('right', show=True)
        self.ax.showAxis('left', show=False)
        self.ax.getAxis('right').setWidth(60)

        self.ax_macd.showGrid(x=True, y=True)
        self.ax_macd.showAxis('right', show=True)
        self.ax_macd.showAxis('left', show=False)
        self.ax_macd.getAxis('right').setWidth(60)
        self.ax_macd.showAxis('bottom', show=False)
        self.ax_macd.setXLink(self.ax)
        self.ax_macd.setMaximumHeight(140)

        self.ax_rsi.showGrid(x=True, y=True)
        self.ax_rsi.showAxis('right', show=True)
        self.ax_rsi.showAxis('left', show=False)
        self.ax_rsi.getAxis('right').setWidth(60)
        self.ax_rsi.setXLink(self.ax)
        self.ax_rsi.setMaximumHeight(120)
        self._apply_indicator_panel_visibility()

        # 鍗佸瓧鍏夋爣 (Crosshair)
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.vLine.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        
        self.hLine = pg.InfiniteLine(angle=0, movable=False)
        self.hLine.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        
        # 鏍囩 (Labels) - 鏀圭敤 TextItem 浠ヤ究绮剧‘瀹氫綅
        self.txt_price = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(1, 1))
        self.txt_price.setZValue(20) # 纭繚鍦ㄦ渶涓婂眰
        self.ax.addItem(self.txt_price, ignoreBounds=True)

        self.txt_time = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(0.5, 1))
        self.txt_time.setZValue(20)
        self.ax.addItem(self.txt_time, ignoreBounds=True)

        # 浠峰樊娴嬬畻鎻愮ず
        self.txt_measure = pg.TextItem(text="", color='#FFFFFF', fill='#333333', anchor=(0, 1))
        self.txt_measure.setZValue(20)
        self.txt_measure.hide()
        self.ax.addItem(self.txt_measure, ignoreBounds=True)

        # K绾夸俊鎭彁绀?(宸︿笂瑙?
        self.txt_kinfo = pg.TextItem(text="", color='#FFFFFF', fill='#222222', anchor=(0, 0))
        self.txt_kinfo.setZValue(20)
        self.ax.addItem(self.txt_kinfo, ignoreBounds=True)
        
        self.ax.addItem(self.vLine, ignoreBounds=True)
        self.ax.addItem(self.hLine, ignoreBounds=True)

        # MACD/RSI vertical cursor lines
        self.vLine_macd = pg.InfiniteLine(angle=90, movable=False)
        self.vLine_macd.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        self.ax_macd.addItem(self.vLine_macd, ignoreBounds=True)

        self.vLine_rsi = pg.InfiniteLine(angle=90, movable=False)
        self.vLine_rsi.setPen(pg.mkPen(color='#FFFFFF', style=Qt.PenStyle.DashLine, width=1))
        self.ax_rsi.addItem(self.vLine_rsi, ignoreBounds=True)
        
        # 鐩戝惉榧犳爣绉诲姩
        self.proxy = pg.SignalProxy(self.ax.scene().sigMouseMoved, rateLimit=60, slot=self.on_mouse_move)
        
        # 鐩戝惉榧犳爣鐐瑰嚮 (鐢ㄤ簬鍙抽敭鑿滃崟瀹氫綅)
        self.last_click_scene_pos = None
        self.proxy_click = pg.SignalProxy(self.ax.scene().sigMouseClicked, slot=self.on_mouse_clicked)

        # 娣诲姞鍙抽敭鑿滃崟鍔ㄤ綔
        self.sync_action = self.ax.vb.menu.addAction("Sync Time Center")
        self.sync_action.triggered.connect(self.on_sync_action_triggered)
        self.sync_y_action = self.ax.vb.menu.addAction("Sync Y Center")
        self.sync_y_action.triggered.connect(self.on_sync_y_action_triggered)
        
        
        self.replay_start_action = self.ax.vb.menu.addAction("Set Replay Start")
        self.replay_start_action.triggered.connect(self.on_replay_start_action_triggered)

        self.delete_drawing_action = self.ax.vb.menu.addAction("Delete Drawing")
        self.delete_drawing_action.triggered.connect(self.on_delete_drawing_action)

        self.clear_drawings_action = self.ax.vb.menu.addAction("Clear Drawings")
        self.clear_drawings_action.triggered.connect(self.on_clear_drawings)

        # 缂╂斁浜や簰
        self.ax.setMouseEnabled(x=True, y=True)
        self.ax.getAxis('right').enableAutoSIPrefix(False)
        
        # 瑙ｉ櫎缂╂斁鍜屽钩绉婚檺鍒讹紝鍏佽鏃犻檺鎷栧姩
        self.ax.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        
        # 澧炲己缃戞牸鍙鎬?
        self.ax.getAxis('bottom').setGrid(100) # 0-255
        self.ax.getAxis('right').setGrid(100)

        # 杩樺師榛戣壊鑳屾櫙
        pg.setConfigOptions(foreground='#FFFFFF', background='#000000')
        self.glw.setBackground('k')
        
        self.current_period = "1min"
        self.plot_item = None 
        self.indicator_items = {}
        self.macd_items = {}
        self.rsi_items = {}
        self.full_df = None # 鍏ㄩ噺鏁版嵁寮曠敤
        self.current_df = None # 褰撳墠鍒囩墖鏁版嵁 (View Slice)
        self.current_x = None
        self.current_time_values = None
        self.measure_active = False
        self.measure_start_y = None
        self.fib_settings = default_fib_settings()
        self.draw_mode = None
        self.active_drawing_session = None
        self.draw_preview_items = []
        self.drawings = {}
        self.selected_drawing_id = None
        
        # 鐩戝惉 Range 鍙樺寲锛岀敤浜庡姩鎬佸垏鐗囧姞杞?
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.refresh_visible_view)
        self.ax.sigXRangeChanged.connect(self.on_range_changed)
        self._last_slice_start = -1
        self._last_slice_end = -1
        self._slice_padding = 1000

    def on_range_changed(self):
        # 鍙湁褰撶敤鎴锋嫋鍔ㄦ椂鎵嶈Е鍙戯紙Replay 涔熶細瑙﹀彂锛屼絾鎴戜滑闇€瑕佸畠瑙﹀彂锛?
        # 寤惰繜 20ms 鏇存柊锛屽悎骞跺娆′俊鍙?
        if self.full_df is None or self.full_df.empty:
            return

        min_x, max_x = self.ax.vb.viewRange()[0]
        if not should_refresh_visible_slice(
            view_min=min_x,
            view_max=max_x,
            total_len=len(self.full_df),
            last_slice_start=self._last_slice_start,
            last_slice_end=self._last_slice_end,
            padding=self._slice_padding,
        ):
            return

        self.update_timer.start(20)

    def refresh_visible_view(self, force=False):
        if self.full_df is None or self.full_df.empty:
            return
            
        # 1. 鑾峰彇褰撳墠瑙嗗浘鑼冨洿
        view_range = self.ax.vb.viewRange()[0]
        min_x, max_x = view_range
        
        # 2. 璁＄畻闇€瑕佸姞杞界殑鏁版嵁鑼冨洿 (Padding)
        # 棰勮宸﹀彸鍚?1000 鏍?(Buffer)
        if not force and not should_refresh_visible_slice(
            view_min=min_x,
            view_max=max_x,
            total_len=len(self.full_df),
            last_slice_start=self._last_slice_start,
            last_slice_end=self._last_slice_end,
            padding=self._slice_padding,
        ):
            return

        slice_start, slice_end = build_visible_slice_window(
            view_min=min_x,
            view_max=max_x,
            total_len=len(self.full_df),
            padding=self._slice_padding,
        )
        
        # 3. 妫€鏌ユ槸鍚﹂渶瑕佹洿鏂?

        # 4. 鎵ц鍒囩墖
        df_slice = self.full_df.iloc[slice_start:slice_end]
        if df_slice.empty:
            return
            
        self._last_slice_start = slice_start
        self._last_slice_end = slice_end
        
        # 5. 鏇存柊鍥捐〃
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
            'EMA50': '#00FF00', 'EMA60': '#0000FF',
            'EMA100': '#00BFFF', 'EMA240': '#FF66CC'
        }

        for name, color in ema_colors.items():
            enabled = self._is_ema_enabled(name)
            if enabled and name in df.columns:
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
                    self.indicator_items[name].setVisible(True)
            elif name in self.indicator_items:
                self.indicator_items[name].setVisible(False)

        # Bollinger Bands
        bb_color = '#FFFFFF'
        for name in ['BB_Upper', 'BB_Lower']:
            if self._is_bollinger_enabled() and name in df.columns:
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
                    self.indicator_items[name].setVisible(True)
            elif name in self.indicator_items:
                self.indicator_items[name].setVisible(False)

        # MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].to_numpy(dtype=np.float64)
            signal = df['MACD_Signal'].to_numpy(dtype=np.float64)
            hist = df.get('MACD_Hist')
            hist_data = hist.to_numpy(dtype=np.float64) if hist is not None else None

            if 'MACD' not in self.macd_items:
                macd_curve = pg.PlotCurveItem(x=x_data, y=macd, pen=pg.mkPen('#00FFFF', width=1.2))
                signal_curve = pg.PlotCurveItem(x=x_data, y=signal, pen=pg.mkPen('#FFAA00', width=1.2))
                self.ax_macd.addItem(macd_curve)
                self.ax_macd.addItem(signal_curve)
                self.macd_items['MACD'] = macd_curve
                self.macd_items['MACD_Signal'] = signal_curve
            else:
                self.macd_items['MACD'].setData(x=x_data, y=macd)
                self.macd_items['MACD_Signal'].setData(x=x_data, y=signal)

            if hist_data is not None:
                # Remove legacy single-color histogram if present
                if 'MACD_Hist' in self.macd_items:
                    self.ax_macd.removeItem(self.macd_items['MACD_Hist'])
                    del self.macd_items['MACD_Hist']

                pos = np.where(hist_data > 0, hist_data, 0)
                neg = np.where(hist_data < 0, hist_data, 0)

                if 'MACD_Hist_Pos' not in self.macd_items:
                    hist_pos = pg.BarGraphItem(x=x_data, height=pos, width=0.6, brush=pg.mkBrush('#FF3333'))
                    hist_neg = pg.BarGraphItem(x=x_data, height=neg, width=0.6, brush=pg.mkBrush('#66CCFF'))
                    self.ax_macd.addItem(hist_pos)
                    self.ax_macd.addItem(hist_neg)
                    self.macd_items['MACD_Hist_Pos'] = hist_pos
                    self.macd_items['MACD_Hist_Neg'] = hist_neg
                else:
                    self.macd_items['MACD_Hist_Pos'].setOpts(x=x_data, height=pos)
                    self.macd_items['MACD_Hist_Neg'].setOpts(x=x_data, height=neg)

        # RSI
        rsi_defs = [
            ("RSI6", "#66FF66"),
            ("RSI12", "#FFD24D"),
            ("RSI24", "#66A3FF"),
        ]
        has_any_rsi = False
        for name, color in rsi_defs:
            if name in df.columns:
                has_any_rsi = True
                rsi = df[name].to_numpy(dtype=np.float64)
                if name not in self.rsi_items:
                    rsi_curve = pg.PlotCurveItem(x=x_data, y=rsi, pen=pg.mkPen(color, width=1.2))
                    self.ax_rsi.addItem(rsi_curve)
                    self.rsi_items[name] = rsi_curve
                else:
                    self.rsi_items[name].setData(x=x_data, y=rsi)

        if has_any_rsi:
            if 'RSI_20' not in self.rsi_items:
                line20 = pg.InfiniteLine(angle=0, pos=20, pen=pg.mkPen('#444444', width=1, style=Qt.PenStyle.DashLine))
                line80 = pg.InfiniteLine(angle=0, pos=80, pen=pg.mkPen('#444444', width=1, style=Qt.PenStyle.DashLine))
                self.ax_rsi.addItem(line20)
                self.ax_rsi.addItem(line80)
                self.rsi_items['RSI_20'] = line20
                self.rsi_items['RSI_80'] = line80
            self.ax_rsi.setYRange(0, 100, padding=0)
        else:
            # clear stale RSI drawings if new data lacks RSI columns
            if self.rsi_items:
                for item in self.rsi_items.values():
                    try:
                        self.ax_rsi.removeItem(item)
                    except Exception:
                        pass
                self.rsi_items = {}

        # View limits
        self.ax.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.ax_macd.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
        self.ax_rsi.vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

    def on_mouse_move(self, evt):
        pos = evt[0]
        if self.ax.sceneBoundingRect().contains(pos):
            mousePoint = self.ax.vb.mapSceneToView(pos)

            # 缁樺浘棰勮
            if self.active_drawing_session is not None:
                self._clear_preview()
                preview_spec = self.active_drawing_session.build_preview_spec(
                    self.get_datetime_from_x(mousePoint.x()),
                    float(mousePoint.y()),
                )
                if preview_spec is not None:
                    self.draw_preview_items = render_spec_items(
                        self.ax,
                        preview_spec,
                        self._x_from_datetime,
                        preview=True,
                    )

            mods = QApplication.keyboardModifiers()
            buttons = QApplication.mouseButtons()
            ctrl_down = bool(mods & Qt.KeyboardModifier.ControlModifier)
            left_down = bool(buttons & Qt.MouseButton.LeftButton)
            if ctrl_down and left_down:
                if not self.measure_active:
                    self.measure_active = True
                    self.measure_start_y = mousePoint.y()
                diff = abs(mousePoint.y() - (self.measure_start_y or mousePoint.y()))
                self.txt_measure.setText(f"螖 {diff:.3f}")
                self.txt_measure.setPos(mousePoint.x(), mousePoint.y())
                self.txt_measure.show()
            else:
                if self.measure_active:
                    self.measure_active = False
                    self.measure_start_y = None
                    self.txt_measure.hide()
            
            # 1. 绉诲姩鑷繁鐨勬按骞崇嚎 (浠锋牸)
            is_dragging_view = left_down and not ctrl_down and self.draw_mode is None
            if is_dragging_view:
                self.txt_time.setText("")
                self.txt_kinfo.setHtml("")
                return

            self.hLine.setPos(mousePoint.y())
            # 绉诲姩鍨傜洿绾?(鏃堕棿) - 鎭㈠骞虫粦璺熼殢榧犳爣锛屼笉寮哄埗鍚搁檮
            self.vLine.setPos(mousePoint.x())
            self.vLine_macd.setPos(mousePoint.x())
            self.vLine_rsi.setPos(mousePoint.x())
            
            # 鑾峰彇褰撳墠瑙嗗浘鑼冨洿
            view_range = self.ax.vb.viewRange()
            x_min, x_max = view_range[0]
            y_min, y_max = view_range[1]
            
            # 鏇存柊浠锋牸鏍囩
            self.txt_price.setText(f"{mousePoint.y():.4f}")
            self.txt_price.setPos(x_max, mousePoint.y())
            self.txt_price.setAnchor((1, 0.5))
            
            # 2. 鏃堕棿鏍囩澶勭悊
            if self.current_df is not None and self.current_x is not None and len(self.current_x) > 0:
                idx = int(round(mousePoint.x()))
                last_idx = len(self.current_x) - 1
                
                if 0 <= idx <= last_idx:
                    dt = self.current_df.index[idx]
                else:
                    # 澶栨帹鏃堕棿
                    if idx > last_idx:
                        diff = idx - last_idx
                        base_dt = self.current_df.index[last_idx]
                    else: # idx < 0
                        diff = idx
                        base_dt = self.current_df.index[0]
                    
                    # 浣跨敤 time_axis 璁＄畻鍑虹殑姝ラ暱
                    delta = self.time_axis._delta if self.time_axis._delta else datetime.timedelta(minutes=1)
                    dt = base_dt + delta * diff

                dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                self.txt_time.setText(dt_str)
                
                # 浣嶇疆璺熼殢榧犳爣 (骞虫粦)锛岃€屼笉鏄惛闄勫埌 idx
                self.txt_time.setPos(mousePoint.x(), y_min)
                self.txt_time.setAnchor((0.5, 1))

                # K绾夸俊鎭?(寮€楂樹綆鏀?+ RSI6/12/24)
                if 0 <= idx <= last_idx:
                    row = self.current_df.iloc[idx]
                    o = row.get('open', np.nan)
                    h = row.get('high', np.nan)
                    l = row.get('low', np.nan)
                    c = row.get('close', np.nan)
                    rsi6 = row.get('RSI6', np.nan)
                    rsi12 = row.get('RSI12', np.nan)
                    rsi24 = row.get('RSI24', np.nan)
                    info = (
                        f"O {o:.3f}  H {h:.3f}  L {l:.3f}  C {c:.3f}  "
                        f"<span style='color:#66FF66'>RSI6 {rsi6:.3f}</span>  "
                        f"<span style='color:#FFD24D'>RSI12 {rsi12:.3f}</span>  "
                        f"<span style='color:#66A3FF'>RSI24 {rsi24:.3f}</span>"
                    )
                    self.txt_kinfo.setHtml(info)
                    self.txt_kinfo.setPos(x_min, y_max)

                # 鍙戝皠淇″彿 (濡傛灉鏄湭鏉ユ椂闂达紝涔熷彂灏勬椂闂存埑锛屼互渚垮叾浠栫獥鍙ｅ悓姝?
                # 浣跨敤 dt.value (绾崇) 杞负绉掞紝閬垮厤 naive datetime 鐨?timestamp() 鏃跺尯闂
                ts_seconds = dt.value / 1e9
                self.sig_mouse_moved.emit(ts_seconds)
                self.sig_mouse_moved_with_price.emit(ts_seconds, mousePoint.y())
            else:
                self.txt_time.setText("")

    def on_mouse_clicked(self, evt):
        # 璁板綍鐐瑰嚮浣嶇疆锛屼緵鍙抽敭鑿滃崟浣跨敤
        event = evt[0]
        self.last_click_scene_pos = event.scenePos()

        # 閫夋嫨缁樺浘瀵硅薄
        items = self.ax.scene().items(event.scenePos())
        sel_id = None
        for it in items:
            if getattr(it, "_is_drawing", False):
                sel_id = getattr(it, "_drawing_id", None)
                if sel_id is not None:
                    break
        self.selected_drawing_id = sel_id

        if event.button() == Qt.MouseButton.LeftButton:
            if self.draw_mode:
                self._handle_draw_click(event.scenePos())

    def on_sync_action_triggered(self):
        if self.last_click_scene_pos is None or self.current_df is None:
            return
        
        # 灏?Scene 鍧愭爣杞崲涓?View 鍧愭爣 (X杞翠负 Index)
        mousePoint = self.ax.vb.mapSceneToView(self.last_click_scene_pos)
        dt = self.get_datetime_from_x(mousePoint.x())
        
        if dt:
            self.sig_sync_center_requested.emit(dt)

    def on_sync_y_action_triggered(self):
        if self.last_click_scene_pos is None:
            return

        mousePoint = self.ax.vb.mapSceneToView(self.last_click_scene_pos)
        self.sig_sync_y_center_requested.emit(float(mousePoint.y()))

    def on_replay_start_action_triggered(self):
        if self.last_click_scene_pos is None or self.current_df is None:
            return
        
        mousePoint = self.ax.vb.mapSceneToView(self.last_click_scene_pos)
        dt = self.get_datetime_from_x(mousePoint.x())
        
        if dt:
            self.sig_set_replay_start.emit(dt)

    def on_delete_drawing_action(self):
        if self.selected_drawing_id is not None:
            self.sig_drawing_delete_request.emit(int(self.selected_drawing_id))

    def set_fib_settings(self, fib_settings):
        self.fib_settings = fib_settings
        if self.draw_mode in ("fib", "fib_ext"):
            self.set_draw_mode(self.draw_mode)

    def on_open_fib_config(self):
        self.sig_fib_config_requested.emit()

    def on_clear_drawings(self):
        self.sig_drawing_clear_request.emit()

    def _snapshot_for_tool(self, mode):
        if mode == "fib":
            return {"levels": self.fib_settings.retracement.effective_levels}
        if mode == "fib_ext":
            return {"levels": self.fib_settings.extension.effective_levels}
        return None

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        if mode is None:
            self.active_drawing_session = None
        else:
            self.active_drawing_session = DrawingSession(
                TOOL_DEFINITIONS[mode],
                config_snapshot=self._snapshot_for_tool(mode),
            )
        self._clear_preview()

    def _clear_preview(self):
        for it in self.draw_preview_items:
            try:
                self.ax.removeItem(it)
            except Exception:
                pass
        self.draw_preview_items = []

    def _handle_draw_click(self, scene_pos):
        if self.active_drawing_session is None:
            return
        mouse_point = self.ax.vb.mapSceneToView(scene_pos)
        spec = self.active_drawing_session.add_point(
            self.get_datetime_from_x(mouse_point.x()),
            float(mouse_point.y()),
        )
        if spec is not None:
            self.sig_drawing_request.emit(spec)
            self.set_draw_mode(None)

    def add_drawing(self, spec):
        spec = normalize_drawing_spec(spec)
        draw_id = spec.get("id")
        if draw_id is None:
            return
        items = render_spec_items(self.ax, spec, self._x_from_datetime)
        if items:
            self.drawings[draw_id] = items

    def remove_drawing(self, draw_id):
        items = self.drawings.pop(draw_id, [])
        for it in items:
            try:
                self.ax.removeItem(it)
            except Exception:
                pass
        if self.selected_drawing_id == draw_id:
            self.selected_drawing_id = None

    def clear_drawings(self):
        for draw_id in list(self.drawings.keys()):
            self.remove_drawing(draw_id)

    def _x_from_datetime(self, dt):
        if dt is None or self.full_df is None or self.full_df.empty:
            return None
        ts = pd.Timestamp(dt)
        if self.full_df.index.tz is None and ts.tzinfo is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        elif self.full_df.index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(self.full_df.index.tz)
        idx = int(self.full_df.index.searchsorted(ts))
        if idx < 0:
            idx = 0
        if idx >= len(self.full_df):
            idx = len(self.full_df) - 1
        return idx

    def get_datetime_from_x(self, x_val):
        """UI helper."""
        if self.current_df is None or self.current_df.empty:
            return None
            
        idx = int(round(x_val))
        last_idx = len(self.current_df) - 1
        
        if 0 <= idx <= last_idx:
            return self.current_df.index[idx]
        
        # 澶栨帹
        if idx > last_idx:
            diff = idx - last_idx
            base_dt = self.current_df.index[last_idx]
        else:
            diff = idx
            base_dt = self.current_df.index[0]
            
        delta = self.time_axis._delta if self.time_axis._delta else datetime.timedelta(minutes=1)
        return base_dt + delta * diff

    def get_timestamp_from_x(self, x_val):
        """Convert an x-axis value back to a timestamp-like coordinate."""
        if self.current_x is None or len(self.current_x) == 0:
            return None
        return float(x_val)

    def sync_vline(self, timestamp):
        """Move the vertical cursor line to the requested timestamp."""
        if self.current_time_values is None or len(self.current_time_values) == 0:
            return
        
        # 1. 灏濊瘯鍦ㄨ寖鍥村唴鏌ユ壘
        # current_time_values 鏄撼绉掔骇 int64 (view) 杞垚鐨?float
        # timestamp 鏄?float (绉?
        # 闇€瑕佺粺涓€鍗曚綅銆俻andas view('int64') 鏄撼绉掋€倀imestamp 鏄銆?
        # 绛夌瓑锛屼箣鍓嶇殑 current_time_values 宸茬粡鏄?float64 鍚楋紵
        # self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        # timestamp() 杩斿洖鐨勬槸绉掋€?
        # 杩欐槸涓€涓弗閲嶇殑鍗曚綅涓嶅尮閰嶉殣鎮ｏ紝涔嬪墠鍙兘鍥犱负纰板阀鏁板€煎ぇ娌℃姤閿欐垨鑰呴€昏緫琚帺鐩栥€?
        # 璁╂垜浠鏌?update_chart 涓殑璧嬪€笺€?
        
        # 鍦?update_chart: self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64) // 10**9 
        # 蹇呴』闄や互 10^9 鎵嶆槸绉掋€傚師浠ｇ爜婕忎簡闄ゆ硶鍚楋紵
        # 鍘熶唬鐮侊細self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        # 杩欏氨鏄撼绉掋€?
        # 鑰?sig_mouse_moved 鍙戝皠鐨勬槸 dt.timestamp() (绉?銆?
        # searchsorted 浼氬畬鍏ㄥけ鏁堬紙鎬绘槸杩斿洖 0 鎴?len锛夈€?
        
        # 鏃㈢劧鐜板湪鎴戣閲嶅啓 sync_vline锛屾垜蹇呴』纭繚瀛樺偍鐨勬槸绉掞紝鎴栬€呰浆鎹竴涓嬨€?
        # 鑰冭檻鍒版€ц兘锛屾渶濂藉瓨绉掋€?
        
        # 涓轰簡鏈€灏忓寲鏀瑰姩椋庨櫓锛屾垜浼氬湪 searchsorted 鍓嶈浆鎹?timestamp 涓虹撼绉掞紝鎴栬€?..
        # 涓嶏紝鏈€濂藉湪 update_chart 閲屼慨姝?current_time_values 鐨勫崟浣嶃€?
        # 浣?update_chart 杩樻病鏀广€?
        
        # 璁╂垜浠厛鍋囧畾 current_time_values 鏄撼绉掋€?
        ts_ns = timestamp * 1e9
        
        if len(self.current_time_values) > 0:
            first_ts = self.current_time_values[0]
            last_ts = self.current_time_values[-1]

            if first_ts <= ts_ns <= last_ts:
                # 鑼冨洿鍐?
                idx = int(np.searchsorted(self.current_time_values, ts_ns))
                self.vLine.setPos(idx)
                self.vLine_macd.setPos(idx)
                self.vLine_rsi.setPos(idx)
            else:
                # 鑼冨洿澶栵紝杩涜鍙嶇畻
                delta_ns = self.time_axis._delta.total_seconds() * 1e9
                if delta_ns > 0:
                    if ts_ns > last_ts:
                        diff = (ts_ns - last_ts) / delta_ns
                        idx = (len(self.current_time_values) - 1) + diff
                    else:
                        diff = (ts_ns - first_ts) / delta_ns
                        idx = diff # 璐熸暟
                    self.vLine.setPos(idx)
                    self.vLine_macd.setPos(idx)
                    self.vLine_rsi.setPos(idx)

    def sync_crosshair(self, timestamp, price):
        self.sync_vline(timestamp)
        self.hLine.setPos(price)

    def on_btn_period_clicked(self, btn):
        period = btn.property("period")
        self.set_period(period)

    def set_period(self, period):
        self.current_period = period
        # 鏇存柊鎸夐挳鐘舵€?
        display = self.display_map.get(period, period)
        for btn in self.btn_group.buttons():
            if btn.text() == display:
                btn.setChecked(True)
                break
        
        self.sig_period_changed.emit(display)

    def _is_ema_enabled(self, name):
        btn = self.ema_toggle_buttons.get(name)
        if btn is None:
            return True
        return btn.isChecked()

    def _is_bollinger_enabled(self):
        return self.btn_toggle_bb is None or self.btn_toggle_bb.isChecked()

    def _are_indicator_panels_enabled(self):
        return self.btn_toggle_macd_rsi is None or self.btn_toggle_macd_rsi.isChecked()

    def _apply_indicator_panel_visibility(self):
        if not hasattr(self, "ax_macd") or not hasattr(self, "ax_rsi"):
            return
        visible = self._are_indicator_panels_enabled()
        self.ax_macd.setVisible(visible)
        self.ax_rsi.setVisible(visible)

    def on_indicator_toggle_changed(self, _checked):
        if not hasattr(self, "full_df") or self.full_df is None or self.full_df.empty:
            return
        # 寮哄埗閲嶆柊鍒囩墖鍒锋柊锛岀珛鍗冲簲鐢‥MA鏄鹃殣
        self._last_slice_start = -1
        self._last_slice_end = -1
        self.refresh_visible_view(force=True)

    def on_indicator_panel_toggle_changed(self, _checked):
        self._apply_indicator_panel_visibility()

    def update_chart(self, df, auto_scale=False, highlight_idx=None):
        if df is None or df.empty:
            return

        # 鎬ц兘浼樺寲锛氭鏌ユ暟鎹槸鍚︾湡鐨勬洿鏂颁簡
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

        # 淇濆瓨鍏ㄩ噺鏁版嵁寮曠敤
        if df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        
        self.full_df = df
        self.current_df = df # 鍏煎鏃ч€昏緫
        
        # 鏁版嵁婧愬彉鏇达紝閲嶇疆鍒囩墖缂撳瓨锛岀‘淇?refresh_visible_view 鑳借Е鍙戞洿鏂?
        self._last_slice_start = -1
        self._last_slice_end = -1
        
        # 璁板綍鏇存柊鍓嶇殑瑙嗗浘鐘舵€?
        last_len = len(self.current_x) if self.current_x is not None else 0
        self.current_x = np.arange(len(df), dtype=np.float64)
        self.current_time_values = np.asarray(df.index.view('int64'), dtype=np.float64)
        
        # 璋冭瘯锛歎I 灞傛敹鍒扮殑鏁版嵁妫€鏌?
        # print(f"UI update_chart received {len(df)} rows. First 3: {df.index[:3].tolist()}")
        
        self.time_axis.set_datetime_index(df.index)

        # 澶勭悊瑙嗗浘鑼冨洿
        view_range = self.ax.vb.viewRange()
        view_right = view_range[0][1]
        is_following = (view_right >= last_len - 0.5)

        if len(df) > 0:
            if auto_scale:
                # 绫讳技 TradingView锛氬畾浣嶅埌鐩爣绱㈠紩锛堟垨鏈熬锛夛紝鏄剧ず绾?150 鏍?K 绾?
                idx = highlight_idx if highlight_idx is not None else len(df) - 1
                x_start = max(0, idx - 150)
                x_end = idx + 20 # 棰勭暀鍙充晶绌虹櫧
                
                # 璁＄畻璇ヨ寖鍥村唴鐨?Y 杞?
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
                
                # 璺熼殢妯″紡涓嬩篃鑷姩璋冩暣 Y 杞?(鏈€杩?150 鏍?
                idx = len(df) - 1
                visible_slice = df.iloc[max(0, idx-150):idx+1]
                y_min = visible_slice['low'].min()
                y_max = visible_slice['high'].max()
                y_pad = (y_max - y_min) * 0.1
                self.ax.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

        # 瑙﹀彂棣栨娓叉煋
        self.refresh_visible_view(force=True)
        
        if hasattr(self, '_custom_wheel_event'):
            self.ax.vb.wheelEvent = self._custom_wheel_event

    def on_detach_clicked(self):
        self.sig_detach_requested.emit(self)

    def set_detached_state(self, detached: bool):
        self.is_detached = detached
        self.btn_detach.setText("Dock" if detached else "Pop")
