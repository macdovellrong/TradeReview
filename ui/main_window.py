import datetime
import os
from functools import partial

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QSplitter, QFileDialog,
                             QGridLayout, QTabWidget, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QDateTime, QSettings
from engine.data_engine import DataEngine
from engine.replay_engine import ReplayEngine
# Compatibility exports for legacy imports.
from ui.main_controls import MainControls
from ui.controllers.replay_controller import ReplayController
from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
from ui.chart_widget import ChartWidget
from ui.chart_window import FloatingChartWindow
from ui.crosshair_sync import CrosshairSyncController
from ui.drawings.dialogs import FibConfigDialog
from ui.drawings.fib_config import load_fib_settings, save_fib_settings
from ui.drawings.specs import normalize_drawing_spec
from ui.services.data_loading import DataLoadingFacade
from ui.session_state import SessionState, load_session_state, save_session_state
from ui.time_navigation import clamp_timestamp, normalize_jump_timestamp, resolve_chart_target


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradeReview")
        self.resize(1400, 950)
        
        self.engine = DataEngine(parquet_file=None) 
        self.replay_engine = ReplayEngine(self.engine)
        self.replay_controller = ReplayController(self.replay_engine)
        self.data_loading = DataLoadingFacade(self.engine)
        self.settings = QSettings("TradeReview", "TradeReview")
        self.fib_settings = load_fib_settings(self.settings)
        self.current_time = datetime.datetime.now()
        self.is_playing = False
        self.replay_speed = 60 
        self.crosshair_sync_controller = CrosshairSyncController()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        self.create_control_panel()
        
        self.chart_container_layout = QVBoxLayout() 
        self.main_layout.addLayout(self.chart_container_layout)
        
        self.floating_windows = []
        self.charts = []
        self.tabs = None # 寮曠敤 tabs 缁勪欢
        self.init_charts()
        
        self.switch_layout("Tabs") # 榛樿 Tabs
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(100) 
        QTimer.singleShot(0, self.restore_saved_view)

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

    def _to_qdatetime(self, dt):
        ts = normalize_jump_timestamp(dt)
        if isinstance(ts, pd.Timestamp):
            dt = ts.to_pydatetime()
        else:
            dt = ts
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0)

    def _set_date_edit(self, dt):
        if dt is None:
            return
        self.date_edit.setDateTime(self._to_qdatetime(dt))

    def _update_date_edit_bounds(self):
        if not hasattr(self, "date_edit"):
            return
        if self.engine.df_ticks is None or self.engine.df_ticks.empty:
            return
        start = normalize_jump_timestamp(self.engine.df_ticks.index[0])
        end = normalize_jump_timestamp(self.engine.df_ticks.index[-1])
        if end < start:
            end = start
        self.date_edit.setMinimumDateTime(self._to_qdatetime(start))
        self.date_edit.setMaximumDateTime(self._to_qdatetime(end))

    def _center_charts_on_time(self, target_dt):
        target_ts = self._normalize_time(target_dt)
        for chart in self._get_enabled_charts():
            if chart.full_df is None or chart.full_df.empty:
                continue

            idx, close_price = resolve_chart_target(chart.full_df, target_ts)
            if idx is None:
                continue

            x_range = chart.ax.vb.viewRange()[0]
            x_span = x_range[1] - x_range[0]
            if x_span <= 0:
                x_span = max(self._get_view_count_for_period(chart.current_period), 100)
            chart.ax.setXRange(idx - x_span / 2, idx + x_span / 2, padding=0)

            if close_price is None or not np.isfinite(close_price):
                continue
            y_range = chart.ax.vb.viewRange()[1]
            y_span = y_range[1] - y_range[0]
            if not np.isfinite(y_span) or y_span <= 0:
                y_span = max(abs(close_price) * 0.02, 1.0)
            half_span = y_span / 2
            chart.ax.setYRange(close_price - half_span, close_price + half_span, padding=0)

    def jump_to_time(self, target_dt):
        if self.engine.df_ticks is None or self.engine.df_ticks.empty:
            return

        target_ts = normalize_jump_timestamp(self._normalize_time(target_dt))
        start = normalize_jump_timestamp(self.engine.df_ticks.index[0])
        end = normalize_jump_timestamp(self.engine.df_ticks.index[-1])
        target_ts = clamp_timestamp(target_ts, start, end)
        self.current_time = target_ts

        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)

        if self.chk_replay.isChecked():
            self._ensure_replay_engine()
            self.replay_controller.reset(self.current_time)

        self.refresh_all_charts(auto_scale=False)
        self._center_charts_on_time(self.current_time)

    def on_date_edit_finished(self):
        if self.engine.df_ticks is None or self.engine.df_ticks.empty:
            return
        selected_dt = self.date_edit.dateTime().toPyDateTime().replace(second=0, microsecond=0)
        self.jump_to_time(selected_dt)

    def _get_view_center_time(self):
        charts = self._get_enabled_charts()
        if self.combo_layout.currentText() == "Tabs" and self.tabs is not None:
            current_chart = self.tabs.currentWidget()
            if current_chart is not None:
                charts = [current_chart]

        for chart in charts:
            if chart.full_df is None or chart.full_df.empty:
                continue
            x_min, x_max = chart.ax.vb.viewRange()[0]
            center_dt = chart.get_datetime_from_x((x_min + x_max) / 2.0)
            if center_dt is not None:
                return normalize_jump_timestamp(self._normalize_time(center_dt))
        if self.engine.df_ticks is None or self.engine.df_ticks.empty:
            return None
        return normalize_jump_timestamp(self._normalize_time(self.current_time))

    def on_save_view(self):
        if self.engine.df_ticks is None or self.engine.df_ticks.empty or not self.engine.parquet_file:
            QMessageBox.information(self, "Save View", "Load a database before saving the current view.")
            return

        center_time = self._get_view_center_time()
        if center_time is None:
            QMessageBox.warning(self, "Save View", "No chart position is available to save.")
            return

        save_session_state(
            self.settings,
            SessionState(
                db_path=str(self.engine.parquet_file),
                center_time=center_time,
            ),
        )
        QMessageBox.information(self, "Save View", "Current database and chart position have been saved.")

    def restore_saved_view(self):
        if self.engine.df_ticks is not None and not self.engine.df_ticks.empty:
            return

        state = load_session_state(self.settings)
        if state is None:
            return
        if not os.path.exists(state.db_path):
            return

        self.load_data_file(state.db_path, restore_time=state.center_time)

    def _get_replay_periods(self):
        return [chart.current_period for chart in self._get_enabled_charts()]

    def _ensure_replay_engine(self):
        if self.engine.df_ticks is None:
            return
        if not self.replay_engine.states:
            self.replay_controller.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )

    def _get_replay_max_count_map(self):
        max_count_map = {}
        for chart in self._get_enabled_charts():
            period = chart.current_period
            max_count_map[period] = self._get_max_count_for_period(period)
        return max_count_map

    def _get_max_count_for_period(self, period):
        tf = str(period).strip().lower()
        if tf.endswith("s"):
            try:
                seconds = int(tf[:-1])
            except ValueError:
                return 1500
            if seconds <= 30:
                return 6000
            return 4000
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
        if tf.endswith("s"):
            try:
                seconds = int(tf[:-1])
            except ValueError:
                return 400
            if seconds <= 30:
                return 1200
            return 800
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
        # 鍒濆鍙垱寤?涓紝榛樿缁欎笉鍚屽懆鏈?
        configs = [("1h", "1h"), ("15min", "15min"), ("5min", "5min"), ("1min", "1min")]
        for i, (display, period) in enumerate(configs):
            chart = ChartWidget(display)
            # 璁剧疆鍒濆鍛ㄦ湡骞惰Е鍙戝姞杞?
            chart.set_period(period)
            chart.set_fib_settings(self.fib_settings)
            
            # 鐩戝惉鍛ㄦ湡鏀瑰彉锛岃Е鍙戦噸缁?+ 鏇存柊 Tab 鏍囬
            chart.sig_period_changed.connect(lambda p, c=chart: self.on_chart_period_changed(c, p))
            
            # 鐩戝惉鍒嗙璇锋眰
            chart.sig_detach_requested.connect(self.toggle_chart_detach)
            
            # 鐩戝惉鍚屾璇锋眰
            chart.sig_sync_center_requested.connect(self.sync_charts_center)
            chart.sig_sync_y_center_requested.connect(self.sync_charts_y_center)
            
            # 鐩戝惉鍥炴斁璺宠浆璇锋眰
            chart.sig_set_replay_start.connect(self.set_replay_start_time)
            
            # 杩炴帴鍏夋爣鍚屾淇″彿
            self.crosshair_sync_controller.register_chart(chart)
            chart.sig_mouse_moved_with_price.connect(
                partial(self.sync_all_charts_crosshair, chart)
            )
            chart.sig_drawing_request.connect(self.on_drawing_request)
            chart.sig_drawing_delete_request.connect(self.on_drawing_delete)
            chart.sig_drawing_clear_request.connect(self.on_drawing_clear)
            chart.sig_fib_config_requested.connect(self.on_open_fib_config)
            self.charts.append(chart)

        self.refresh_crosshair_sync_targets()

    def set_replay_start_time(self, target_dt):
        """UI helper."""
        if self.engine.df_ticks is None:
            return

        # 濡傛灉褰撳墠涓嶅湪鍥炴斁妯″紡锛岃嚜鍔ㄥ紑鍚?
        if not self.chk_replay.isChecked():
            self.chk_replay.setChecked(True)
            
        # 鏇存柊鍐呴儴鏃堕棿
        self.current_time = self._normalize_time(target_dt)
        if self.replay_engine is not None:
            self.replay_controller.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )
        
        # 鏇存柊鏃堕棿鏄剧ず妗?
        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)
        
        # 寮哄埗鍒锋柊鍥捐〃
        self.refresh_all_charts(auto_scale=True) # 璺宠浆鍚庨€氬父甯屾湜鑷姩鑱氱劍

    def sync_charts_center(self, target_dt):
        """UI helper."""
        target_ts = self._normalize_time(target_dt)
        for chart in self._get_enabled_charts():
            if chart.full_df is None or chart.full_df.empty:
                continue

            idx = self._get_chart_index_for_dt(chart, target_ts)
            if idx is None:
                continue

            view_range = chart.ax.vb.viewRange()[0]
            span = view_range[1] - view_range[0]

            new_min = idx - span / 2
            new_max = idx + span / 2

            chart.ax.setXRange(new_min, new_max, padding=0)

    def sync_charts_y_center(self, target_price):
        for chart in self._get_enabled_charts():
            view_range = chart.ax.vb.viewRange()[1]
            span = view_range[1] - view_range[0]
            if span <= 0:
                span = max(abs(target_price) * 0.02, 1.0)
            half_span = span / 2
            chart.ax.setYRange(target_price - half_span, target_price + half_span, padding=0)

    def _get_chart_index_for_dt(self, chart, target_dt):
        df = chart.full_df
        if df is None or df.empty:
            return None
        idx, _ = resolve_chart_target(df, target_dt)
        return idx

    def on_chart_period_changed(self, chart, period_display):
        if self.chk_replay.isChecked() and self.engine.df_ticks is not None:
            self.replay_controller.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )

        # 1. 鍒锋柊鏁版嵁
        self.refresh_single_chart(chart, auto_scale=True)
        
        # 2. 濡傛灉鍦?Tabs 妯″紡锛屾洿鏂版爣棰?
        if self.combo_layout.currentText() == "Tabs" and self.tabs is not None:
            idx = self.tabs.indexOf(chart)
            if idx != -1:
                self.tabs.setTabText(idx, period_display)

    def sync_all_charts(self, timestamp):
        """UI helper."""
        for chart in self._get_enabled_charts():
            chart.sync_vline(timestamp)

    def sync_all_charts_crosshair(self, source_chart, timestamp, price):
        self.crosshair_sync_controller.sync_from(source_chart, timestamp, price)

    def on_drawing_request(self, spec):
        draw_id = getattr(self, "_drawing_id_counter", 0) + 1
        self._drawing_id_counter = draw_id
        spec = normalize_drawing_spec(spec)
        spec = dict(spec)
        spec["id"] = draw_id
        for chart in self.charts:
            chart.add_drawing(spec)

    def on_drawing_delete(self, draw_id):
        for chart in self.charts:
            chart.remove_drawing(draw_id)

    def on_drawing_clear(self):
        for chart in self.charts:
            chart.clear_drawings()

    def on_open_fib_config(self):
        dialog = FibConfigDialog(self.fib_settings, self)
        if not dialog.exec():
            return
        fib_settings = dialog.build_settings()
        save_fib_settings(self.settings, fib_settings)
        self.fib_settings = fib_settings
        for chart in self.charts:
            chart.set_fib_settings(self.fib_settings)

    def create_control_panel(self):
        self.controls = MainControls(
            current_time=self.current_time,
            replay_speed=self.replay_speed,
            parent=self,
        )
        self.controls.load_requested.connect(self.open_file_dialog)
        self.controls.reset_requested.connect(self.reset_charts_view)
        self.controls.save_view_requested.connect(self.on_save_view)
        self.controls.layout_changed.connect(self.switch_layout)
        self.controls.pop_layout_requested.connect(self.detach_layout_charts)
        self.controls.chart_count_changed.connect(self.on_chart_count_changed)
        self.controls.replay_mode_changed.connect(self.on_mode_change)
        self.controls.play_requested.connect(self.toggle_play)
        self.controls.step_back_requested.connect(self.on_step_back)
        self.controls.step_forward_requested.connect(self.on_step_forward)
        self.controls.speed_changed.connect(self.set_speed)
        self.controls.date_edit_finished.connect(self.on_date_edit_finished)

        self.combo_layout = self.controls.combo_layout
        self.btn_detach_layout = self.controls.btn_detach_layout
        self.combo_chart_count = self.controls.combo_chart_count
        self.chk_replay = self.controls.chk_replay
        self.btn_play = self.controls.btn_play
        self.btn_step_back = self.controls.btn_step_back
        self.btn_step_forward = self.controls.btn_step_forward
        self.combo_step = self.controls.combo_step
        self.speed_btn_group = self.controls.speed_btn_group
        self.date_edit = self.controls.date_edit

        self.main_layout.addWidget(self.controls)


    def toggle_chart_detach(self, chart):
        if chart.is_detached:
            self.attach_chart(chart)
        else:
            self.detach_chart(chart)

    def detach_layout_charts(self):
        layout_name = self.combo_layout.currentText()
        charts_to_detach = self._get_layout_charts(layout_name)
        detached_any = False
        for chart in charts_to_detach:
            if chart.is_detached:
                continue
            self.detach_chart(chart, refresh_layout=False)
            detached_any = True
        if detached_any:
            self.switch_layout(layout_name)

    def detach_chart(self, chart, refresh_layout=True):
        if chart.is_detached:
            return

        chart.set_detached_state(True)

        fw = FloatingChartWindow(chart)
        fw.sig_window_closed.connect(self.on_floating_window_closed)
        fw.show()
        self.floating_windows.append(fw)

        if refresh_layout:
            self.switch_layout(self.combo_layout.currentText())

    def on_chart_count_changed(self, count_text):
        try:
            enabled_count = max(1, int(count_text))
        except ValueError:
            return

        for chart in self.charts[enabled_count:]:
            if chart.is_detached:
                self.attach_chart(chart)

        if self.chk_replay.isChecked() and self.engine.df_ticks is not None:
            self.replay_controller.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )

        self.refresh_crosshair_sync_targets()
        self.switch_layout(self.combo_layout.currentText())
        self.refresh_all_charts(auto_scale=True)

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

    def _get_chart_count(self):
        if hasattr(self, "combo_chart_count"):
            try:
                return max(1, min(int(self.combo_chart_count.currentText()), len(self.charts)))
            except ValueError:
                pass
        return len(self.charts)

    def _get_enabled_charts(self):
        return self.charts[:self._get_chart_count()]

    def _get_attached_charts(self):
        return [chart for chart in self._get_enabled_charts() if not chart.is_detached]

    def _get_layout_charts(self, layout_name):
        active_charts = self._get_attached_charts()
        if layout_name == "Dual Vertical":
            return active_charts[:2]
        return active_charts

    def refresh_crosshair_sync_targets(self):
        enabled_charts = list(self._get_enabled_charts())
        enabled_set = set(enabled_charts)

        # Keep crosshair sync scoped to enabled charts only.
        for chart in list(self.crosshair_sync_controller.iter_charts()):
            if chart not in enabled_set:
                self.crosshair_sync_controller.unregister_chart(chart)

        for chart in enabled_charts:
            self.crosshair_sync_controller.register_chart(chart)

    def switch_layout(self, layout_name):
        active_charts = self._get_attached_charts()
        layout_charts = self._get_layout_charts(layout_name)
        for chart in active_charts:
            if chart.parent() is not None:
                chart.setParent(None)

        while self.chart_container_layout.count():
            item = self.chart_container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.tabs = None

        if layout_name in ("Vertical", "Dual Vertical"):
            splitter = QSplitter(Qt.Orientation.Vertical)
            for chart in layout_charts:
                splitter.addWidget(chart)
                chart.show()
            self.chart_container_layout.addWidget(splitter)

        elif layout_name == "Grid 2x2":
            grid_widget = QWidget()
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(2)
            grid_widget.setLayout(grid)

            # Simple grid logic for dynamic number of charts
            for i, chart in enumerate(layout_charts):
                row = i // 2
                col = i % 2
                grid.addWidget(chart, row, col)
                chart.show()
            grid.setRowStretch(0, 1)
            grid.setRowStretch(1, 1)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1)

            self.chart_container_layout.addWidget(grid_widget)

        elif layout_name == "Tabs":
            self.tabs = QTabWidget()
            self.tabs.setStyleSheet(
                "QTabBar::tab {"
                " height: 30px; padding: 6px 16px; font-size: 12px;"
                " background: #1e1e1e; color: #9aa0a6; border: 1px solid #2a2a2a;"
                " border-bottom: 0; border-top-left-radius: 6px; border-top-right-radius: 6px;"
                " margin-right: 4px;"
                "}"
                "QTabBar::tab:selected { background: #2b2b2b; color: #ffffff; border-color: #3a3a3a; }"
                "QTabBar::tab:hover { background: #252525; color: #d0d0d0; }"
                "QTabWidget::pane { border-top: 1px solid #3a3a3a; }"
            )
            for chart in layout_charts:
                title = chart.display_map.get(chart.current_period, chart.current_period)
                self.tabs.addTab(chart, title)
                chart.show()
            self.chart_container_layout.addWidget(self.tabs)

        # Layout switching only reflows the containers.
    def load_data_file(self, file_path, restore_time=None):
        result = self.data_loading.load(file_path)
        if not result.success:
            QMessageBox.critical(
                self,
                "Load Data Failed",
                result.error,
            )
            return

        self.current_time = result.initial_time

        if hasattr(self, 'date_edit'):
            self._update_date_edit_bounds()
            self._set_date_edit(self.current_time)
        self.reset_charts_view()
        if hasattr(self, 'replay_engine'):
            self.replay_controller.initialize(
                self._get_replay_periods(),
                self.current_time,
                max_count_map=self._get_replay_max_count_map(),
            )
        if restore_time is not None:
            self.jump_to_time(restore_time)
        if result.warnings:
            QMessageBox.warning(
                self,
                "Data Load Warning",
                "\n\n".join(result.warnings),
            )

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data",
            "",
            "Data Files (*.parquet *.duckdb);;Parquet Files (*.parquet);;DuckDB Files (*.duckdb);;All Files (*)",
        )
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
            self.replay_controller.reset(self.current_time)
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
        self.replay_controller.set_enabled(is_replay)
        self.is_playing = self.replay_controller.is_playing
        self.btn_play.setEnabled(is_replay)
        self.btn_play.setText("Pause" if self.is_playing else "Play")
        if is_replay:
            self._ensure_replay_engine()
        self.refresh_all_charts(auto_scale=True)

    def toggle_play(self):
        self.is_playing = self.replay_controller.toggle_playing()
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def set_speed(self, speed):
        self.replay_speed = speed
        self.replay_controller.set_speed(speed)

    def reset_charts_view(self):
        self.refresh_all_charts(auto_scale=True)

    def refresh_single_chart(self, chart, auto_scale=False):
        if self.engine.df_ticks is None:
            return
        
        target_idx = None
        if self.chk_replay.isChecked():
            self._ensure_replay_engine()
            view_count = self._get_view_count_for_period(chart.current_period)
            df = self.replay_controller.get_view(chart.current_period, count=view_count, with_indicators=True)
            if df is not None:
                target_idx = len(df) - 1
        else:
            df = self.engine.get_candles(chart.current_period) 
            # 鍏ㄩ噺妯″紡涓嬶紝鏍规嵁褰撳墠鏃堕棿鎵惧埌瀵瑰簲鐨?index
            if df is not None and not df.empty:
                search_time = self.current_time
                # 濡傛灉 K 绾跨储寮曟槸 Naive (鍥犱负 NY Close 杞崲)锛岃€?current_time 鏄?Aware锛屽垯鍘婚櫎鏃跺尯
                if df.index.tz is None and search_time.tzinfo is not None:
                    search_time = search_time.replace(tzinfo=None)
                
                target_idx = df.index.searchsorted(search_time)
                
        chart.update_chart(df, auto_scale=auto_scale, highlight_idx=target_idx)

    def refresh_all_charts(self, auto_scale=False):
        for chart in self._get_enabled_charts():
            self.refresh_single_chart(chart, auto_scale=auto_scale)

    def on_timer_tick(self):
        if not self.chk_replay.isChecked() or not self.is_playing:
            return
        
        current_ts = self._normalize_time(self.current_time)
        target_time = current_ts + pd.Timedelta(seconds=self.replay_speed)

        self._ensure_replay_engine()
        actual_time = self.replay_controller.advance_to(target_time)
        if actual_time is None:
            return
        self.current_time = self.replay_controller.current_time

        if self.engine.df_ticks is not None:
            idx = min(self.replay_engine.tick_pos, len(self.engine.df_ticks) - 1)
            if idx >= len(self.engine.df_ticks) - 1:
                self.toggle_play()

        self.date_edit.blockSignals(True)
        self._set_date_edit(self.current_time)
        self.date_edit.blockSignals(False)
        self.refresh_all_charts()
