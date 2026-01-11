import finplot as fplt
import pandas as pd
from PyQt6.QtWidgets import QApplication, QGridLayout, QWidget
from PyQt6.QtCore import QTimer
from data_engine import DataEngine
import datetime

class TradeReviewUI:
    def __init__(self):
        self.engine = DataEngine()
        
        # 状态变量
        # 从数据的开始时间 + 一小段偏移开始，确保初始就有K线看
        self.current_time = self.engine.df_ticks.index[1000] 
        self.replay_speed = 60  # 每一帧前进 60 秒 (模拟 1 分钟/帧)
        self.is_playing = True
        
        # 初始化界面
        self.app = QApplication([])
        self.win = QWidget()
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        self.win.resize(1200, 800)
        self.win.setWindowTitle("Gemini Trade Review - Multi-Period Replay")

        # 创建 Finplot 视图
        # ax1: H1, ax2: M5
        self.ax_h1 = fplt.create_plot('H1 Period', rows=1)
        self.ax_m5 = fplt.create_plot('M5 Period', rows=1)
        
        # 将 Finplot 的视图添加到 PyQt 布局中
        self.layout.addWidget(self.ax_h1.vb.win, 0, 0)
        self.layout.addWidget(self.ax_m5.vb.win, 1, 0)

        # 初始数据对象 (用于后续 update)
        self.plots = {}
        
        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_replay)
        self.timer.start(100) # 100ms 刷新一次界面

    def update_replay(self):
        if not self.is_playing:
            return

        # 1. 推进时间
        self.current_time += datetime.timedelta(seconds=self.replay_speed)
        
        # 2. 获取新数据
        df_h1 = self.engine.get_candles_by_time('1h', self.current_time, count=100)
        df_m5 = self.engine.get_candles_by_time('5min', self.current_time, count=100)

        # 3. 绘制/更新
        if df_h1 is not None:
            if 'h1' not in self.plots:
                self.plots['h1'] = fplt.candlestick_ochl(df_h1[['open', 'close', 'high', 'low']], ax=self.ax_h1)
            else:
                self.plots['h1'].update_data(df_h1[['open', 'close', 'high', 'low']])

        if df_m5 is not None:
            if 'm5' not in self.plots:
                self.plots['m5'] = fplt.candlestick_ochl(df_m5[['open', 'close', 'high', 'low']], ax=self.ax_m5)
            else:
                self.plots['m5'].update_data(df_m5[['open', 'close', 'high', 'low']])

        # 刷新界面
        # fplt.refresh() # 在某些版本中可能需要

    def run(self):
        fplt.show(qt_exec=False) # 开启 finplot
        self.win.show()
        self.app.exec()

if __name__ == "__main__":
    ui = TradeReviewUI()
    ui.run()
