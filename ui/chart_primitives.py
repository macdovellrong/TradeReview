import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPainter, QPicture


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
            # 鐠侊紕鐣诲銉╂毐閿涙艾褰囬崜?10 娑撳仯閻ㄥ嫬妯婇崐鑲╂畱娑撶秴閺佸府绱濇禒銉╂Щ閺佺増宓佸鈧径瀛樻箒缂傚搫褰?
            count = min(10, len(dt_index) - 1)
            deltas = []
            for i in range(count):
                deltas.append(dt_index[i + 1] - dt_index[i])
            # 缁犫偓閸楁洜娈戦崣鏍﹁厬娴ｅ秵鏆?
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

        # 鐠嬪啳鐦敍姘ⅵ閸楁澘澧犻崙鐘遍嚋閸掕瀹抽惃鍕Ё鐏忓嫭鍎忛崘?(娴犲懎缍?values 閸栧懎鎯堟潏鍐ㄧ毈缁便垹绱╅弮?
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
