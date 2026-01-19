import numpy as np
import pandas as pd


class ReplayEngine:
    def __init__(self, data_engine, max_count=1000, max_ticks_per_frame=20000):
        self.engine = data_engine
        self.max_count = max_count
        self.max_ticks_per_frame = max_ticks_per_frame
        self.periods = []
        self.states = {}
        self.tick_pos = 0
        self._tick_times = None
        self._tick_prices = None
        self._tick_volumes = None

    def initialize(self, periods, start_time):
        self.periods = list(periods)
        self._build_tick_arrays()
        self._build_states(start_time)

    def reset(self, start_time):
        self._build_tick_arrays()
        self._build_states(start_time)

    def advance_to(self, end_time):
        if self._tick_times is None or len(self._tick_times) == 0:
            return None

        end_ts = self._to_naive(end_time)
        end_ts64 = end_ts.to_datetime64()
        end_pos = int(np.searchsorted(self._tick_times, end_ts64, side="right"))
        if end_pos <= self.tick_pos:
            return end_ts

        max_pos = min(end_pos, self.tick_pos + self.max_ticks_per_frame)
        for i in range(self.tick_pos, max_pos):
            ts = self._tick_times[i]
            price = self._tick_prices[i]
            volume = self._tick_volumes[i]
            for state in self.states.values():
                self._update_state_with_tick(state, ts, price, volume)

        self.tick_pos = max_pos
        if self.tick_pos == 0:
            return end_ts
        return pd.Timestamp(self._tick_times[self.tick_pos - 1])

    def get_view(self, period, count=300, with_indicators=True):
        state = self.states.get(period)
        if state is None:
            return None

        idx = list(state["completed_index"])
        o = list(state["completed_open"])
        h = list(state["completed_high"])
        l = list(state["completed_low"])
        c = list(state["completed_close"])
        v = list(state["completed_volume"])

        if state["cur_start"] is not None:
            idx.append(state["cur_start"])
            o.append(state["cur_open"])
            h.append(state["cur_high"])
            l.append(state["cur_low"])
            c.append(state["cur_close"])
            v.append(state["cur_volume"])

        if len(idx) == 0:
            return None

        df = pd.DataFrame(
            {"open": o, "close": c, "high": h, "low": l, "volume": v},
            index=pd.to_datetime(idx),
        )
        if count is not None:
            df = df.tail(count)
        if with_indicators:
            df = self.engine._calculate_indicators(df)
        return df

    def _build_tick_arrays(self):
        if self.engine.df_ticks is None or self.engine.df_ticks.empty:
            self._tick_times = None
            self._tick_prices = None
            self._tick_volumes = None
            self.tick_pos = 0
            return

        idx = self.engine.df_ticks.index
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        self._tick_times = idx.values
        self._tick_prices = self.engine.df_ticks["price"].to_numpy(dtype=np.float64)
        self._tick_volumes = self.engine.df_ticks["volume"].to_numpy(dtype=np.float64)

    def _build_states(self, start_time):
        self.states = {}
        self.tick_pos = 0
        if self._tick_times is None or len(self._tick_times) == 0:
            return

        start_ts = self._to_naive(start_time)
        start_ts64 = start_ts.to_datetime64()
        self.tick_pos = int(np.searchsorted(self._tick_times, start_ts64, side="left"))

        for period in self.periods:
            df_full = self.engine.get_candles(period)
            if df_full is None or df_full.empty:
                continue

            period_index = df_full.index
            period_values = period_index.values
            pos = int(np.searchsorted(period_values, start_ts64, side="right") - 1)
            if pos < 0:
                pos = 0

            completed = df_full.iloc[:pos]
            if len(completed) > self.max_count:
                completed = completed.tail(self.max_count)

            completed_index = list(completed.index)
            completed_open = list(completed["open"].to_numpy(dtype=np.float64))
            completed_high = list(completed["high"].to_numpy(dtype=np.float64))
            completed_low = list(completed["low"].to_numpy(dtype=np.float64))
            completed_close = list(completed["close"].to_numpy(dtype=np.float64))
            completed_volume = list(completed["volume"].to_numpy(dtype=np.float64))

            cur_start = period_values[pos] if len(period_values) > 0 else None
            cur_open = np.nan
            cur_high = np.nan
            cur_low = np.nan
            cur_close = np.nan
            cur_volume = 0.0

            if cur_start is not None:
                cur_start64 = pd.Timestamp(cur_start).to_datetime64()
                start_idx = int(np.searchsorted(self._tick_times, cur_start64, side="left"))
                end_idx = int(np.searchsorted(self._tick_times, start_ts64, side="right"))
                if end_idx > start_idx:
                    prices = self._tick_prices[start_idx:end_idx]
                    volumes = self._tick_volumes[start_idx:end_idx]
                    cur_open = prices[0]
                    cur_close = prices[-1]
                    cur_high = float(np.nanmax(prices))
                    cur_low = float(np.nanmin(prices))
                    cur_volume = float(np.nansum(volumes))

            next_start = period_values[pos + 1] if pos + 1 < len(period_values) else None

            self.states[period] = {
                "period_values": period_values,
                "pos": pos,
                "next_start": next_start,
                "completed_index": completed_index,
                "completed_open": completed_open,
                "completed_high": completed_high,
                "completed_low": completed_low,
                "completed_close": completed_close,
                "completed_volume": completed_volume,
                "cur_start": cur_start,
                "cur_open": cur_open,
                "cur_high": cur_high,
                "cur_low": cur_low,
                "cur_close": cur_close,
                "cur_volume": cur_volume,
            }

    def _update_state_with_tick(self, state, ts, price, volume):
        if state["cur_start"] is None:
            return

        if state["next_start"] is None or ts < state["next_start"]:
            self._update_current_candle(state, price, volume)
            return

        period_values = state["period_values"]
        target_pos = int(np.searchsorted(period_values, ts, side="right") - 1)
        if target_pos < state["pos"]:
            return

        # Finalize current candle.
        self._append_completed(state, state["cur_start"], state["cur_open"],
                               state["cur_high"], state["cur_low"],
                               state["cur_close"], state["cur_volume"])

        # Fill skipped candles (no ticks).
        for pos in range(state["pos"] + 1, target_pos):
            self._append_completed(state, period_values[pos], np.nan, np.nan, np.nan, np.nan, 0.0)

        state["pos"] = target_pos
        state["cur_start"] = period_values[target_pos]
        state["cur_open"] = price
        state["cur_high"] = price
        state["cur_low"] = price
        state["cur_close"] = price
        state["cur_volume"] = volume
        next_pos = target_pos + 1
        state["next_start"] = period_values[next_pos] if next_pos < len(period_values) else None

    def _update_current_candle(self, state, price, volume):
        if np.isnan(state["cur_open"]):
            state["cur_open"] = price
            state["cur_high"] = price
            state["cur_low"] = price
            state["cur_close"] = price
            state["cur_volume"] = volume
            return

        state["cur_high"] = max(state["cur_high"], price)
        state["cur_low"] = min(state["cur_low"], price)
        state["cur_close"] = price
        state["cur_volume"] += volume

    def _append_completed(self, state, ts, o, h, l, c, v):
        state["completed_index"].append(ts)
        state["completed_open"].append(o)
        state["completed_high"].append(h)
        state["completed_low"].append(l)
        state["completed_close"].append(c)
        state["completed_volume"].append(v)
        if len(state["completed_index"]) > self.max_count:
            state["completed_index"].pop(0)
            state["completed_open"].pop(0)
            state["completed_high"].pop(0)
            state["completed_low"].pop(0)
            state["completed_close"].pop(0)
            state["completed_volume"].pop(0)

    def _to_naive(self, dt):
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("America/New_York").tz_localize(None)
