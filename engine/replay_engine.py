import numpy as np
import pandas as pd


class ReplayEngine:
    def __init__(self, data_engine, max_count=1000, max_ticks_per_frame=20000):
        self.engine = data_engine
        self.max_count = max_count
        self.max_count_map = {}
        self.max_ticks_per_frame = max_ticks_per_frame
        self.auto_skip_gaps = True
        self.indicator_tail = 300
        self.periods = []
        self.states = {}
        self.tick_pos = 0
        self._tick_times = None
        self._tick_prices = None
        self._tick_volumes = None

    def initialize(self, periods, start_time, max_count_map=None):
        self.periods = list(periods)
        self.max_count_map = max_count_map or {}
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
            if self.auto_skip_gaps and self.tick_pos < len(self._tick_times):
                ts = self._tick_times[self.tick_pos]
                price = self._tick_prices[self.tick_pos]
                volume = self._tick_volumes[self.tick_pos]
                for state in self.states.values():
                    self._update_state_with_tick(state, ts, price, volume)
                self.tick_pos += 1
                return pd.Timestamp(ts)
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
        indicator_cols = state.get("indicator_cols", [])
        completed_indicators = state.get("completed_indicators", {})
        indicator_data = {}
        for col in indicator_cols:
            values = list(completed_indicators.get(col, []))
            if len(values) < len(idx):
                values.extend([np.nan] * (len(idx) - len(values)))
            indicator_data[col] = values

        if state["cur_start"] is not None:
            idx.append(state["cur_start"])
            o.append(state["cur_open"])
            h.append(state["cur_high"])
            l.append(state["cur_low"])
            c.append(state["cur_close"])
            v.append(state["cur_volume"])
            if with_indicators and indicator_cols:
                cur_indicators = self._calc_current_indicators(state)
                for col in indicator_cols:
                    indicator_data.setdefault(col, [np.nan] * (len(idx) - 1))
                    indicator_data[col].append(cur_indicators.get(col, np.nan))

        if len(idx) == 0:
            return None

        data = {"open": o, "close": c, "high": h, "low": l, "volume": v}
        if with_indicators and indicator_cols:
            data.update(indicator_data)
        df = pd.DataFrame(data, index=pd.to_datetime(idx))
        if count is not None:
            df = df.tail(count)
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
            max_count = self.max_count_map.get(period, self.max_count)
            if len(completed) > max_count:
                completed = completed.tail(max_count)

            completed_index = list(completed.index)
            completed_open = list(completed["open"].to_numpy(dtype=np.float64))
            completed_high = list(completed["high"].to_numpy(dtype=np.float64))
            completed_low = list(completed["low"].to_numpy(dtype=np.float64))
            completed_close = list(completed["close"].to_numpy(dtype=np.float64))
            completed_volume = list(completed["volume"].to_numpy(dtype=np.float64))
            indicator_cols = [col for col in df_full.columns if col not in {"open", "high", "low", "close", "volume"}]
            completed_indicators = {}
            indicator_full = {}
            for col in indicator_cols:
                indicator_full[col] = df_full[col].to_numpy()
                completed_indicators[col] = list(completed[col].to_numpy())

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
                "max_count": max_count,
                "completed_index": completed_index,
                "completed_open": completed_open,
                "completed_high": completed_high,
                "completed_low": completed_low,
                "completed_close": completed_close,
                "completed_volume": completed_volume,
                "indicator_cols": indicator_cols,
                "completed_indicators": completed_indicators,
                "indicator_full": indicator_full,
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
        self._append_completed(state, state["pos"], state["cur_start"], state["cur_open"],
                               state["cur_high"], state["cur_low"],
                               state["cur_close"], state["cur_volume"])

        # Fill skipped candles (no ticks).
        for pos in range(state["pos"] + 1, target_pos):
            self._append_completed(state, pos, period_values[pos], np.nan, np.nan, np.nan, np.nan, 0.0)

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

    def _append_completed(self, state, pos, ts, o, h, l, c, v):
        state["completed_index"].append(ts)
        state["completed_open"].append(o)
        state["completed_high"].append(h)
        state["completed_low"].append(l)
        state["completed_close"].append(c)
        state["completed_volume"].append(v)
        indicator_cols = state.get("indicator_cols", [])
        completed_indicators = state.get("completed_indicators", {})
        indicator_full = state.get("indicator_full", {})
        for col in indicator_cols:
            values = completed_indicators.setdefault(col, [])
            if pos is not None and col in indicator_full:
                try:
                    values.append(indicator_full[col][pos])
                except Exception:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        max_count = state.get("max_count", self.max_count)
        if len(state["completed_index"]) > max_count:
            state["completed_index"].pop(0)
            state["completed_open"].pop(0)
            state["completed_high"].pop(0)
            state["completed_low"].pop(0)
            state["completed_close"].pop(0)
            state["completed_volume"].pop(0)
            for col in indicator_cols:
                if completed_indicators.get(col):
                    completed_indicators[col].pop(0)

    def _to_naive(self, dt):
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("America/New_York").tz_localize(None)

    def _calc_current_indicators(self, state):
        indicator_cols = state.get("indicator_cols", [])
        if not indicator_cols:
            return {}
        tail_len = min(len(state["completed_index"]), self.indicator_tail)
        if tail_len <= 0:
            return {}

        idx = list(state["completed_index"][-tail_len:])
        idx.append(state["cur_start"])
        o = list(state["completed_open"][-tail_len:]) + [state["cur_open"]]
        h = list(state["completed_high"][-tail_len:]) + [state["cur_high"]]
        l = list(state["completed_low"][-tail_len:]) + [state["cur_low"]]
        c = list(state["completed_close"][-tail_len:]) + [state["cur_close"]]
        v = list(state["completed_volume"][-tail_len:]) + [state["cur_volume"]]

        df = pd.DataFrame(
            {"open": o, "close": c, "high": h, "low": l, "volume": v},
            index=pd.to_datetime(idx),
        )
        df = self.engine._calculate_indicators(df)
        if df.empty:
            return {}
        last = df.iloc[-1]
        return {col: last.get(col, np.nan) for col in indicator_cols}
