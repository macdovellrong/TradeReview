import os
import pandas as pd
import numpy as np

class DataEngine:
    def __init__(self, parquet_file="data/ticks.parquet"):
        self.parquet_file = parquet_file
        self.df_ticks = None
        self.calendar_name = "CME_FX"
        self._calendar = None
        self._candles_cache = {}
        self._duckdb_path = None
        self._duckdb_candles_tables = set()
        self.load_data()

    def load_data(self):
        """加载 Tick 数据并进行基础处理"""
        print(f"Loading data from {self.parquet_file}...")
        try:
            if not self.parquet_file:
                return

            ext = os.path.splitext(self.parquet_file)[1].lower()
            if ext == ".duckdb":
                try:
                    import duckdb
                except Exception as e:
                    print(f"Error loading DuckDB: {e}")
                    return

                self._duckdb_path = self.parquet_file
                con = duckdb.connect(self.parquet_file, read_only=True)
                tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                self._duckdb_candles_tables = {t for t in tables if t.startswith("candles_")}
                if self._duckdb_candles_tables:
                    periods = [self._period_from_table(t) for t in sorted(self._duckdb_candles_tables)]
                    print(f"DuckDB candle tables: {', '.join(periods)}")

                df = con.execute("SELECT * FROM ticks ORDER BY timestamp").df()
                con.close()
                if "timestamp" not in df.columns:
                    print("Error loading DuckDB: missing 'timestamp' column in ticks table. Please rebuild DuckDB.")
                    return
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                self.df_ticks = df
            else:
                self._duckdb_path = None
                self._duckdb_candles_tables = set()
                self.df_ticks = pd.read_parquet(self.parquet_file)
            
            # 1. 时区处理
            # 用户反馈数据源(QDM)已导出为美东时间 (America/New_York)
            # 如果是 Naive Time，直接视为美东时间，不再假设为 UTC
            if self.df_ticks.index.tz is None:
                self.df_ticks.index = self.df_ticks.index.tz_localize('America/New_York')
            else:
                # 如果自带时区，则转换为美东时间 (以防万一)
                self.df_ticks.index = self.df_ticks.index.tz_convert('America/New_York')
            
            self.df_ticks.sort_index(inplace=True)
            self._candles_cache.clear()
            
            print(f"Loaded {len(self.df_ticks)} ticks. Time range: {self.df_ticks.index[0]} - {self.df_ticks.index[-1]}")
            
        except Exception as e:
            print(f"Error loading data: {e}")

    def _duckdb_table_for_timeframe(self, timeframe):
        tf = str(timeframe).strip().lower()
        if tf.endswith("s"):
            suffix = tf
        elif tf.endswith("min"):
            suffix = f"{tf[:-3]}m"
        elif tf.endswith("h"):
            suffix = tf
        elif tf.endswith("d"):
            suffix = f"{tf[:-1]}D"
        elif tf.endswith("w"):
            suffix = f"{tf[:-1]}W"
        elif tf.endswith("m") and not tf.endswith("min"):
            suffix = f"{tf[:-1]}M"
        else:
            return None
        return f"candles_{suffix}"

    def _period_from_table(self, table_name):
        suffix = table_name.replace("candles_", "")
        if suffix.endswith("m") and suffix[:-1].isdigit():
            return f"{suffix[:-1]}min"
        return suffix

    def get_candles(self, timeframe='1min'):
        """
        将 Tick 数据重采样为 OHLCV K线数据 (全量)
        支持纽约时间切分 (NY Close at 17:00)
        """
        if self.df_ticks is None:
            return None
        if timeframe in self._candles_cache:
            return self._candles_cache[timeframe]

        if self._duckdb_path:
            table = self._duckdb_table_for_timeframe(timeframe)
            if table and table in self._duckdb_candles_tables:
                try:
                    import duckdb
                    con = duckdb.connect(self._duckdb_path, read_only=True)
                    df = con.execute(f"SELECT * FROM {table} ORDER BY timestamp").df()
                    con.close()
                    if "timestamp" not in df.columns and "time" in df.columns:
                        df = df.rename(columns={"time": "timestamp"})
                    if "timestamp" not in df.columns and "datetime" in df.columns:
                        df = df.rename(columns={"datetime": "timestamp"})
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df.set_index("timestamp", inplace=True)
                    self._candles_cache[timeframe] = df
                    return df
                except Exception as e:
                    print(f"Error loading DuckDB candles for {timeframe}: {e}")

        # 规范化周期格式
        # Pandas 新版本推荐使用 'h' 而不是 'H'，这里不再强制大写

        print(f"Resampling to {timeframe}...")
        
        # 为了实现 "NY Close" (17:00 对齐) 且不受夏令时漂移影响，
        # 我们需要先转换为 Naive Time (墙上时间) 再 Resample
        # 创建临时 Series 以避免复制整个 DataFrame
        naive_index = self.df_ticks.index.tz_localize(None)
        
        price_series = pd.Series(self.df_ticks['price'].values, index=naive_index)
        vol_series = pd.Series(
            self.df_ticks['volume'].values, index=naive_index, name="volume"
        )
        
        # 设置锚点为 17:00
        origin_ts = pd.Timestamp("2000-01-01 17:00:00")
        
        # 1. 价格 OHLC (明确左闭左标)
        timeframe_norm = self._normalize_timeframe(timeframe)
        ohlc = price_series.resample(timeframe_norm, closed='left', label='left', origin=origin_ts).ohlc()
        
        # 2. 成交量 Sum
        vol = vol_series.resample(timeframe_norm, closed='left', label='left', origin=origin_ts).sum()
        
        # 3. 合并
        df_candles = pd.concat([ohlc, vol], axis=1)
        if not self._should_dropna(timeframe):
            full_index = self._build_full_index(
                naive_index[0], naive_index[-1], timeframe_norm, origin_ts
            )
            if self._should_filter_by_calendar(timeframe):
                full_index = self._filter_index_by_calendar(full_index, timeframe)
            df_candles = df_candles.reindex(full_index)

        
        # 4. 清洗
        if self._should_dropna(timeframe):
            df_candles.dropna(inplace=True)

        # 调试：打印前 5 行 K 线 (清洗后)
        print(f"First 5 candles ({timeframe}):")
        print(df_candles.head())
        
        # 5. 计算指标
        
        # 5. 计算指标
        df_candles = self._calculate_indicators(df_candles)
        
        self._candles_cache[timeframe] = df_candles
        return df_candles

    def get_candles_by_time(self, timeframe, end_time, count=200):
        """
        获取截止到 end_time 的最近 count 根 K线
        用于回放动态刷新
        """
        if self.df_ticks is None:
            return None

        if timeframe in self._candles_cache:
            df_full = self._candles_cache[timeframe]
            if df_full is None or df_full.empty:
                return None
            end_ts = pd.Timestamp(end_time)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize(None)
            df_slice = df_full.loc[:end_ts]
            if df_slice.empty:
                return None
            return df_slice.tail(count)
        # Build once for replay if not cached.
        df_full = self.get_candles(timeframe)
        if df_full is not None and not df_full.empty:
            end_ts = pd.Timestamp(end_time)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_localize(None)
            df_slice = df_full.loc[:end_ts]
            if df_slice.empty:
                return None
            return df_slice.tail(count)
            
        # 1. 截取到当前回放时间点的数据
        mask = self.df_ticks.index <= end_time
        # 为了指标计算准确，需要多截取一些历史数据
        # 假设最大周期 60，我们需要至少前 200 根来让 EMA 稳定
        # 所以这里我们不限制 recent_ticks 的起始点，或者限制得宽松一些
        # 暂时全量截取，如果性能有问题再优化
        recent_ticks = self.df_ticks.loc[mask]
        
        if len(recent_ticks) == 0:
            return None

        # 转换 Naive Time 用于对齐
        naive_index = recent_ticks.index.tz_localize(None)
        price_series = pd.Series(recent_ticks['price'].values, index=naive_index)
        vol_series = pd.Series(
            recent_ticks['volume'].values, index=naive_index, name="volume"
        )
        
        origin_ts = pd.Timestamp("2000-01-01 17:00:00")

        # 2. 合成 K 线
        timeframe_norm = self._normalize_timeframe(timeframe)
        ohlc = price_series.resample(timeframe_norm, closed='left', label='left', origin=origin_ts).ohlc()
        vol = vol_series.resample(timeframe_norm, closed='left', label='left', origin=origin_ts).sum()
        df_candles = pd.concat([ohlc, vol], axis=1)
        if not self._should_dropna(timeframe):
            full_index = self._build_full_index(
                naive_index[0], naive_index[-1], timeframe_norm, origin_ts
            )
            if self._should_filter_by_calendar(timeframe):
                full_index = self._filter_index_by_calendar(full_index, timeframe)
            df_candles = df_candles.reindex(full_index)
        if self._should_dropna(timeframe):
            df_candles.dropna(inplace=True)
        
        # 3. 计算指标 (在切片前计算，保证数值准确)
        df_candles = self._calculate_indicators(df_candles)
        
        # 4. 只返回最近的 count 根
        return df_candles.tail(count)

    def _calculate_indicators(self, df):
        """计算 EMA 和 布林带"""
        # EMA
        for span in [20, 30, 40, 50, 60]:
            df[f'EMA{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        
        # Bollinger Bands (20, 2)
        # 很多软件用的是 SMA 作为中轨
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['BB_Upper'] = sma20 + 2 * std20
        df['BB_Lower'] = sma20 - 2 * std20
        # df['BB_Mid'] = sma20 # 如果需要画中轨

        # MACD (12, 26, 9)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI (6/12/24 + 14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        for rsi_len in [6, 12, 24, 14]:
            avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            if rsi_len == 14:
                df['RSI'] = 100 - (100 / (1 + rs))
            else:
                df[f'RSI{rsi_len}'] = 100 - (100 / (1 + rs))

        return df


    def _should_dropna(self, timeframe):
        tf = str(timeframe).strip().lower()
        if tf.endswith("min"):
            try:
                minutes = int(tf[:-3])
            except ValueError:
                return True
            return minutes <= 60
        if tf.endswith("h"):
            try:
                int(tf[:-1])
            except ValueError:
                return True
            return True
        if tf.endswith("d") or tf.endswith("w") or (tf.endswith("m") and not tf.endswith("min")):
            return True
        return False

    def _normalize_timeframe(self, timeframe):
        tf = str(timeframe).strip()
        if tf.endswith("M") and tf[:-1].isdigit():
            return f"{tf[:-1]}ME"
        return tf

    def _should_filter_by_calendar(self, timeframe):
        tf = str(timeframe).strip().lower()
        if tf.endswith("min") or tf.endswith("h"):
            return True
        return False


    def _build_full_index(self, start_ts, end_ts, timeframe, origin_ts):
        if start_ts > end_ts:
            return pd.DatetimeIndex([])
        base = pd.date_range(start=origin_ts, end=end_ts, freq=timeframe)
        if base.empty:
            return base
        pos = base.searchsorted(start_ts, side="right") - 1
        if pos < 0:
            start = base[0]
        else:
            start = base[pos]
        return pd.date_range(start=start, end=end_ts, freq=timeframe)

    def _get_calendar(self):
        if self._calendar is not None:
            return self._calendar
        try:
            import pandas_market_calendars as mcal
        except Exception:
            self._calendar = None
            return None
        for name in [self.calendar_name, "CME_FX", "CME"]:
            if not name:
                continue
            try:
                self._calendar = mcal.get_calendar(name)
                return self._calendar
            except Exception:
                continue
        self._calendar = None
        return None

    def _filter_index_by_calendar(self, full_index, timeframe):
        if full_index is None or full_index.empty:
            return full_index

        cal = self._get_calendar()
        if cal is None:
            return full_index

        try:
            offset = pd.tseries.frequencies.to_offset(timeframe)
            freq_delta = offset.delta
        except Exception:
            return full_index
        if freq_delta is None:
            return full_index

        # Build schedule slightly wider than the data range.
        start_date = (full_index[0] - pd.Timedelta(days=2)).date()
        end_date = (full_index[-1] + pd.Timedelta(days=2)).date()
        try:
            schedule = cal.schedule(start_date=start_date, end_date=end_date)
        except Exception:
            return full_index
        if schedule is None or schedule.empty:
            return full_index

        try:
            schedule = schedule.tz_convert("America/New_York")
        except Exception:
            # If schedule is naive, assume NY wall time.
            schedule = schedule.tz_localize("America/New_York")

        opens = schedule["market_open"].dt.tz_localize(None)
        closes = schedule["market_close"].dt.tz_localize(None)

        mask = np.zeros(len(full_index), dtype=bool)
        for open_ts, close_ts in zip(opens, closes):
            # Keep bars whose interval overlaps the trading session.
            start = open_ts - freq_delta
            end = close_ts
            left = full_index.searchsorted(start, side="left")
            right = full_index.searchsorted(end, side="left")
            if right > left:
                mask[left:right] = True

        return full_index[mask]

if __name__ == "__main__":
    pass
