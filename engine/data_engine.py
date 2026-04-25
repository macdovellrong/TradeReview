import os
import pandas as pd
import numpy as np

from engine.data_validation import (
    CURRENT_CANDLE_INDICATOR_COLUMNS,
    inspect_duckdb_schema,
    normalize_candle_dataframe,
    normalize_tick_dataframe,
    validate_duckdb_candle_table,
)

class DataEngine:
    def __init__(self, parquet_file="data/ticks.parquet"):
        self.parquet_file = parquet_file
        self.df_ticks = None
        self.calendar_name = "CME_FX"
        self._calendar = None
        self._candles_cache = {}
        self._duckdb_path = None
        self._duckdb_candles_tables = set()
        self.tick_start = None
        self.tick_end = None
        self.tick_count = 0
        self.last_load_error = None
        self.last_load_warnings = []
        self.load_data()

    def load_data(self):
        """加载 Tick 数据并进行基础处理"""
        print(f"Loading data from {self.parquet_file}...")
        self.last_load_error = None
        self.last_load_warnings = []
        self.df_ticks = None
        self.tick_start = None
        self.tick_end = None
        self.tick_count = 0
        self._candles_cache.clear()
        try:
            if not self.parquet_file:
                return

            ext = os.path.splitext(self.parquet_file)[1].lower()
            if ext == ".duckdb":
                try:
                    import duckdb
                except Exception as e:
                    self.last_load_error = f"Error loading DuckDB: {e}"
                    print(f"Error loading DuckDB: {e}")
                    return

                self._duckdb_path = self.parquet_file
                con = duckdb.connect(self.parquet_file, read_only=True)
                try:
                    report = inspect_duckdb_schema(con, self.parquet_file)
                    self._duckdb_candles_tables = report.candle_tables
                    if self._duckdb_candles_tables:
                        periods = [self._period_from_table(t) for t in sorted(self._duckdb_candles_tables)]
                        print(f"DuckDB candle tables: {', '.join(periods)}")
                        if "candles_1M" in self._duckdb_candles_tables and "candles_1m" not in self._duckdb_candles_tables:
                            warning = (
                                "Legacy DuckDB naming detected: candles_1M exists but candles_1m is missing. "
                                "Older converters let the monthly table collide with the 1-minute table. "
                                "1-minute candles will be rebuilt from ticks; rebuild the DuckDB to restore the precomputed 1-minute table."
                            )
                            self.last_load_warnings.append(warning)
                            print(f"Warning: {warning}")
                        valid_candle_tables = set()
                        for table_name in sorted(self._duckdb_candles_tables):
                            try:
                                validate_duckdb_candle_table(
                                    con,
                                    self.parquet_file,
                                    table_name,
                                    allow_gap_rows=self._table_allows_gap_rows(table_name),
                                )
                            except Exception as e:
                                warning = (
                                    f"Skipping invalid precomputed candle table {table_name}: {e}. "
                                    "This timeframe will be rebuilt from ticks when requested."
                                )
                                self.last_load_warnings.append(warning)
                                print(f"Warning: {warning}")
                                continue
                            valid_candle_tables.add(table_name)
                        self._duckdb_candles_tables = valid_candle_tables

                    row = con.execute(
                        """
                        SELECT
                            count(*) AS row_count,
                            min(timestamp) AS min_ts,
                            max(timestamp) AS max_ts
                        FROM ticks
                        """
                    ).fetchone()
                    self.tick_count = int(row[0] or 0)
                    self.tick_start = pd.Timestamp(row[1]) if row[1] is not None else None
                    self.tick_end = pd.Timestamp(row[2]) if row[2] is not None else None
                finally:
                    con.close()
                print(
                    f"Loaded DuckDB metadata. Ticks: {self.tick_count}. "
                    f"Time range: {self.tick_start} - {self.tick_end}"
                )
                return
            else:
                self._duckdb_path = None
                self._duckdb_candles_tables = set()
                self.df_ticks = normalize_tick_dataframe(pd.read_parquet(self.parquet_file), self.parquet_file)
            
            # 1. 时区处理
            # 用户反馈数据源(QDM)已导出为美东时间 (America/New_York)
            # 如果是 Naive Time，直接视为美东时间，不再假设为 UTC
            if self.df_ticks.index.tz is None:
                self.df_ticks.index = self.df_ticks.index.tz_localize('America/New_York')
            else:
                # 如果自带时区，则转换为美东时间 (以防万一)
                self.df_ticks.index = self.df_ticks.index.tz_convert('America/New_York')
            
            self.df_ticks.sort_index(inplace=True)
            self.tick_count = len(self.df_ticks)
            self.tick_start = self.df_ticks.index[0]
            self.tick_end = self.df_ticks.index[-1]
            
            print(f"Loaded {len(self.df_ticks)} ticks. Time range: {self.df_ticks.index[0]} - {self.df_ticks.index[-1]}")
            
        except Exception as e:
            self.last_load_error = str(e)
            self.df_ticks = None
            self.tick_start = None
            self.tick_end = None
            self.tick_count = 0
            self._candles_cache.clear()
            print(f"Error loading data: {e}")

    def _duckdb_table_for_timeframe(self, timeframe):
        tf = str(timeframe).strip().lower()
        candidates = []
        if tf.endswith("s"):
            candidates.append(f"candles_{tf}")
        elif tf.endswith("min"):
            candidates.append(f"candles_{tf[:-3]}m")
        elif tf.endswith("h"):
            candidates.append(f"candles_{tf}")
        elif tf.endswith("d"):
            candidates.append(f"candles_{tf[:-1]}D")
        elif tf.endswith("w"):
            candidates.append(f"candles_{tf[:-1]}W")
        elif tf.endswith("m") and not tf.endswith("min"):
            candidates.append(f"candles_{tf[:-1]}mo")
            candidates.append(f"candles_{tf[:-1]}M")
        else:
            return None
        for table_name in candidates:
            if table_name in self._duckdb_candles_tables:
                return table_name
        return candidates[0]

    def _table_allows_gap_rows(self, table_name):
        suffix = table_name.replace("candles_", "").lower()
        if suffix.endswith("s") and suffix[:-1].isdigit():
            return True
        if suffix.endswith("m") and suffix[:-1].isdigit():
            return int(suffix[:-1]) > 60
        return False

    def _period_from_table(self, table_name):
        suffix = table_name.replace("candles_", "")
        if suffix.endswith("mo") and suffix[:-2].isdigit():
            return f"{suffix[:-2]}M"
        if suffix.endswith("m") and suffix[:-1].isdigit():
            return f"{suffix[:-1]}min"
        return suffix

    def get_candles(self, timeframe='1min'):
        """
        将 Tick 数据重采样为 OHLCV K线数据 (全量)
        支持纽约时间切分 (NY Close at 17:00)
        """
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
                    df = normalize_candle_dataframe(
                        df,
                        f"{self._duckdb_path}::{table}",
                        allow_gap_rows=self._table_allows_gap_rows(table),
                    )
                    if any(col not in df.columns for col in CURRENT_CANDLE_INDICATOR_COLUMNS):
                        df = self._calculate_indicators(df)
                    self._candles_cache[timeframe] = df
                    return df
                except Exception as e:
                    print(f"Error loading DuckDB candles for {timeframe}: {e}")

        # 规范化周期格式
        # Pandas 新版本推荐使用 'h' 而不是 'H'，这里不再强制大写

        if self.df_ticks is None:
            return None

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

    def get_candles_window(self, timeframe, start_time, end_time):
        if not self._duckdb_path:
            df_full = self.get_candles(timeframe)
            if df_full is None or df_full.empty:
                return None
            start_ts = pd.Timestamp(start_time)
            end_ts = pd.Timestamp(end_time)
            if df_full.index.tz is not None and start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(df_full.index.tz)
                end_ts = end_ts.tz_localize(df_full.index.tz)
            elif df_full.index.tz is None and start_ts.tzinfo is not None:
                start_ts = start_ts.tz_localize(None)
                end_ts = end_ts.tz_localize(None)
            return df_full.loc[start_ts:end_ts]

        table = self._duckdb_table_for_timeframe(timeframe)
        if table not in self._duckdb_candles_tables:
            return None

        import duckdb

        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_localize(None)

        con = duckdb.connect(self._duckdb_path, read_only=True)
        try:
            df = con.execute(
                f"""
                SELECT *
                FROM {table}
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
                """,
                [start_ts.to_pydatetime(), end_ts.to_pydatetime()],
            ).df()
        finally:
            con.close()

        if df.empty:
            empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            empty.index = pd.DatetimeIndex([], name="timestamp")
            return empty

        return normalize_candle_dataframe(
            df,
            f"{self._duckdb_path}::{table}",
            allow_gap_rows=self._table_allows_gap_rows(table),
        )

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
        for span in [20, 30, 40, 50, 60, 100, 240]:
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
        if tf.endswith("me"):
            base = tf[:-2]
            return base.isdigit()
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
