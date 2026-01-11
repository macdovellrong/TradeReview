import pandas as pd
import numpy as np

class DataEngine:
    def __init__(self, parquet_file="data/ticks.parquet"):
        self.parquet_file = parquet_file
        self.df_ticks = None
        self.load_data()

    def load_data(self):
        """加载 Tick 数据并进行基础处理"""
        print(f"Loading data from {self.parquet_file}...")
        try:
            self.df_ticks = pd.read_parquet(self.parquet_file)
            
            # 1. 时区转换: UTC -> America/New_York
            # 假设 Parquet 里的时间是 UTC (Naive -> UTC -> NY)
            # 如果已经是 UTC aware (有+00:00), 这一步可能要微调
            if self.df_ticks.index.tz is None:
                self.df_ticks.index = self.df_ticks.index.tz_localize('UTC')
            
            self.df_ticks.index = self.df_ticks.index.tz_convert('America/New_York')
            
            print(f"Loaded {len(self.df_ticks)} ticks. Time range: {self.df_ticks.index[0]} - {self.df_ticks.index[-1]}")
            
        except Exception as e:
            print(f"Error loading data: {e}")

    def get_candles(self, timeframe='1min'):
        """
        将 Tick 数据重采样为 OHLCV K线数据 (全量)
        """
        if self.df_ticks is None:
            return None

        print(f"Resampling to {timeframe}...")
        
        # 1. 价格 OHLC
        ohlc = self.df_ticks['price'].resample(timeframe).ohlc()
        
        # 2. 成交量 Sum
        vol = self.df_ticks['volume'].resample(timeframe).sum()
        
        # 3. 合并
        df_candles = pd.concat([ohlc, vol], axis=1)
        
        # 4. 清洗
        df_candles.dropna(inplace=True)
        
        # 5. 计算指标
        df_candles = self._calculate_indicators(df_candles)
        
        return df_candles

    def get_candles_by_time(self, timeframe, end_time, count=200):
        """
        获取截止到 end_time 的最近 count 根 K线
        用于回放动态刷新
        """
        if self.df_ticks is None:
            return None
            
        # 1. 截取到当前回放时间点的数据
        mask = self.df_ticks.index <= end_time
        # 为了指标计算准确，需要多截取一些历史数据
        # 假设最大周期 60，我们需要至少前 200 根来让 EMA 稳定
        # 所以这里我们不限制 recent_ticks 的起始点，或者限制得宽松一些
        # 暂时全量截取，如果性能有问题再优化
        recent_ticks = self.df_ticks.loc[mask]
        
        if len(recent_ticks) == 0:
            return None

        # 2. 合成 K 线
        ohlc = recent_ticks['price'].resample(timeframe).ohlc()
        vol = recent_ticks['volume'].resample(timeframe).sum()
        df_candles = pd.concat([ohlc, vol], axis=1).dropna()
        
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
        
        return df

if __name__ == "__main__":
    pass
