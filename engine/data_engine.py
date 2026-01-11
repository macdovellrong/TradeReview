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

    def get_candles_by_time(self, timeframe, end_time, count=200):
        """
        获取截止到 end_time 的最近 count 根 K线
        用于回放动态刷新
        """
        if self.df_ticks is None:
            return None
            
        # 1. 截取到当前回放时间点的数据
        mask = self.df_ticks.index <= end_time
        # 为了性能，如果数据量太大，可以只截取最近的一段 Tick 数据，这里暂且全量截取
        recent_ticks = self.df_ticks.loc[mask]
        
        if len(recent_ticks) == 0:
            return None

        # 2. 合成 K 线
        ohlc = recent_ticks['price'].resample(timeframe).ohlc()
        vol = recent_ticks['volume'].resample(timeframe).sum()
        df_candles = pd.concat([ohlc, vol], axis=1).dropna()
        
        # 3. 只返回最近的 count 根，保证绘图流畅
        return df_candles.tail(count)

if __name__ == "__main__":
    pass
