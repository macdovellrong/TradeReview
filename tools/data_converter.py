import pandas as pd
import os

# 配置
CSV_FILE = "2026-XAUUSD-TICK-No Session.csv"
PARQUET_FILE = "ticks.parquet"

def convert_csv_to_parquet():
    if not os.path.exists(CSV_FILE):
        print(f"Error: File {CSV_FILE} not found.")
        return

    print(f"Reading {CSV_FILE}...")
    
    # 1. 读取 CSV
    # 使用 chunksize 分块读取，防止内存溢出（如果文件非常大）
    # 但对于 ~100MB 的文件，直接读取也完全没问题。这里直接读取。
    df = pd.read_csv(CSV_FILE)
    
    print("Parsing timestamps...")
    # 2. 解析时间
    # 格式: 20260101 23:00:00.596
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S.%f')
    
    # 3. 处理列
    # 保留 Bid 作为 Price，以及 Volume
    # 这里的 Volume 是 Tick Volume，不是真实的成交量，但在外汇/CFD中通常就用这个
    df_clean = pd.DataFrame({
        'time': df['DateTime'],
        'price': df['Bid'],
        'volume': df['Volume']
    })
    
    # 4. 设置索引 (Parquet 对索引支持很好)
    df_clean.set_index('time', inplace=True)
    
    print(f"Saving to {PARQUET_FILE}...")
    # 5. 保存为 Parquet
    # compression='zstd' 压缩率和速度都很优秀
    df_clean.to_parquet(PARQUET_FILE, engine='pyarrow', compression='zstd')
    
    print("Conversion complete!")
    print(f"Original CSV size: {os.path.getsize(CSV_FILE) / 1024 / 1024:.2f} MB")
    print(f"Parquet file size: {os.path.getsize(PARQUET_FILE) / 1024 / 1024:.2f} MB")
    
    # 验证读取
    print("\nVerifying data...")
    df_check = pd.read_parquet(PARQUET_FILE)
    print(df_check.head())
    print(f"\nTotal ticks: {len(df_check)}")

if __name__ == "__main__":
    convert_csv_to_parquet()
