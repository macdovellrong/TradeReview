import pandas as pd

file_path = "2026-XAUUSD-TICK-No Session.csv"

# 只读取前 10 行
try:
    df = pd.read_csv(file_path, nrows=10)
    print("CSV Header:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # 检查是否有显而易见的时间列
    print("\nData Types:")
    print(df.dtypes)
    
except Exception as e:
    print(f"Error reading CSV: {e}")

