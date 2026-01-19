import pandas as pd
import os
import glob
from datetime import datetime

def convert_csv_to_parquet():
    # 1. 查找 CSV 文件
    csv_files = glob.glob("*.csv") + glob.glob("data/*.csv")
    if not csv_files:
        print("未找到 CSV 文件。请将 QDM 导出的 CSV 放入项目根目录或 data/ 目录。")
        return

    print("找到以下 CSV 文件:")
    for i, f in enumerate(csv_files):
        print(f"{i + 1}. {f}")
    
    choice = input("\n请输入文件编号进行转换 (默认 1): ").strip()
    if not choice:
        choice = 0
    else:
        choice = int(choice) - 1
        
    file_path = csv_files[choice]
    output_path = os.path.join("data", os.path.splitext(os.path.basename(file_path))[0] + ".parquet")
    
    print(f"\n正在读取 {file_path} ...")
    
    # 2. 读取 CSV (尝试自动推断格式)
    # QDM 导出通常没有 Header，或者有特定的 Header
    # 先读取几行看看
    preview = pd.read_csv(file_path, nrows=5)
    print("前 5 行预览:")
    print(preview)
    
    has_header = input("\nCSV 是否包含标题行? (y/n, 默认 y): ").strip().lower() != 'n'
    
    if has_header:
        df = pd.read_csv(file_path)
    else:
        # 如果没有标题，通常 QDM 格式是: Date, Time, Bid, Ask, BidVol, AskVol (Tick)
        # 或者 Date, Time, Price, Volume
        print("假设无标题格式为: Date, Time, Price, Volume (或类似)")
        # 这里需要用户确认列索引，为了简化，先读取为默认列名
        df = pd.read_csv(file_path, header=None)
        print("列名:", df.columns.tolist())
        
        date_col = int(input("请输入【日期】列的索引 (例如 0): "))
        time_col = int(input("请输入【时间】列的索引 (例如 1): "))
        price_col = int(input("请输入【价格】列的索引 (例如 2): "))
        vol_col = int(input("请输入【成交量】列的索引 (如果有，例如 3，否则回车跳过): ") or -1)
        
        # 重命名
        rename_map = {
            df.columns[date_col]: 'date',
            df.columns[time_col]: 'time',
            df.columns[price_col]: 'price'
        }
        if vol_col != -1:
            rename_map[df.columns[vol_col]] = 'volume'
            
        df.rename(columns=rename_map, inplace=True)

    # 3. 处理列名 (如果有标题，尝试自动识别)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 尝试找到 日期/时间/价格/量
    # 支持 QDM 导出的 'datetime' 这种合并列
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'date' in df.columns and 'time' in df.columns:
        print("合并 Date 和 Time 列...")
        df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    else:
        # 如果上面手动指定了，应该已经有了
        if 'date' in df.columns and 'time' in df.columns:
             df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        else:
            print("错误: 无法识别日期时间列。请确保列名为 date/time 或 datetime")
            return

    # 4. 设置索引
    df.set_index('datetime', inplace=True)
    
    # 5. 处理时区 (关键步骤)
    print("正在处理时区...")
    # 假设源数据已经是美东时间 (America/New_York)
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
        
    # 6. 筛选必要列
    cols_to_keep = []
    
    # 价格逻辑优化：优先查找 price, 然后是 bid (QDM 常用), 然后是 close
    if 'price' in df.columns:
        cols_to_keep.append('price')
    elif 'bid' in df.columns:
        df['price'] = df['bid'] # 将 Bid 作为主价格
        cols_to_keep.append('price')
    elif 'close' in df.columns:
        df['price'] = df['close']
        cols_to_keep.append('price')
        
    # 成交量
    if 'volume' in df.columns:
        cols_to_keep.append('volume')
    elif 'vol' in df.columns:
        df['volume'] = df['vol']
        cols_to_keep.append('volume')
    else:
        df['volume'] = 0 # 填充 0
        cols_to_keep.append('volume')
        
    df = df[cols_to_keep]
    
    # 7. 排序
    df.sort_index(inplace=True)
    
    # 8. 保存
    print(f"正在保存到 {output_path} ...")
    df.to_parquet(output_path, compression='zstd') # 使用 zstd 压缩，速度快体积小
    print("转换完成！")
    print(f"包含 {len(df)} 条 Tick 数据。")
    print(f"时间范围: {df.index[0]} - {df.index[-1]}")

if __name__ == "__main__":
    try:
        convert_csv_to_parquet()
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
