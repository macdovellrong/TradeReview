import pandas as pd
import sys
from pathlib import Path

def read_tick_data(file_path, n=10):
    """
    读取CSV文件并打印前N行
    
    参数:
        file_path: CSV文件的完整路径或相对路径
        n: 打印的行数，默认10
    """
    path = Path(file_path)
    
    # 检查文件是否存在
    if not path.exists():
        print(f"错误: 文件 {path} 不存在")
        return
    
    try:
        # 读取CSV文件
        df = pd.read_csv(path)
        
        # 打印文件信息
        print(f"文件: {path.name}")
        print(f"总行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"\n列名: {list(df.columns)}\n")
        
        # 打印前N行
        print(f"前 {n} 行数据:")
        print("-" * 120)
        print(df.head(n).to_string())
        
    except Exception as e:
        print(f"错误: 读取文件失败 - {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python read_csv.py <file_path> [n]")
        print("示例: python read_csv.py '.\\data\\tick.csv' 20")
        sys.exit(1)
    
    file_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    read_tick_data(file_path, n)