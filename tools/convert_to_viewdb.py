import duckdb
import pandas as pd
import sys
from pathlib import Path

def csv_to_duckdb(csv_file, db_file=None):
    """
    将CSV文件转换为DuckDB数据库
    
    参数:
        csv_file: CSV文件路径
        db_file: DuckDB数据库文件路径，默认为CSV文件名.duckdb
    """
    csv_path = Path(csv_file)
    
    # 检查文件是否存在
    if not csv_path.exists():
        print(f"错误: 文件 {csv_path} 不存在")
        return
    
    # 确定数据库文件名
    if db_file is None:
        db_file = csv_path.stem + '.duckdb'
    
    db_path = Path(db_file)
    
    try:
        # 连接DuckDB
        conn = duckdb.connect(str(db_path))
        
        # 创建表名（使用CSV文件名去掉扩展名，添加tbl_前缀）
        table_name = 'tbl_' + csv_path.stem.replace('-', '_').replace('.', '_').replace(' ', '_')
        
        # 读取CSV并导入到DuckDB
        print(f"正在转换: {csv_path.name}")
        df = pd.read_csv(csv_path)
        
        # 注册DataFrame为DuckDB表
        conn.register('temp_df', df)
        
        # 创建表
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" AS
            SELECT * FROM temp_df
        """)
        
        # 获取表信息
        result = conn.execute(f'SELECT COUNT(*) as row_count FROM "{table_name}"').fetchall()
        row_count = result[0][0]
        
        # 获取列信息
        columns = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        
        print(f"\n✓ 转换成功!")
        print(f"数据库文件: {db_path}")
        print(f"表名: {table_name}")
        print(f"总行数: {row_count}")
        print(f"\n列信息:")
        for col in columns:
            print(f"  {col[0]:15} {col[1]}")
        
        # 显示前5行预览
        print(f"\n前5行数据预览:")
        print("-" * 80)
        preview = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 5').df()
        print(preview.to_string())
        
        conn.close()
        
    except Exception as e:
        print(f"错误: 转换失败 - {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python csv_to_duckdb.py <csv_file> [db_file]")
        print("示例: python csv_to_duckdb.py '.\\data\\tick.csv'")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    db_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    csv_to_duckdb(csv_file, db_file)