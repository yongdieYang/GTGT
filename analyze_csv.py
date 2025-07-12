import pandas as pd

def analyze_csv(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 获取行数和列数
        num_rows = len(df)
        num_cols = len(df.columns)
        
        # 打印基本信息
        print(f"文件 '{file_path}' 的基本信息：")
        print(f"行数：{num_rows}")
        print(f"列数：{num_cols}")
        print("\n列名：")
        for col in df.columns:
            print(f"- {col}")
            
        # 打印数据预览
        print("\n数据预览（前5行）：")
        print(df.head())
        
        # 打印数据类型信息
        print("\n数据类型信息：")
        print(df.dtypes)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    # 替换为您的CSV文件路径
    file_path = r"E:\yyd-正\gogogo\yyd\raw\yyd_data_train.csv"
    analyze_csv(file_path)