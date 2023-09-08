import pandas as pd
import glob

def read_all_data(csv_dir):
    # 获取目录中所有CSV文件的文件名
    csv_files = glob.glob(csv_dir + "*.csv")

    # 创建一个空的DataFrame，用于存储所有CSV文件的数据
    combined_data = pd.DataFrame()

    # 循环读取每个CSV文件并将其合并到combined_data中
    for file in csv_files:
        # 使用read_csv函数读取CSV文件
        data = pd.read_csv(file, header=None)
        # print(data)
        # 将数据追加到combined_data中
        combined_data = pd.concat([combined_data, data], axis=1, ignore_index=True)

    # 现在，combined_data包含了所有CSV文件的数据，每个CSV文件的行数仍然保持不变
    return combined_data