import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
import os

import glob

import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore", category=UserWarning)



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

def plot_one(data_dir):
    data = pd.read_csv(data_dir, header=None)

    state = data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                    for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
    
    # npstate是笛卡尔坐标数据
    trajectory = npstate

    # 提取x、y、z坐标
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # 创建三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹线
    ax.plot(x, y, z, marker='o', markersize=5)

    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 显示图像
    plt.show()

def plot_all(csv_dir):
    # 获取目录中所有CSV文件的文件名
    csv_files = glob.glob(csv_dir + "*.csv")
    print(csv_files)

    # 创建三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 循环读取每个CSV文件并绘制轨迹线
    for file in csv_files:
        print(file)
        data = pd.read_csv(file, header=None)
        state = data.iloc[0].to_numpy()
        npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                        for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
        x = npstate[:, 0]
        y = npstate[:, 1]
        z = npstate[:, 2]
        ax.plot(x, y, z, marker='o', label=file[-9:-4])  # 使用文件名作为标签

    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 添加图例
    ax.legend()

    # 显示图像
    plt.show()

def plot_np(nparry):
    
    # npstate是笛卡尔坐标数据
    trajectory = nparry

    # 提取x、y、z坐标
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # 创建三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', label = 'test')

    # 绘制轨迹线
    ax.plot(x, y, z, marker='o', markersize=1)

    # 绘制标准轨迹
    data = pd.read_csv("B:/code/kortex/imitation_learning/v2/data/simulated_rule/data1.csv", header=None)
    state = data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                    for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
    x = npstate[:, 0]
    y = npstate[:, 1]
    z = npstate[:, 2]
    ax.plot(x, y, z, marker='o', label="standard", markersize=1)  # 使用文件名作为标签


    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 显示图像
    plt.show()

    
def draw_np(nparry, pic_dir):
    
    # npstate是笛卡尔坐标数据
    trajectory = nparry

    # 提取x、y、z坐标
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    # 创建三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', label = 'test')

    # 绘制轨迹线
    ax.plot(x, y, z, marker='o', markersize=1)

    # 绘制标准轨迹
    data = pd.read_csv("/home/lsy/Projects/kinova-imitation-learning/imitation_learning/v2/data/simulated_rule/data1.csv", header=None)
    state = data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                    for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
    x = npstate[:, 0]
    y = npstate[:, 1]
    z = npstate[:, 2]
    ax.plot(x, y, z, marker='o', label="standard", markersize=1)  # 使用文件名作为标签


    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    plt.savefig(pic_dir)

def plot_4dnp(nparry, pic_dir):
    
    # npstate是笛卡尔坐标数据
    trajectory = nparry

    # 提取x、y、z坐标
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    a = trajectory[:, 3]

    # 创建三维图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', label = 'test')

    # 绘制轨迹线
    # ax.plot(x, y, z, marker='o', markersize=1)
    colors = np.where(a < 0.01, 'green', np.where(a > 0.3, 'red', 'black'))
    for i in range(1, len(x)):
        ax.plot([x[i - 1], x[i]], [y[i - 1], y[i]], [z[i - 1], z[i]], c=colors[i], markersize=4)

    # 绘制标准轨迹
    data = pd.read_csv("/home/lsy/Projects/kinova-imitation-learning/imitation_learning/v2/data/simulated_rule/data1.csv", header=None)
    state = data.iloc[0].to_numpy()
    npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                    for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
    x = npstate[:, 0]
    y = npstate[:, 1]
    z = npstate[:, 2]
    a = npstate[:, 3]
    
    ax.plot(x, y, z, marker='o', label="standard", markersize=1)  # 使用文件名作为标签


    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    
    plt.savefig(pic_dir)

# # data
# dir = str(Path.cwd())
# data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/data1.csv"
# csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/"
# plot_one(data_dir)
# plot_all(csv_dir)



