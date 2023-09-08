# 三层隐层 √
# 归一化
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime
import os

from readdata import read_all_data
from plot import plot_np

def normalize_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def origin_data(data):
    min_vals = np.array([0.299900302, -0.17102845, 0.05590736, -0.000087572115])
    max_vals = np.array([0.58015204, 0.00020189775, 0.299989649, 0.36635616])
    origin_data_data = (max_vals - min_vals) * data + min_vals
    return origin_data_data

def overlay_frames(state, action, frame_length):
    inputs = []
    outputs = []
    for i in range(len(state) - frame_length):
        input_sequence = state[i:(i + frame_length)].flatten()
        output_sequence = action[i + frame_length]  # 使用npaction的第四行作为输出
        inputs.append(input_sequence)
        outputs.append(output_sequence)
    return np.array(inputs), np.array(outputs)

# data
def read_data(if_all, if_delt, if_test):
    if not if_test:
        # dir = str(Path.cwd())
        # data_dir = "./data/rule/data1.csv"
        # csv_dir = "./data/rule/"
        data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/data1.csv"
        csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/"

        if if_all:
            data = read_all_data(csv_dir)
        else:
            data = pd.read_csv(data_dir, header=None)

        state = data.iloc[0].to_numpy()
        npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                        for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
        action = data.iloc[1].to_numpy()
        npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
        npstate = normalize_data(npstate)
        npaction = normalize_data(npaction)
        if if_delt:
            npaction = npaction - npstate # 使用相对值
        return npstate, npaction
    else:
        data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/test/data5.csv"
        data = pd.read_csv(data_dir, header=None)
        state = data.iloc[0].to_numpy()
        npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                        for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
        action = data.iloc[1].to_numpy()
        npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
        npstate = normalize_data(npstate)
        npaction = normalize_data(npaction)
        if if_delt:
            npaction = npaction - npstate # 使用相对值
        return npstate, npaction

def run_model(model, initial_input, num_steps, frame):
    outputs = []  # 存储模型输出的列表
    current_input = initial_input  # 初始输入

    for step in range(num_steps):
        # 将当前输入转换为 PyTorch 张量
        input_tensor = torch.Tensor(current_input)

        # 使用模型进行推断
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 将输出添加到列表中
        output_array = output_tensor.numpy()
        outputs.append(output_array)

        current_input = current_input[4:]

        # 将模型的输出作为下一个步骤的输入
        if frame != 1:
            current_input = np.concatenate((current_input, output_array))
        else:
            current_input = output_array
    return outputs

# net
class BehaviorCloningModel(nn.Module):  # 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()  # 继承自父类的构造
        self.fc = nn.Sequential(nn.Linear(input_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, output_dim)
                                )  # 搭建网络，两层隐藏层

    def forward(self, x):  # 前向传播方法
        return self.fc(x)

def main1():
    # parameter
    input_dim = 4
    out_dim = 4
    learning_rate = 0.0001
    batch_size = 64
    epochs = 20000
    save_interval = 5000

    if_train = False
    if_test = False
    if_run_model = True

    # # 测试最大最小值
    # npstate, npaciton = read_data(if_all=True,if_delt=False,if_test=False)
    # column_max_values = np.max(npstate, axis=0) # axis=0表示沿着垂直方向
    # column_min_values = np.min(npstate, axis=0) 
    # print(f"max: {column_max_values}") # max: [5.80152040e-01 2.01897750e-04 2.99989649e-01 3.66356160e-01]
    # print(f"min: {column_min_values}") # min: [ 2.99900302e-01 -1.71028450e-01  5.59073600e-02 -8.75721150e-05]

    # train
    if if_train:
        time = datetime.now()
        log_dir = "B:/code/kortex/imitation_learning/v2" + "\\" + "run\\"  + str(time.month) + '_' + str(time.day) + '_'  + str(time.hour) + '_' + str(time.minute)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        model = BehaviorCloningModel(input_dim, out_dim)  # 使用BC进行训练
        criterion = nn.MSELoss()  # 损失函数：均方损失
        # criterion = nn.NLLLoss()  # 损失函数：负对数似然损失
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器
        npstate, npaciton = read_data(if_all=True, if_delt=False, if_test=False) # 全部、不绝对值、不测试
        dataset = TensorDataset(torch.Tensor(npstate), torch.Tensor(npaciton))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle=true打乱顺序抽取数据
        with tqdm(total=epochs, desc="Processing") as pbar:
            for epoch in range(epochs):
                total_loss = 0.0
                for e_state, e_action in dataloader:
                    optimizer.zero_grad() # 梯度清零
                    pre_aciton = model(e_state) # 预测action
                    loss = criterion(pre_aciton, e_action) # 计算损失
                    loss.backward() # 梯度回传
                    optimizer.step() # 优化
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                writer.add_scalar('avg_loss', avg_loss, epoch)
                if (epoch + 1) % 1000 == 0:
                    print("epoch: {},    loss: {:.6f}".format(epoch, avg_loss*10000000)) # loss = 35.9

                # save model
                if (epoch + 1) % save_interval == 0:
                    model_save_path = os.path.join(log_dir, f'model_epoch{epoch + 1}.pth')
                    torch.save(model.state_dict(), model_save_path)

                pbar.update(1) # 进度条+1

    if if_test:
        # 读取测试数据
        test_npstate, test_npaction = read_data(False, False, True)

        # 将测试数据转换为PyTorch张量
        test_state_tensor = torch.Tensor(test_npstate)
        test_action_tensor = torch.Tensor(test_npaction)

        # 使用模型进行推断
        if if_train:
            with torch.no_grad():
                predicted_actions = model(test_state_tensor)
        else:
            with torch.no_grad():
                model_path = 'B:/code/kortex/imitation_learning/v2/run/9_7_17_42/model_epoch50000.pth'
                model = BehaviorCloningModel(4, 4)
                model.load_state_dict(torch.load(model_path))
                predicted_actions = model(test_state_tensor)

        # 计算模型的性能指标（例如均方误差）
        mse = nn.MSELoss()
        test_loss = mse(predicted_actions, test_action_tensor)
        print(f"Test Mean Squared Error (MSE): {test_loss.item()*10000000}") # loss = 58.7

    if if_run_model:
        with torch.no_grad():
            model_path = 'B:/code/kortex/imitation_learning/v2/run/9_8_10_59/model_epoch20000.pth'
            model = BehaviorCloningModel(4, 4)
            model.load_state_dict(torch.load(model_path))
            outputs = run_model(model, initial_input=normalize_data(np.array([0.3, 0, 0.3, 0])), num_steps=400, frame=1)
            outputs = np.array(outputs)
            outputs = origin_data(outputs)
            print(outputs)
            plot_np(outputs)

def main2():
    # parameter
    frame = 3 # 叠加帧数
    input_dim = 4*frame
    out_dim = 4
    learning_rate = 0.0001
    batch_size = 64
    epochs = 20000
    save_interval = 5000

    if_train = False
    if_test = False
    if_run_model = True

    # # 测试最大最小值
    # npstate, npaciton = read_data(if_all=True,if_delt=False,if_test=False)
    # column_max_values = np.max(npstate, axis=0) # axis=0表示沿着垂直方向
    # column_min_values = np.min(npstate, axis=0) 
    # print(f"max: {column_max_values}") # max: [5.80152040e-01 2.01897750e-04 2.99989649e-01 3.66356160e-01]
    # print(f"min: {column_min_values}") # min: [ 2.99900302e-01 -1.71028450e-01  5.59073600e-02 -8.75721150e-05]

    # train
    if if_train:
        time = datetime.now()
        log_dir = "B:/code/kortex/imitation_learning/v2" + "\\" + "run\\"  + str(time.month) + '_' + str(time.day) + '_'  + str(time.hour) + '_' + str(time.minute)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        model = BehaviorCloningModel(input_dim, out_dim)  # 使用BC进行训练
        criterion = nn.MSELoss()  # 损失函数：均方损失
        # criterion = nn.NLLLoss()  # 损失函数：负对数似然损失
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器
        npstate, npaciton = read_data(if_all=True, if_delt=False, if_test=False) # 全部、不绝对值、不测试

        npstate, npaciton = overlay_frames(npstate, npaciton, frame)

        dataset = TensorDataset(torch.Tensor(npstate), torch.Tensor(npaciton))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # shuffle=true打乱顺序抽取数据
        with tqdm(total=epochs, desc="Processing") as pbar:
            for epoch in range(epochs):
                total_loss = 0.0
                for e_state, e_action in dataloader:
                    optimizer.zero_grad() # 梯度清零
                    pre_aciton = model(e_state) # 预测action
                    loss = criterion(pre_aciton, e_action) # 计算损失
                    loss.backward() # 梯度回传
                    optimizer.step() # 优化
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                writer.add_scalar('avg_loss', avg_loss, epoch)
                if (epoch + 1) % 1000 == 0:
                    print("epoch: {},    loss: {:.6f}".format(epoch, avg_loss*10000000)) # loss = 35.9

                # save model
                if (epoch + 1) % save_interval == 0:
                    model_save_path = os.path.join(log_dir, f'model_epoch{epoch + 1}.pth')
                    torch.save(model.state_dict(), model_save_path)

                pbar.update(1) # 进度条+1

    if if_test:
        # 读取测试数据
        test_npstate, test_npaction = read_data(False, False, True)

        test_npstate, test_npaction = overlay_frames(test_npstate, test_npaction, frame)

        # 将测试数据转换为PyTorch张量
        test_state_tensor = torch.Tensor(test_npstate)
        test_action_tensor = torch.Tensor(test_npaction)

        # 使用模型进行推断
        if if_train:
            with torch.no_grad():
                predicted_actions = model(test_state_tensor)
        else:
            with torch.no_grad():
                model_path = 'B:/code/kortex/imitation_learning/v2/run/9_8_13_15/model_epoch20000.pth'
                model = BehaviorCloningModel(input_dim, out_dim)
                model.load_state_dict(torch.load(model_path))
                predicted_actions = model(test_state_tensor)

        # 计算模型的性能指标（例如均方误差）
        mse = nn.MSELoss()
        test_loss = mse(predicted_actions, test_action_tensor)
        print(f"Test Mean Squared Error (MSE): {test_loss.item()*10000000}") # loss = 58.7

    if if_run_model:
        with torch.no_grad():
            model_path = 'B:/code/kortex/imitation_learning/v2/run/9_8_13_15/model_epoch20000.pth'
            model = BehaviorCloningModel(input_dim, out_dim)
            model.load_state_dict(torch.load(model_path))
            initial_input = normalize_data(np.array([[ 2.99922740e-01, -3.85967414e-05,  2.99946854e-01,  2.65256679e-03],[2.99932234e-01, 1.24275835e-04, 2.99952932e-01, 4.80660593e-04],[2.99923132e-01, 6.25820728e-05, 2.99937792e-01, 3.69983390e-03]]))
            initial_input = initial_input.flatten()
            print(initial_input)
            outputs = run_model(model, initial_input, num_steps=400, frame=frame)
            outputs = np.array(outputs)
            outputs = origin_data(outputs)
            print(outputs)
            plot_np(outputs)

if __name__ == '__main__':
    main2()
