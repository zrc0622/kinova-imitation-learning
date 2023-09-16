# 三层隐层 √
# 归一化 √
# 帧叠加 √
# 修复帧叠加数据问题（序列头尾相接） √ 
# lstm多序列问题（序列头尾相接）
# lstm死循环问题（在数据密集处易死循环，尝试增大学习率跳出）——貌似增大epoch可以解决
# delt 测试
# lstm 单数据过拟合(修改保存model方式，比较loss来决定是否保存)
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
from plot import plot_np, plot_all, plot_4dnp

import glob

import argparse

#距离过小或夹爪变化小于 gripper_change_deta 删除本组数据
def data_processing(state, delt, gripper_change_delt):
    p = state[0,:]
    delete_line = []
    for i in range(1,state.shape[0]):
        distance = np.sum(np.square(p[:3]-state[i][:3]))  
        gripper_change = abs(p[3]-state[i][3])
        
        if(gripper_change <= gripper_change_delt and distance < delt):
            delete_line.append(i)
        else:
            p = state[i]
    processed_data = np.delete(state,delete_line,0)
    return processed_data

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

def save_parameters_to_txt(log_dir, **kwargs):
    # os.makedirs(log_dir)
    filename = os.path.join(log_dir, "log.txt")
    with open(filename, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key}={value}\n")

# 叠加帧
def overlay_frames(state, action, frame_length):
    inputs = []
    outputs = []
    for i in range(len(state) - frame_length):
        input_sequence = state[i:(i + frame_length)].flatten()
        output_sequence = state[i + frame_length]  # 使用npaction的第四行作为输出
        inputs.append(input_sequence)
        outputs.append(output_sequence)
    return np.array(inputs), np.array(outputs)

# 读取数据
def read_data(if_all, if_delt, if_test, frame):
    delt = 5e-6
    gripper_change_delt = 0.03/(0.36635616+0.000087572115)

    if frame == 1 or if_test:
        # train and frame 1
        if not if_test: 
            # dir = str(Path.cwd())
            # data_dir = "./data/rule/data1.csv"
            # csv_dir = "./data/rule/"
            
            data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/data1.csv"
            csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/"

            # data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/optimize/data1.csv"
            # csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/optimize/"

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
            npstate = data_processing(npstate, delt, gripper_change_delt)
            npaction = data_processing(npaction, delt, gripper_change_delt)
            if if_delt:
                npaction = npaction - npstate # 使用相对值
            return npstate, npaction
        # test
        else:
            data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/test/data5.csv"
            # data_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/optimize/data5.csv"
            data = pd.read_csv(data_dir, header=None)
            state = data.iloc[0].to_numpy()
            npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                            for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
            action = data.iloc[1].to_numpy()
            npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
            npstate = normalize_data(npstate)
            npaction = normalize_data(npaction)
            npstate = data_processing(npstate, delt, gripper_change_delt)
            npaction = data_processing(npaction, delt, gripper_change_delt)
            if if_delt:
                npaction = npaction - npstate # 使用相对值
            return npstate, npaction
    else:
        # train and frame not 1
        npstates = []
        npactions = []
        csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/"
        # csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/optimize/"
        csv_files = glob.glob(csv_dir + "*.csv")
        for file in csv_files:
            data = pd.read_csv(file, header=None)
            state = data.iloc[0].to_numpy()
            npstate = np.array([np.fromstring(item[1:-1], sep=' ')
                            for item in state])  # 将state变为(?, 4)的格式，一行代表一个state
            action = data.iloc[1].to_numpy()
            npaction = np.array([np.fromstring(item[1:-1], sep=' ') for item in action])
            npstate = normalize_data(npstate)
            npaction = normalize_data(npaction)
            npstate = data_processing(npstate, delt, gripper_change_delt)
            npaction = data_processing(npaction, delt, gripper_change_delt)
            if if_delt:
                npaction = npaction - npstate # 使用相对值
            lay_state, lay_action = overlay_frames(npstate, npaction, frame)
            npstates.append(lay_state)
            npactions.append(lay_action)
        npstates = np.vstack(npstates)
        npactions = np.vstack(npactions)
        # print(npstates.shape)
        # print(npactions.shape)
        return npstates, npactions

# 测试模型
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

def run_model_for_lstm(model, initial_input, num_steps, frame):
    outputs = []  # 存储模型输出的列表
    current_input = initial_input  # 初始输入

    for step in range(num_steps):
        # 将当前输入转换为 PyTorch 张量，添加批次和序列长度维度
        input_tensor = torch.Tensor(current_input).unsqueeze(0).unsqueeze(0)

        # 使用模型进行推断
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # 将输出添加到列表中
        output_array = output_tensor.numpy()
        outputs.append(output_array)

        current_input = current_input[4:]

        # 将模型的输出作为下一个步骤的输入
        if frame != 1:
            current_input = np.concatenate((current_input, output_array[0, 0]))
        else:
            current_input = output_array[0, 0]
    return outputs

# MLP网络
class MLPModel(nn.Module):  # 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(MLPModel, self).__init__()  # 继承自父类的构造
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

# LSTM网络
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden

# MLP
def MLP_train(if_train, if_test, if_run_model, if_all_data):
    # parameter
    input_dim = 4
    out_dim = 4
    learning_rate = 0.0001
    batch_size = 64
    epochs = 100000
    save_interval = 10000

    # if_train = False
    # if_test = False
    # if_run_model = True

    # # 测试最大最小值
    # npstate, npaciton = read_data(if_all=True,if_delt=False,if_test=False)
    # column_max_values = np.max(npstate, axis=0) # axis=0表示沿着垂直方向
    # column_min_values = np.min(npstate, axis=0) 
    # print(f"max: {column_max_values}") # max: [5.80152040e-01 2.01897750e-04 2.99989649e-01 3.66356160e-01]
    # print(f"min: {column_min_values}") # min: [ 2.99900302e-01 -1.71028450e-01  5.59073600e-02 -8.75721150e-05]

    # train
    if if_train:
        time = datetime.now()
        log_dir = "B:/code/kortex/imitation_learning/v2" + "\\" + "MLP_run\\"  + str(time.month) + '_' + str(time.day) + '_'  + str(time.hour) + '_' + str(time.minute)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        model = MLPModel(input_dim, out_dim)  # 使用BC进行训练
        criterion = nn.MSELoss()  # 损失函数：均方损失
        # criterion = nn.NLLLoss()  # 损失函数：负对数似然损失
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器
        npstate, npaciton = read_data(if_all=if_all_data, if_delt=False, if_test=False, frame=1) # 全部、不绝对值、不测试
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
        test_npstate, test_npaction = read_data(False, False, True, frame=1)

        # 将测试数据转换为PyTorch张量
        test_state_tensor = torch.Tensor(test_npstate)
        test_action_tensor = torch.Tensor(test_npaction)

        # 使用模型进行推断
        if if_train:
            with torch.no_grad():
                predicted_actions = model(test_state_tensor)
        else:
            with torch.no_grad():
                model_path = 'B:/code/kortex/imitation_learning/v2/MLP_run/9_7_17_42/model_epoch50000.pth'
                model = MLPModel(4, 4)
                model.load_state_dict(torch.load(model_path))
                predicted_actions = model(test_state_tensor)

        # 计算模型的性能指标（例如均方误差）
        mse = nn.MSELoss()
        test_loss = mse(predicted_actions, test_action_tensor)
        print(f"Test Mean Squared Error (MSE): {test_loss.item()*10000000}") # loss = 58.7

    if if_run_model:
        with torch.no_grad():
            model_path = 'B:/code/kortex/imitation_learning/v2/MLP_run/9_8_10_59/model_epoch20000.pth'
            model = MLPModel(4, 4)
            model.load_state_dict(torch.load(model_path))
            outputs = run_model(model, initial_input=normalize_data(np.array([0.3, 0, 0.3, 0])), num_steps=400, frame=1)
            outputs = np.array(outputs)
            outputs = origin_data(outputs)
            print(outputs)
            plot_np(outputs)

# 帧叠加MLP
def MLP_train_with_frame(if_train, if_test, if_run_model, frame, if_all_data):
    # parameter
    # frame = 5 # 叠加帧数
    input_dim = 4*frame
    out_dim = 4
    learning_rate = 0.001
    batch_size = 64
    epochs = 300000
    save_interval = 5000
    print_loss = 10000

    # if_train = True
    # if_test = False
    # if_run_model = False

    # # 测试最大最小值
    # npstate, npaciton = read_data(if_all=True,if_delt=False,if_test=False)
    # column_max_values = np.max(npstate, axis=0) # axis=0表示沿着垂直方向
    # column_min_values = np.min(npstate, axis=0) 
    # print(f"max: {column_max_values}") # max: [5.80152040e-01 2.01897750e-04 2.99989649e-01 3.66356160e-01]
    # print(f"min: {column_min_values}") # min: [ 2.99900302e-01 -1.71028450e-01  5.59073600e-02 -8.75721150e-05]

    # train
    if if_train:
        
        time = datetime.now()
        log_dir = "B:/code/kortex/imitation_learning/v2" + "\\" + "MLP_run\\" + "new_run\\" + str(time.month) + '_' + str(time.day) + '_'  + str(time.hour) + '_' + str(time.minute)
        os.makedirs(log_dir, exist_ok=True)
        save_parameters_to_txt(log_dir = log_dir, frame = frame, learning_rate = learning_rate, batch_size = batch_size, if_all = if_all_data)
        writer = SummaryWriter(log_dir)
        model = MLPModel(input_dim, out_dim)  # 使用BC进行训练
        criterion = nn.MSELoss()  # 损失函数：均方损失
        # criterion = nn.NLLLoss()  # 损失函数：负对数似然损失
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器
        npstate, npaciton = read_data(if_all=if_all_data, if_delt=False, if_test=False, frame=frame) # 全部、不绝对值、不测试

        # npstate, npaciton = overlay_frames(npstate, npaciton, frame)

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
                # if (epoch + 1) % print_loss == 0:
                #     print("epoch: {},    loss: {:.6f}".format(epoch, avg_loss*10000000)) # loss = 35.9

                # save model
                if (epoch + 1) % save_interval == 0:
                    model_save_path = os.path.join(log_dir, f'model_epoch{epoch + 1}.pth')
                    torch.save(model.state_dict(), model_save_path)

                pbar.update(1) # 进度条+1

    if if_test:
        # 读取测试数据
        test_npstate, test_npaction = read_data(False, False, True, frame=frame)

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
                model_path = 'B:/code/kortex/imitation_learning/v2/MLP_run/9_8_13_15/model_epoch50000.pth'
                model = MLPModel(input_dim, out_dim)
                model.load_state_dict(torch.load(model_path))
                predicted_actions = model(test_state_tensor)

        # 计算模型的性能指标（例如均方误差）
        mse = nn.MSELoss()
        test_loss = mse(predicted_actions, test_action_tensor)
        print(f"Test Mean Squared Error (MSE): {test_loss.item()*10000000}") # loss = 58.7

    if if_run_model:
        with torch.no_grad():
            model_path = 'B:\\code\\kortex\\imitation_learning\\v2\\MLP_run\\new_run\\9_13_15_44\\model_epoch50000.pth'
            model_path = "C:\\Users\\LEGION\\Desktop\\model_epoch75000.pth"
            model = MLPModel(input_dim, out_dim)
            model.load_state_dict(torch.load(model_path))
            initial_input = normalize_data(np.array([ 2.99922740e-01, -3.85967414e-05,  2.99946854e-01,  2.65256679e-03]))#[0.32018349 ,-0.00349947 , 0.12678419 , 0.36635616]
            initial_input = np.tile(initial_input, (frame, 1))
            initial_input = initial_input.flatten()
            print(initial_input)
            outputs = run_model(model, initial_input, num_steps=50, frame=frame)
            outputs = np.array(outputs)
            outputs = origin_data(outputs)
            print(outputs)
            plot_4dnp(outputs)

# LSTM网络
def LSTM_train(if_train, if_test, if_run_model, if_all_data):
    # 初始化模型
    input_size = 4  # 输入特征数
    hidden_size = 128  # 隐藏层大小
    output_size = 4  # 输出特征数（与输入的特征数相同）
    batch_size = 32
    save_model = 5000
    lr = 0.001
    num_epochs = 50000
    model = LSTMModel(input_size, hidden_size, output_size)
    time = datetime.now()
    log_dir = "B:/code/kortex/imitation_learning/v2" + "\\" + "LSTM_run\\" + "new_run\\" + str(time.month) + '_' + str(time.day) + '_'  + str(time.hour) + '_' + str(time.minute)
    

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 读取数据
    states, actions = read_data(if_all_data, False, False, 1)

    actions = states[1:]
    states = states[:-1]

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)

    if if_train:
        os.makedirs(log_dir, exist_ok=True) 
        save_parameters_to_txt(log_dir = log_dir, learning_rate = lr, batch_size = batch_size, if_all = if_all_data, hidden_size=hidden_size)
        writer = SummaryWriter(log_dir)

        with tqdm(total=num_epochs, desc="Processing") as pbar:
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                hidden = None
                # 将数据分成批次
                for i in range(0, states.size(0), batch_size):
                    batch_states = states[i:i + batch_size]
                    batch_actions = actions[i:i + batch_size]
                    outputs, _ = model(batch_states, hidden)
                    loss = criterion(outputs.squeeze(1), batch_actions)
                    loss.backward()
                    optimizer.step()               
                writer.add_scalar('loss', loss.item(), epoch)
                pbar.update(1) # 进度条+1
                if (epoch + 1) % save_model == 0:
                    model_sava_path = os.path.join(log_dir, f'model_epoch{epoch + 1}.pth')
                    torch.save(model.state_dict(), model_sava_path)

    
    # 使用模型生成轨迹
    if if_run_model:
        model_path = 'B:\\code\\kortex\\imitation_learning\\v2\\LSTM_run\\new_run\\9_15_19_6\\model_epoch50000.pth'
        # model_path = "C:\\Users\\LEGION\\Desktop\\model_epoch145000.pth"
        model = LSTMModel(input_size, hidden_size, output_size)
        model.load_state_dict(torch.load(model_path))

        initial_input = normalize_data(np.array([ 2.99922740e-01, -3.85967414e-05,  2.99946854e-01,  2.65256679e-03]))
        # initial_input = normalize_data(np.array([0.32018349 ,-0.00349947 , 0.12678419 , 0.36635616]))
        initial_input = torch.tensor(initial_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (batch, time, feature)
        
        # 预测轨迹
        trajectory = [initial_input]

        with torch.no_grad():
            hidden = None
            for _ in range(200):  # 生成300个时刻的轨迹
                output, hidden = model(trajectory[-1], hidden)
                trajectory.append(output)

        # 将生成的轨迹转换为 numpy 数组
        trajectory = origin_data(torch.cat(trajectory, dim=1).squeeze().numpy())
        
        plot_4dnp(trajectory)
        

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str)
    parser.add_argument('--train', action='store_true')  # 使用action参数，出现了则为true
    parser.add_argument('--test', action='store_true')   # 使用action参数
    parser.add_argument('--run_model', action='store_true')  # 使用action参数
    parser.add_argument('--all_data', action='store_true') 
    parser.add_argument('--frame', type=int)

    args = parser.parse_args()

    net = args.net
    if_train = args.train
    if_test = args.test
    if_run_model = args.run_model
    frame = args.frame
    if_all_data = args.all_data

    if net == 'mlp':
        if frame == 1:
            MLP_train(if_train, if_test, if_run_model, if_all_data)
        else:
            MLP_train_with_frame(if_train, if_test, if_run_model, frame, if_all_data)
    else:
        LSTM_train(if_train, if_test, if_run_model, if_all_data)

if __name__ == '__main__':
    # main1 为普通的全连接网络
    # main2 为使用了帧叠加的全连接网络
    # main3 为LSTM网络
    main()

    # csv_dir = "B:/code/kortex/imitation_learning/v2/data/simulated_rule/"
    # plot_all(csv_dir)

# python B:\code\kortex\imitation_learning\v2\BCTrain.py --net lstm --train --all_data --frame 1
# python B:\code\kortex\imitation_learning\v2\BCTrain.py --net lstm --train --frame 1
# python B:\code\kortex\imitation_learning\v2\BCTrain.py --net lstm --run_model --frame 1

# python B:\code\kortex\imitation_learning\v2\BCTrain.py --net mlp --train --all_data --frame 5
# python B:\code\kortex\imitation_learning\v2\BCTrain.py --net mlp --run_model --frame 5