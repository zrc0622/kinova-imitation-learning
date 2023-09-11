import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BCTrain import read_data, normalize_data, origin_data
from plot import plot_np

train = True
# 假设你已经有了包含轨迹的numpy数组 trajectories
# trajectories 是一个列表，每个元素都是形状为 (?, 3) 的numpy数组

# 合并所有轨迹成一个大的数据集
states, actions = read_data(True, False, False, 1)

# 转换成 PyTorch 张量
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)
        return out, hidden


# 设置模型参数
input_size = 4  # 输入维度
hidden_size = 64  # 隐藏状态的维度
output_size = 4  # 输出维度

# 创建模型
model = RNNModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    hidden = None
    outputs, _ = model(states.unsqueeze(0), hidden)
    loss = criterion(outputs.squeeze(0), actions)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'rnn_model_five.pth')



# 初始输入
initial_input = normalize_data(np.array([ 2.99922740e-01, -3.85967414e-05,  2.99946854e-01,  2.65256679e-03]))
initial_input = torch.tensor(initial_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 预测轨迹
trajectory = [initial_input]

with torch.no_grad():
    hidden = None
    for _ in range(320):  # 生成300个时刻的轨迹
        output, hidden = model(trajectory[-1], hidden)
        trajectory.append(output)

# 将生成的轨迹转换为 numpy 数组
trajectory = origin_data(torch.cat(trajectory, dim=1).squeeze().numpy())

plot_np(trajectory)