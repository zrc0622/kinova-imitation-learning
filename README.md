# Kinova Gen3 Arm Peg Insert Task With Imitation Learning
train Kinova Gen3 Arm in Gazebo with imitation learning

## BC训练代码
`imitation_learning/v2/BCTrain.py`

## BC训练脚本
### LSTM网络
```
CUDA_VISIBLE_DEVICES=2 python BCTrain.py --net lstm --train --frame 1
```
1. 训练代码：`450-518行`，网络结构：`223-233行`
2. 训练方式：一个epoch输入一个完整的`专家状态序列`，并以`生成动作序列`和`专家动作序列`的MSE作为loss对网络进行优化
### MLP网络
```
CUDA_VISIBLE_DEVICES=2 python BCTrain.py --net mlp --train --all_data --frame 4
```