#!/usr/bin/python3

import sys
sys.path.append('/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it')
import time
import rospy
import moveit_commander
import moveit_msgs.msg
from gazebo_msgs.srv import *
from geometry_msgs.msg import *
from math import pi
from control_msgs.msg import *
from trajectory_msgs.msg import *
import actionlib
from std_srvs.srv import Empty
from tf import TransformListener
# import gen3env
from robot import Robot
# from task import peg_in
# from gen3env import gen3env
import gym
# from stable_baselines3  import TD3
import torch.nn as nn
# from BC_model import BehaviorCloningModel
import torch
from Gen3Env.gen3env import gen3env
from stable_baselines3 import TD3
import os


# net
class BehaviorCloningModel(nn.Module):  # 搭建神经网络
    def __init__(self, input_dim, output_dim):
        super(BehaviorCloningModel, self).__init__()  # 继承自父类的构造
        self.fc = nn.Sequential(nn.Linear(input_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Linear(64, output_dim)
                                )  # 搭建网络，两层隐藏层

    def forward(self, x):  # 前向传播方法
        return self.fc(x)

def RL_train():
   env=gym.make(id='peg_in_hole-v0')
   env.reset()
   log_path='./log'
   if not os.path.exists(log_path):
      os.makedirs(log_path)
  #  print(torch.cuda.is_available())
   if torch.cuda.is_available():
      print('cuda is available, train on GPU!')
   model=TD3('MlpPolicy', env, verbose=1,tensorboard_log=log_path,device='cuda')
   model.learn(total_timesteps=1000)

def BC_test():
  # env=gym.make(id='Pendulum-v1')
  env=gym.make(id='peg_in_hole-v0')
  # env=gen3env()
  env.reset()
  model_path = '/catkin_workspace/src/ros_kortex/kortex_examples/src/move_it/model/model_epoch50000_mse_1r_32b_jue.pth'
  model = BehaviorCloningModel(4, 4)
  model.load_state_dict(torch.load(model_path))

  # absolute
  for episode in range(10):
    print("episode: {}".format(episode))
    env.robot.move(pose=[0.5, 0, 0.5])
    obs = env.reset()
    # print(type(obs))
    # print(obs)
    model_obs = obs[:4]
    done = False
    # while not done:
    for step in range(2):
      with torch.no_grad():
        action = model(torch.Tensor(model_obs)).tolist()
        # print(type(action))
      next_obs,reward,done,_=env.step(action=action)
      # print('reward={}'.format(reward))
      model_obs = next_obs[:4]
  env.robot.move(pose=[0.5, 0, 0.5])
  # model=TD3('MlpPolicy', env, verbose=1)
  # model.learn(total_timesteps=1000)

  # # delt
  # for episode in range(10):
  #   print("episode: {}".format(episode))
  #   env.robot.move(pose=[0.5, 0, 0.5])
  #   print("start")
  #   obs = env.reset()
  #   model_obs = obs[:4]
  #   done = False
  #   # while not done:
  #   for step in range(2):
  #     with torch.no_grad():
  #       action = model(torch.Tensor(model_obs)).tolist() + model_obs
  #     next_obs,reward,done,_=env.step(action=action)
  #     # print('reward={}'.format(reward))
  #     model_obs = next_obs[:4]
  # env.robot.move(pose=[0.5, 0, 0.5])
  
if __name__ == '__main__':
  BC_test()
