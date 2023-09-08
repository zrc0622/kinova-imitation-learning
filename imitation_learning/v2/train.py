import gym
from TD3_module import TD3,train

env_name='BipedalWalker-v3'
env=gym.make(id=env_name,render_mode='rgb_array')
agent=TD3(env)
train(env,agent)
