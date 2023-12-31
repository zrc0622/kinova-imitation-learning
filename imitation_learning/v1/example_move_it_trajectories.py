# 示例文件，直接执行函数对应的动作
#!/usr/bin/env python2

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_moveit_trajectories.py __ns:=my_gen3
import sys
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
from robot import Robot
from task import peg_in
from gen3env import gen3env


def main():
  # arm = Robot()

  # success = arm.is_init_success
  # try:
  #     rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
  # except:
  #     pass
  # arm.remove_scene()
  # arm.init_scene(peg_pose=[0.1,0,0.3],hole_pose=[1.,-0.2,0.3])
  # arm.remove_scene()
  # arm.remove_peg()
  # arm.init_peg(obj_pose=[1,-0.066070,0.433983])
  # print(success)
  # example.move(pose=[0.3,0.3,0.7],ori_pose=[0.2,0.,0.,0.],is_add=False)
  
#pick&place task
  # if success:
  #    success=pick_place(robot=arm,pick_pose=[0.42,0.1,-0.185],place_pose=[-0.2,-0.2,0],joint_rota=pi/2,success=success)
 
  # arm.move(pose=[0.3,0.2,0.6])
  # gen3_pose=arm.get_obj_pose('peg')
  # arm.reach_gripper_position(0)
  # opening=arm.get_gripper_position()
  # print(opening)
  # pose_x,pose_y,pose_z=arm.get_link_pose('left_inner_finger')
  # print([pose_x,pose_y,pose_z])
#screw task
  # if success:
  #    success&=screw(robot=arm,nut_pose=[0.42,0.,-0.185],target_pose=[0.1,0.15,0.23])
  # arm.move(pose=[0.2,0.3,0.1])
  # arm.reach_named_position('home')


#peg in hole task
  # if success:
  #    success=peg_in(robot=arm,peg_pose=[0.3,0,1],hole_pose=[0.5,-0.2,1])
  # gripper_pose=arm.get_gripper_position()
  # print(gripper_pose)
  # For testing purposes
  # env=gen3env()
  # action=env.action_space.sample()
  # arm.move(pose=[])
  # rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

  # if not success:
  #     rospy.logerr("The example encountered an error.")
  env=gen3env()
  # env.reset()
  # action=env.action_space.sample()
  # next_obs,reward,done,info=env.step(action)
  # print('obs=',next_obs)
  # print('reward=',reward)
  # print('done=',done)
  # env.robot.reach_named_position('home')
  # # env.robot.reach_gripper_position(0.)
  # # env.robot.move(pose=[0.61,0,0.05])
  # # env.robot.reach_gripper_position(0.5)
  # # env.robot.move(pose=[0.8,0.035,0.15])
  # action=env.action_space.sample()
  # print(env.scale(action))
  # env.do_sim(action=action)
  # env.robot.reach_gripper_position(0.9)
  peg_in(robot=env.robot,peg_pose=[0.5,0.2,0.3],hole_pose=[0.9,0.2,0.3])
  
if __name__ == '__main__':
  main()
