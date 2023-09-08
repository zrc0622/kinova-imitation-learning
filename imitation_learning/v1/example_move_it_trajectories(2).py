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
import gen3env
# from module import TD3,train
import gym

def main(): 
    # Initialize ROS node 

    
    # Create a Robot instance 
    robot = Robot()
    env = gen3env()
    env.reset()
    

    print('Execute peg in hole tast...')
    
    # if robot.is_init_success: 
    #     # Continuously retrieve and print Cartesian pose 
    #     rate = rospy.Rate(1) 
    #     # Rate of 1 Hz 
    #     while not rospy.is_shutdown(): 
    #       cartesian_pose = robot.get_cartesian_pose() 
    #       print("Cartesian Pose:") 
    #       print("Position: [x={}, y={}, z={}]".format(cartesian_pose.position.x, cartesian_pose.position.y, cartesian_pose.position.z)) 
    #       print("Orientation: [x={}, y={}, z={}, w={}]".format(cartesian_pose.orientation.x, cartesian_pose.orientation.y, cartesian_pose.orientation.z, cartesian_pose.orientation.w)) 
    #       print("---") 
    #       rate.sleep() 
    #     else: 
    #        print("Robot initialization failed.") 

    if robot.is_init_success: 
      # Continuously retrieve and print Cartesian pose 
      rate = rospy.Rate(0.2) 
      # Rate of 1 Hz 
      while not rospy.is_shutdown(): 
        action = env.get_action()
        obs = env.get_obs()
        print("last acton:")
        print("Action: [end_x{},end_y{},end_z{},gripper_opening{}]".format(action[0], action[1], action[2], action[3]))
        print("current pose:") 
        print("Position: [end_x{},end_y{},end_z{},gripper_opening{},peg_x{},peg_y{},peg_z{},hole_x{},hole_y{},hole_z{}]".format(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9])) 
        print("---") 
        rate.sleep() 
      else: 
          print("Robot initialization failed.") 

if __name__ == '__main__': 
   main()
