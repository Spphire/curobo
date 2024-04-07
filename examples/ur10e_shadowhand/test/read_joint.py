import fastapi
import uvicorn
from typing import List
from pydantic import BaseModel
import numpy as np
import math
import time
import rospy
import tf
from sr_robot_commander.sr_hand_commander import SrHandCommander
from sr_robot_commander.sr_arm_commander import SrArmCommander
rospy.init_node("right_hand_arm_joint_pos__3", anonymous=True)
hand_commander = SrHandCommander(name="right_hand")
hand_commander.set_max_velocity_scaling_factor(0.1)

print(time.time())
for i in range(1000):
    q=hand_commander.get_current_state()
    print(q)
    time.sleep(0.001)
print(time.time())