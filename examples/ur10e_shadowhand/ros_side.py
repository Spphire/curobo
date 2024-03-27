import fastapi
import uvicorn
from typing import List
from pydantic import BaseModel
import numpy as np
import math

import rospy
import tf
from sr_robot_commander.sr_hand_commander import SrHandCommander
from sr_robot_commander.sr_arm_commander import SrArmCommander
rospy.init_node("right_hand_arm_joint_pos__3", anonymous=True)
hand_commander = SrHandCommander(name="right_hand")
hand_commander.set_max_velocity_scaling_factor(0.1)
arm_commander = SrArmCommander(name="right_arm", set_ground=True)
arm_commander.set_max_velocity_scaling_factor(0.01)


class HandJoints(BaseModel):
    q: List[float]
    isTracking: int
    wrist_pos: List[float]

class ArmJoints(BaseModel):
    q: List[float]

class TipName(BaseModel):
    name: str

joint_names = [
    "ra_shoulder_pan_joint",
    "ra_shoulder_lift_joint",
    "ra_elbow_joint",
    "ra_wrist_1_joint",
    "ra_wrist_2_joint",
    "ra_wrist_3_joint"
]

joint_names_hand = [
                'rh_WRJ1', 'rh_WRJ2',
                'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',

                'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
                'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
                'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
                'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
            ]

tip_names = {
    'thumbTip' : '/rh_thtip',
    'indexTip' : '/rh_fftip',
    'middleTip' : '/rh_mftip',
    'ringTip' : '/rh_rftip',
    'pinkyTip' : '/rh_lftip',
}

app=fastapi.FastAPI()

@app.post("/getJoints")
def get_joints():
    q=arm_commander.get_current_state()
    return q

@app.post("/handJoints")
def test(joint: HandJoints):
    print(joint.q)
    return {'status':'ok'}


@app.post("/move")
def move(joint: ArmJoints):
    print(joint.q)
    arm_home_joints_goal_target = {i:j for (i, j) in zip(joint_names, joint.q)}
    arm_commander.move_to_joint_value_target_unsafe(
        arm_home_joints_goal_target,
        0.01,  # time duration
        False  # wait for return
    )
    return {'status':'ok'}

@app.post("/move_hand")
def move_hand(joint: ArmJoints):
    q=np.array(joint.q)/180*math.pi
    hand_joints_goal = {i:j for (i, j) in zip(joint_names_hand, q)}
    hand_commander.move_to_joint_value_target_unsafe(
        hand_joints_goal,
        0.01,  # time duration
        False  # wait for return
    )
    return {'status':'ok'}

@app.post("/get_tip_tcp")
def get_tip_tcp(name: TipName):
    print("111")
    listener = tf.TransformListener()
    (trans,rot) = listener.lookupTransform('/rh_palm', '/rh_fftip', rospy.Time(0))
    return {"pos" : trans+rot, "valid" : True}
    print("111")
    while not rospy.is_shutdown():
        try:
            pass
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #print("hahaha")
            continue

    # try:
    #     (trans,rot) = listener.lookupTransform('/rh_palm', tip_names[name.name], rospy.Time(0))
    #     return {"pos" : trans+rot, "valid" : True}
    # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #     return {"valid" : False}


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0", port=8000) #, log_level="critical")