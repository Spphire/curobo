import json 
import os 
import sys 
import numpy as np
import time
import math
import transforms3d as t3d
from threading import Thread
import socket
import requests
rfplanner_root = "/home/yibo/workspace/ShadowHandRfplanner/rfplanner" # os.path.dirname(os.path.abspath(__file__))+"/rfplanner"

sys.path.insert(0,rfplanner_root+"/examples/fcl_robot_planner/script/")

from scipy.spatial.transform import Rotation as R
from BulletEnv import BulletEnv
from SingleHandPlanner import RfHandPlanner
from rf_robot_planner import HandPlannerIKType
from ur10e_controller import Shadowhand_Model

# def unity2zup_right_frame(pos_quat: np.ndarray):
#     assert(pos_quat.shape[0]==7)
#     pos_quat*=np.array([1,-1,1,1,-1,1,-1])
#     rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
#     pos_vec = pos_quat[:3]
#     T=np.eye(4)
#     T[:3,:3]= rot_mat
#     T[:3,3]=pos_vec
#     fit_mat = t3d.euler.axangle2mat([0,1,0],np.pi/2)
#     fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
#     target_rot_mat=fit_mat@rot_mat
#     target_pos_vec=fit_mat@pos_vec
#     target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
#     return target

def unity2zup_right_frame(pos_quat: np.ndarray):
        pos_quat = np.array(pos_quat)
        assert(pos_quat.shape[0]==7)
        pos_quat*=np.array([1,-1,1,1,-1,1,-1])
        rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
        pos_vec = pos_quat[:3]
        T=np.eye(4)
        T[:3,:3]= rot_mat
        T[:3,3]=pos_vec
        fit_mat = t3d.euler.axangle2mat([0,1,0],-np.pi/2)
        #fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
        fit_mat = fit_mat@t3d.euler.axangle2mat([1,0,0],np.pi)
        target_rot_mat=fit_mat@rot_mat
        target_pos_vec=fit_mat@pos_vec
        target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
        return target

def tcp2mat(tcp):
    rot_mat = t3d.quaternions.quat2mat(tcp[3:])
    pos_vec = tcp[:3]
    T=np.eye(4)
    T[:3,:3]= rot_mat
    T[:3,3]=pos_vec
    return T

##
def get_relative_tcp_rfplanner(tcp, root_tcp):
    tcp = np.array(tcp)
    root_tcp = np.array(root_tcp)
    rot_mat = np.linalg.inv(t3d.quaternions.quat2mat(root_tcp[3:]))
    pos = rot_mat @ (tcp[:3]-root_tcp[:3])
    quat = t3d.quaternions.mat2quat(rot_mat @ t3d.quaternions.quat2mat(tcp[3:]))
    #tcp_mat = tcp2mat(tcp)
    #root_mat = tcp2mat(root_tcp)
    #final_mat = np.linalg.inv(root_mat) @ tcp_mat
    #res = np.array(final_mat[:3,3].flatten().tolist() + t3d.quaternions.mat2quat(final_mat[:3,:3]).tolist())
    combine = np.array(pos.flatten().tolist()+quat.tolist())
    return unity2zup_right_frame(combine)

def search_nearest_tip(s:dict):
    tip_name = ["indexTip","middleTip","ringTip","pinkyTip"]
    dis = 10000
    name = "thumbTip"
    for n in tip_name:
        dd = np.linalg.norm(np.array(s["thumbTip"])[:3] - np.array(s[n])[:3]) 
        if dd < dis:
            dis = dd
            name = n
    return name

def fintune_thumb_tip(s:dict):
    nearest_tip = search_nearest_tip(s)
    _thumb = get_relative_tcp_rfplanner(np.array(s["thumbTip"]),np.array(s["pos"]+s["quat"]))
    _nearest_tip = t = get_relative_tcp_rfplanner(np.array(s[nearest_tip]),np.array(s["pos"]+s["quat"]))
    nearest_tip_fk = _nearest_tip ##
    
    res = (_thumb[:3]-_nearest_tip[:3]) + nearest_tip_fk[:3]

    return np.array(res.tolist()+_thumb[3:].tolist())

def getRobotGraspJointVaules(q_goal,
                            q_thumb_values,
                            q_first_values,
                            q_middle_values,
                            q_ring_vaules,
                            q_little_values):
    for item in q_thumb_values:
        q_goal.append(item)
    for item in q_first_values:
        q_goal.append(item)
    for item in q_middle_values:
        q_goal.append(item)
    for item in q_ring_vaules:
        q_goal.append(item)
    for item in q_little_values:
        q_goal.append(item)
    return q_goal

class Receiver(Thread):
    def __init__(self, local_ip= "10.9.11.1",port=8082):
        self.address = (local_ip, port)
        self.socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_obj.setblocking(0)
        self.isOn=True

        ign_vec = []
        self.hand_model = Shadowhand_Model()
        self.hand_planner = RfHandPlanner("SingleUR_10_Shadow.yaml",ign_vec=ign_vec)
        self.bullet_env = BulletEnv(self.hand_planner.rfYamlConfig.getHomePose(),fcl_planner=self.hand_planner)

    def HandIK(self,tcp_pos_quat = None,
               current_activate = False,
               finger_type = HandPlannerIKType.RobotIK,
               is_hard = False):
        # x,y,z,rx,ry,rz,rw
        tcp_goal = np.array([tcp_pos_quat[0],tcp_pos_quat[1],tcp_pos_quat[2],
                             tcp_pos_quat[3],tcp_pos_quat[4],tcp_pos_quat[5],tcp_pos_quat[6]])
        
        if(current_activate == True):
            joint_current = self.bullet_env.bullet_robot.get_activate_joint_for_bullet()
            q_res = self.hand_planner.hand_planner.solveIK(finger_type,tcp_goal,joint_current,hard_solve=is_hard)
            return q_res
        else:
            q_res = self.hand_planner.hand_planner.solveIK(finger_type,tcp_goal,hard_solve=is_hard)
        return q_res

    def run(self):
        self.socket_obj.bind(self.address)
        print("start receiving...")
        while self.isOn:
            self.receive()
            time.sleep(0.01)

    def receive(self):
        s=dict()
        try:
            data, _ = self.socket_obj.recvfrom(2048)
            s=json.loads(data)
        except:
             return

        s["q"] = np.array(s["q"])/180*math.pi

        t = fintune_thumb_tip(s)

        q_thumb_g = self.HandIK(tcp_pos_quat = t,
                            current_activate = False,
                            finger_type = HandPlannerIKType.ThumbIK,
                            is_hard=True)
        q_goal = getRobotGraspJointVaules([0.0004430418193805963, -1.6971481482135218, 2.4696645736694336, -0.6649912039386194, 1.6591527462005615, 0.014633487910032272,],
                                            q_thumb_values=q_thumb_g,
                                            q_first_values=s["q"][20:24][::-1],
                                            q_middle_values=s["q"][16:20][::-1],
                                            q_ring_vaules=s["q"][12:16][::-1],
                                            q_little_values=s["q"][7:12][::-1])
        s["q"][2:7] = q_thumb_g[::-1]
        print(self.hand_model.get_kinematics_state(s["q"].tolist()).link_pose)
        s["q"] = s["q"]*180/math.pi
        #requests.post("http://10.9.11.1:8000/move_hand",json.dumps({'q': s["q"].tolist()}), timeout=0.05)
        self.bullet_env.set_state(q_goal)
        
        
if __name__ == "__main__":
    r = Receiver()
    r.run()