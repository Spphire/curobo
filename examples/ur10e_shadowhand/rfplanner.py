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


class Rfplanner():
    def __init__(self) -> None:
        ign_vec = []
        self.hand_planner = RfHandPlanner("SingleUR_10_Shadow.yaml",ign_vec=ign_vec)
        self.bullet_env = BulletEnv(self.hand_planner.rfYamlConfig.getHomePose(),fcl_planner=self.hand_planner)
        self.tip_name = ["indexTip","middleTip","ringTip","pinkyTip"]

        self.tip_names_map = {
                    'thumbTip' : 'thdistal',
                    'indexTip' : 'ffdistal',
                    'middleTip' : 'mfdistal',
                    'ringTip' : 'rfdistal',
                    'pinkyTip' : 'lfdistal',
                }

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
    
    def unity2zup_right_frame(self, pos_quat: np.ndarray):
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

    def get_relative_tcp_rfplanner(self, tcp, root_tcp):
        tcp = np.array(tcp)
        root_tcp = np.array(root_tcp)
        rot_mat = np.linalg.inv(t3d.quaternions.quat2mat(root_tcp[3:]))
        pos = rot_mat @ (tcp[:3]-root_tcp[:3])
        quat = t3d.quaternions.mat2quat(rot_mat @ t3d.quaternions.quat2mat(tcp[3:]))
        combine = np.array(pos.flatten().tolist()+quat.tolist())
        return self.unity2zup_right_frame(combine)

    def search_nearest_tip(self, s:dict):
        
        dis = 10000
        name = "thumbTip"
        for n in self.tip_name:
            dd = np.linalg.norm(np.array(s["thumbTip"])[:3] - np.array(s[n])[:3]) 
            if dd < dis:
                dis = dd
                name = n
        return name

    def fintune_thumb_tip(self, s:dict, finger_tcp: dict = None):
        nearest_tip = self.search_nearest_tip(s)
        _thumb = self.get_relative_tcp_rfplanner(np.array(s["thumbTip"]),np.array(s["pos"]+s["quat"]))
        _nearest_tip = self.get_relative_tcp_rfplanner(np.array(s[nearest_tip]),np.array(s["pos"]+s["quat"]))

        if finger_tcp is None:
             nearest_tip_fk = _nearest_tip
        else:
            print(finger_tcp.keys(), finger_tcp.values(), self.tip_names_map[nearest_tip])
            pose = finger_tcp[self.tip_names_map[nearest_tip]]
            nearest_tip_fk = pose.position.numpy().cpu().tolist()+[0,1,0,0] # pose.quaternion.numpy().cpu().tolist()
        
        res = (_thumb[:3]-_nearest_tip[:3]) + nearest_tip_fk[:3]

        return np.array(res.tolist()+_thumb[3:].tolist())
    
    def check_format(self, s:dict):
        for n in self.tip_name:
            if len(s[n])!=7:
                return False
        if len(s["thumbTip"])!=7:
                return False
        if len(s["pos"])!=3:
                return False
        if len(s["quat"])!=4:
                return False
            
        return True

    def get_thumb_q(self, s:dict, finger_tcp: dict = None):
        if not self.check_format(s):
             print("hand message format wrong!")
             return [0.0]*5
        if finger_tcp is None:
             t = self.fintune_thumb_tip(s).tolist()
        else:
            t = self.fintune_thumb_tip(s, finger_tcp).tolist()
        q_thumb_g = self.HandIK(tcp_pos_quat = t,
                            current_activate = False,
                            finger_type = HandPlannerIKType.ThumbIK,
                            is_hard=True)
        return np.array(q_thumb_g)[::-1]*180/math.pi