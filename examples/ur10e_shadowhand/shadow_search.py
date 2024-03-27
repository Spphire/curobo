import json 
import os 
import sys 
import numpy as np
import time
import math
rfplanner_root = "/home/yibo/workspace/ShadowHandRfplanner/rfplanner" # os.path.dirname(os.path.abspath(__file__))+"/rfplanner"

sys.path.insert(0,rfplanner_root+"/examples/fcl_robot_planner/script/")

from scipy.spatial.transform import Rotation as R
from BulletEnv import BulletEnv
from SingleHandPlanner import RfHandPlanner
from rf_robot_planner import HandPlannerIKType



    
class ShadowParser:
    def __init__(self,json_path="ConvertData.json") -> None:
        self.json_path = json_path
        self.data = None
        pass
    
    def parser(self):
        if self.data != None:
            self.data.clear()
        with open(self.json_path,'r') as file:
            self.data = json.load(file)
        # for item_map in self.data:
        #     for key,value in item_map.items():
        #         if key != "time":
        #             tmp = value[0]
        #             value[0] = value[2]
        #             value[2] = tmp


class HandIK_Env:
    def __init__(self, parser:ShadowParser) -> None:
        self.parser = parser
        ign_vec = []
        self.hand_planner = RfHandPlanner("SingleUR_10_Shadow.yaml",ign_vec=ign_vec)
        self.bullet_env = BulletEnv(self.hand_planner.rfYamlConfig.getHomePose(),fcl_planner=self.hand_planner)
    
    def getRobotGraspJointVaules(self,
                                 q_goal,
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

    def realIK(self,robot_goal_pos_quat = [0.9, 0., 0.4, 1.0, 0.0, 0.0, 0.0],
                    robot_goal_tolerance = np.array([0.01,0.01,0.01,0.01,0.01,0.01]),
                    
                    thumb_goal_pos_quat = [0.10, 0.07, 0.06, 1.0,0.0, 0.0, 0.0],
                    thumb_goal_tolerance = np.array([0.01,0.01,0.01,999.0,999.0,999.0]),
                    
                    first_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                    first_goal_tolerance = np.array([0.01,0.01,0.01,0.01,0.01,0.01]),
                    
                    middle_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                    middle_goal_tolerance = np.array([0.01,0.01,0.01,0.01,0.01,0.01]),
                    
                    ring_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                    ring_goal_tolerance = np.array([0.01,0.01,0.01,0.01,0.01,0.01]),
                    
                    little_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                    little_goal_tolerance = np.array([0.01,0.01,0.01,0.01,0.01,0.01]),
                    ):
        robot_goal_pos_quat_np = np.array(robot_goal_pos_quat)
        thumb_goal_pos_quat_np = np.array(thumb_goal_pos_quat)
        first_goal_pos_quat_np = np.array(first_goal_pos_quat)
        middle_goal_pos_quat_np = np.array(middle_goal_pos_quat)
        ring_goal_pos_quat_np = np.array(ring_goal_pos_quat)
        little_goal_pos_quat_np = np.array(little_goal_pos_quat)
        
        tcp_trans_robot = robot_goal_pos_quat_np[0:3]
        tcp_trans_thumb = thumb_goal_pos_quat_np[0:3]
        # tcp_trans_first = first_goal_pos_quat_np[0:3]
        # tcp_trans_middle = middle_goal_pos_quat_np[0:3]
        # tcp_trans_ring = ring_goal_pos_quat_np[0:3]
        # tcp_trans_little = little_goal_pos_quat_np[0:3]
        
        tcp_position = np.hstack((tcp_trans_robot,
                                  tcp_trans_thumb))
                                #   tcp_trans_first,
                                #   tcp_trans_middle,
                                #   tcp_trans_ring,
                                #   tcp_trans_little))
        
        robot_goal_quat = robot_goal_pos_quat_np[3:]
        thumb_goal_quat = thumb_goal_pos_quat_np[3:]
        # first_goal_quat = first_goal_pos_quat_np[3:]
        # middle_goal_quat = middle_goal_pos_quat_np[3:]
        # ring_goal_quat = ring_goal_pos_quat_np[3:]
        # little_goal_quat = ring_goal_pos_quat_np[3:]
        
        tcp_quat = np.hstack([robot_goal_quat,
                              thumb_goal_quat,])
                            #   first_goal_quat,
                            #   middle_goal_quat,
                            #   ring_goal_quat,
                            #   little_goal_quat])
   
        tcp_tolerance = np.hstack([robot_goal_tolerance,
                                  thumb_goal_tolerance])
                                #   first_goal_tolerance,
                                #   middle_goal_tolerance,
                                #   ring_goal_tolerance,
                                #   little_goal_tolerance])
        
        q_res = self.hand_planner.hand_planner.solveIKWithRangedIKRealTime(tcp_position,tcp_quat,tcp_tolerance)
        return q_res

    def IK(self,tcp_pos_quat =None,current_activate=False):
        tcp_goal = np.array([tcp_pos_quat[0],tcp_pos_quat[1],tcp_pos_quat[2],
                             tcp_pos_quat[3],tcp_pos_quat[4],tcp_pos_quat[5],tcp_pos_quat[6]])
        
        if(current_activate == True):
            joint_current = self.bullet_env.bullet_robot.get_activate_joint_for_bullet()
            q_res = self.hand_planner.hand_planner.solveIK(HandPlannerIKType.RobotIK,tcp_goal,joint_current,hard_solve=False)
            return q_res
        else:
            q_res = self.hand_planner.hand_planner.solveIK(HandPlannerIKType.RobotIK,tcp_goal,hard_solve=False)
        return q_res

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

    def SolveAllIK(self,robot_goal_pos_quat = [0.9, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                        thumb_goal_pos_quat = [0.10, 0.07, 0.06, 1.0,0.0, 0.0, 0.0],
                        first_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                        middle_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                        ring_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                        little_goal_pos_quat = [0.06, 0.04, 0.06, 1.0,0.0, 0.0, 0.0],
                        is_hard = True):
        q_robot_g = self.IK(tcp_pos_quat=robot_goal_pos_quat)
        
        q_thumb_g = self.HandIK(tcp_pos_quat = thumb_goal_pos_quat,
                                current_activate = False,
                                finger_type = HandPlannerIKType.ThumbIK,
                                is_hard=is_hard)
        
        q_first_g = self.HandIK(tcp_pos_quat=first_goal_pos_quat,
                                current_activate=False,
                                finger_type=HandPlannerIKType.FirstIK,
                                is_hard=is_hard)
        
        q_middle_g = self.HandIK(tcp_pos_quat=middle_goal_pos_quat,
                                 current_activate=False,
                                 finger_type=HandPlannerIKType.MiddleIK,
                                 is_hard=is_hard)
        
        q_ring_g = self.HandIK(tcp_pos_quat=ring_goal_pos_quat,
                               current_activate=False,
                               finger_type=HandPlannerIKType.RingIK,
                               is_hard=is_hard)
        
        q_little_g = self.HandIK(tcp_pos_quat = little_goal_pos_quat,
                                 current_activate = False,
                                 finger_type = HandPlannerIKType.LittleIK,
                                 is_hard=is_hard)
        
        q_goal = self.getRobotGraspJointVaules(q_robot_g,
                                               q_thumb_values=q_thumb_g,
                                               q_first_values=q_first_g,
                                               q_middle_values=q_middle_g,
                                               q_ring_vaules=q_ring_g,
                                               q_little_values=q_little_g)
        return q_goal
        
    def run(self):
        time_stamp = self.parser.data[0]["time"]
        d_time = 0.0
        for item_map in self.parser.data:
            thump_list = None
            Index_list = None 
            Middle_list = None 
            Ring_list = None
            Pinky_list = None
            for key,value in item_map.items():
                if key == "time":
                    d_time = value -time_stamp
                    time_stamp = value
                if key == "Thumb":
                    thump_list = value
                if key == "Index":
                    Index_list = value
                if key == "Middle":
                    Middle_list = value
                if key == "Ring":
                    Ring_list = value
                if key == "Pinky":
                    Pinky_list = value
            start = time.time()
            q_res =self.SolveAllIK(thumb_goal_pos_quat=thump_list,
                                   first_goal_pos_quat=Index_list,
                                   middle_goal_pos_quat=Middle_list,
                                   ring_goal_pos_quat=Ring_list,
                                   little_goal_pos_quat=Pinky_list)
          
            end =time.time()
            duration = start -end;
            print(start-end)
            self.bullet_env.set_state(q_res)
            # print(q_res)
            time.sleep(d_time-duration)
                    


    
if __name__ == "__main__":
    parser = ShadowParser(json_path="ConvertData_z.json")
    parser.parser()
    env = HandIK_Env(parser=parser)
    env.run()
    

