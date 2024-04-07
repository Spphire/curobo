import numpy as np
import transforms3d as t3d
import time
from typing import List
import threading
import json
import requests
import math

from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.rollout.rollout_base import Goal

import logging

def get_custom_world_model(table_height=0.02):
    table = Cuboid(
        name="table",
        dims=[4.0,4.0,4.0],
        pose=[0.0, 0.0, -2.0+table_height, 1.0, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    return WorldConfig(
        cuboid=[table],
    )

class Ur10eController():
    def __init__(self,
                 world_model:WorldConfig,
                 ros_ip="10.9.11.1",
                 ros_port="8000",
                 config_name="ur10e.yml"):
        self.DOF=6
        self.ros_ip = ros_ip
        self.ros_port = ros_port

        self.homing_state=False
        self.tracking_state=False

        self.tensor_args = TensorDeviceType()
        self.world_model = world_model

        self.get_q_from_ros()
        print("connected to ros!")
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), config_name))["robot_cfg"]
        robot_cfg['kinematics']['cspace']['retract_config'] = self.get_current_q()
        self.robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',]
                            #'WRJ2',] # 'WRJ1',]

        

        
        self.hand_model = Shadowhand_Model()
        self.init_ik()
        self.init_mpc()
        
        #self.init_hand_mpc()
        

    def init_mpc(self):
        
        self.mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_model,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            use_lbfgs=False,
            use_es=False,
            use_mppi=True,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.mpc = MpcSolver(self.mpc_config)
        self.retract_cfg = self.mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)
        retract_state = JointState.from_position(self.retract_cfg, joint_names=self.mpc.rollout_fn.joint_names)
        state = self.mpc.rollout_fn.compute_kinematics(retract_state)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        self.init_retract(self.mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0))

        goal = Goal(
            current_state=self.get_current_jointstate(),
            goal_state=self.get_current_jointstate(),
            goal_pose=self.get_current_Pose(),
        )
        self.goal_buffer = self.mpc.setup_solve_single(goal, 1)
        self.mpc.update_goal(self.goal_buffer)

        

        self.start_real_tcp = self.get_current_tcp()
        self.start_unity_tcp = np.zeros(7)
        self.start_unity_tcp[4] = 1

        self.past_pose = None
        self.past_rot = None

        for i in range(10):
            #print(i)
            self.get_q_from_ros()
            self.mpc_excute(target=self.get_current_tcp(), can_move=False)
            time.sleep(0.05)
        print("mpc inited!")

    def init_ik(self):
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_model,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        self.q6=0


    def init_retract(self,tensor):
        self.retract_cfg = tensor 
        if hasattr(self, 'mpc'):
            self.retract_state = JointState.from_position(self.tensor_args.to_device(self.retract_cfg), 
                                                        joint_names=self.mpc.rollout_fn.joint_names)
            state = self.mpc.rollout_fn.compute_kinematics(self.retract_state)
            new_pos = state.ee_pos_seq.cpu().numpy()
            new_quat = state.ee_quat_seq
        elif hasattr(self, 'motion_gen'):
            state = self.motion_gen.kinematics.get_state(self.tensor_args.to_device(self.retract_cfg).view(1,self.DOF))
            new_pos = np.ravel(state.ee_pose.to_list())[:3]
            new_quat = np.ravel(state.ee_pose.to_list())[3:]
        else:
            raise NotImplementedError

        self.retract_pose = Pose(position=self.tensor_args.to_device(new_pos), 
                                 quaternion=self.tensor_args.to_device(new_quat))
    
    def get_q_from_ros(self):
        try:
            q = requests.post("http://"+self.ros_ip+":"+self.ros_port+"/getJoints")
            if len(list(json.loads(q.content).values()))==self.DOF:
                self.q_from_ros = list(json.loads(q.content).values())
                #self.q_from_ros.append(0.0)
                # self.q_from_ros.append(0.0)
            else:
                print("DOF wrong")
                quit()
        except:
            print("Failed to get q from ros")
            quit()

    def get_current_q(self) -> List[float]:
        return self.q_from_ros
        
    def get_current_jointstate(self):
        q = np.array(self.get_current_q()).flatten()
        return JointState.from_position(self.tensor_args.to_device([q.tolist()]), joint_names=self.joint_names)

    def get_current_tcp(self) -> np.ndarray:
        if hasattr(self, 'mpc'):
            state = self.mpc.rollout_fn.compute_kinematics(
                self.get_current_jointstate()
            )
            pos = state.ee_pos_seq.cpu().numpy().flatten()
            tcp = np.array(pos.tolist() + state.ee_quat_seq.cpu().numpy().flatten().tolist())
            return tcp
        # elif hasattr(self, 'motion_gen'):
            # state = self.motion_gen.kinematics.get_state(self.tensor_args.to_device(self.get_current_q()).view(1,self.DOF))
            # tcp = np.ravel(state.ee_pose.to_list())
            # return tcp
        else:
            raise NotImplementedError
        
    def get_current_Pose(self):
        temp = self.get_current_tcp()
        return Pose(
            position=self.tensor_args.to_device(temp[:3]),
            quaternion=self.tensor_args.to_device(temp[3:]),
        )
    
    def can_move(self):
        return not self.homing_state and self.tracking_state

    def move(self, target_q: List[float]):
        if self.can_move():
            requests.post("http://"+self.ros_ip+":"+self.ros_port+"/move",json.dumps({'q': target_q}), timeout=0.05)
    
    def move_hand(self, target_q: List[float]):
        if self.can_move():
            requests.post("http://"+self.ros_ip+":"+self.ros_port+"/move_hand",json.dumps({'q': target_q}), timeout=0.05)

    def set_start_tcp(self, pos_quat:np.ndarray):
        self.start_real_tcp = self.get_current_tcp()
        self.start_unity_tcp = pos_quat
        self.tracking_state=True

    def get_relative_target(self, pos_from_unity):

        target=np.zeros(7)
        target[:3]=pos_from_unity[:3] - self.start_unity_tcp[:3] + self.start_real_tcp[:3]
        target_rot_mat = t3d.quaternions.quat2mat(pos_from_unity[3:]) \
                        @ np.linalg.inv(t3d.quaternions.quat2mat(self.start_unity_tcp[3:])) \
                        @ t3d.quaternions.quat2mat(self.start_real_tcp[3:])
        target[3:]=t3d.quaternions.mat2quat(target_rot_mat).tolist()

        return target
    
    def robot_go_home():
        pass

    def mpc_excute(self, target:np.ndarray, can_move=True):
        ik_suc = False
        target = target.flatten()
        target_position, target_orientation = target[:3],target[3:]

        if self.past_pose is None: self.past_pose = target_position + 1.0
        if self.past_rot is None: self.past_rot = target_orientation +1.0


        ik_goal = Pose(
            position=self.tensor_args.to_device(target_position.tolist()),
            quaternion=self.tensor_args.to_device(target_orientation.tolist()),
        )

        if (
            np.linalg.norm(target_position - self.past_pose) > 1e-2 
            or np.linalg.norm(target_orientation - self.past_rot) > 1e-3
        ):
            
            
            
            self.goal_buffer.goal_pose.copy_(ik_goal)
            #print("-mpc...")
            self.mpc.update_goal(self.goal_buffer)
            
            #print("mpc..")
            self.past_pose = target_position
            self.past_rot = target_orientation
        
        ik_result = self.ik_solver.solve_single(ik_goal)
        if ik_result.success:
            ik_suc = True
            self.q6 = ik_result.js_solution.position.cpu().numpy().reshape(self.DOF)[-1]

        mpc_result = self.mpc.step(self.get_current_jointstate(), max_attempts=2)
        state = mpc_result.js_action.position.cpu().numpy().reshape(self.DOF)
        #print(self.q6, state[:self.DOF].flatten().tolist()[-1])
        if can_move:  
            target_q = state[:self.DOF].flatten().tolist()
            # if ik_suc:
            #     if target_q[-1]>self.q6+math.pi:
            #         self.q6+=math.pi*2
            #     elif target_q[-1]<self.q6-math.pi:
            #         self.q6-=math.pi*2
            #     self.q6=np.clip(self.q6,-2*math.pi,2*math.pi)
            alpha = 5
            #if abs(target_q[-1]-self.q6) > math.pi/9:
            #target_q[-1] = (target_q[-1]*alpha + self.q6)/(alpha+1)
            self.move(target_q)
        #else:
        

class Shadowhand_Model():
    def __init__(self, config_name = "shadowhand.yml") -> None:
        self.world_model = get_custom_world_model()
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), config_name))["robot_cfg"]
        self.hand_robot_cfg = RobotConfig.from_dict(self.robot_cfg, self.tensor_args)
        self.init_hand_mpc()
        self.init_hand_ik()

    def init_hand_mpc(self):
        self.hand_mpc_config = MpcSolverConfig.load_from_robot_config(
            self.hand_robot_cfg,
            self.world_model,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            use_lbfgs=False,
            use_es=False,
            use_mppi=True,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.hand_mpc = MpcSolver(self.hand_mpc_config)

    def init_hand_ik(self):
        self.hand_ik_config = IKSolverConfig.load_from_robot_config(
            self.hand_robot_cfg,
            self.world_model,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.hand_ik_config)
    
    def get_kinematics_state(self, q: np.ndarray):
        joinstate = JointState.from_position(self.tensor_args.to_device(q.tolist()), joint_names=self.hand_mpc.rollout_fn.joint_names)
        return self.hand_mpc.rollout_fn.compute_kinematics(joinstate)
    
    def get_link_pose(self, q: np.ndarray):
        q = self.tensor_args.to_device([q.tolist()])
        return self.ik_solver.fk(q).link_pose

if __name__ == "__main__":
    #uc = Ur10eController(get_custom_world_model())
    sh = Shadowhand_Model()
    link_pose = sh.get_link_pose(np.array([0.0]*24))
    print(link_pose['ffdistal'], link_pose["palm"])
