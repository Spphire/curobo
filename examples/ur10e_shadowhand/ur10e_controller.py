import numpy as np
import transforms3d as t3d
import time
from typing import List
import threading
import json
import requests

from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose

class Ur10eController():
    def __init__(self,
                 world_model:WorldConfig,
                 ros_ip="10.53.21.95",
                 ros_port=8000,):
        self.homing_state=False
        self.tracking_state=False

        self.tensor_args = TensorDeviceType()
        self.world_model = world_model
        self.joint_names = ['ra_shoulder_pan_joint', 'ra_shoulder_lift_joint', 
                            'ra_elbow_joint', 'ra_wrist_1_joint', 
                            'ra_wrist_2_joint', 'ra_wrist_3_joint']

        self.DOF=6

        try:
            q = requests.post("http://10.53.21.95:8000/getJoints")
            if len(list(json.loads(q.content).values()))==self.DOF:
                print("connected to ros!")
        except:
            print("Failed to connect to ros!")
            quit()

    def init_mpc(self,config_name="ur10e.yml"):
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), config_name))["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
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
        

        self.init_retract(self.mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0))

        self.start_real_tcp = self.get_current_tcp()
        self.start_unity_tcp = np.zeros(7)
        self.start_unity_tcp[4] = 1

        self.past_pose = None
        self.past_rot = None

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
        
    def get_current_q(self) -> List[float]:
        try:
            q = requests.post("http://10.53.21.95:8000/getJoints")
            if len(list(json.loads(q.content).values()))==self.DOF:
                return list(json.loads(q.content).values())
        except:
            print("Failed to get q from ros")
            quit()
        
    def get_current_jointstate(self):
        q = self.get_current_q()
        return JointState.from_position(self.tensor_args.to_device(q), joint_names=self.joint_names)

    def get_current_tcp(self) -> np.ndarray:
        if hasattr(self, 'mpc'):
            state = self.mpc.rollout_fn.compute_kinematics(
                self.get_current_jointstate()
            )
            pos = state.ee_pos_seq.cpu().numpy().flatten()
            tcp = np.array(pos.tolist() + state.ee_quat_seq.cpu().numpy().flatten().tolist())
            return tcp
        elif hasattr(self, 'motion_gen'):
            state = self.motion_gen.kinematics.get_state(self.tensor_args.to_device(self.get_current_q()).view(1,self.DOF))
            tcp = np.ravel(state.ee_pose.to_list())
            return tcp
        else:
            raise NotImplementedError
        
    def get_current_Pose(self):
        temp = self.get_current_tcp()
        return Pose(
            position=self.tensor_args.to_device(temp[:3]),
            quaternion=self.tensor_args.to_device(temp[3:]),
        )
    
    def can_move(self):
        return (not self.locked) and (self.homing_state or self.tracking_state)

    def move(self, target_q: List[float]):
        if self.can_move():
            requests.post("http://10.53.21.95:8000/move",json.dumps({'q': target_q}))

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

    def mpc_excute(self, target:np.ndarray):
        target_position, target_orientation = target[:3],target[3:]

        if self.past_pose is None: self.past_pose = target_position + 1.0
        if self.past_rot is None: self.past_rot = target_orientation +1.0


        if (
            np.linalg.norm(target_position - self.past_pose) > 1e-2 
            or np.linalg.norm(target_orientation - self.past_rot) > 1e-3
        ):

            ik_goal = Pose(
                position=self.tensor_args.to_device(target_position),
                quaternion=self.tensor_args.to_device(target_orientation),
            )

            self.goal_buffer.goal_pose.copy_(ik_goal)
            self.mpc.update_goal(self.goal_buffer)
            self.past_pose = target_position
            self.past_rot = target_orientation

        mpc_result = self.mpc.step(self.get_current_jointstate(), max_attempts=2)
        state = mpc_result.js_action.position.cpu().numpy().reshape(self.DOF)
        self.move(state[:self.DOF].flatten().tolist())
