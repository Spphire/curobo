#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


# script running (ubuntu):
#

############################################################


# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="bimanual_srhand_ur.yml", help="robot configuration to load")
args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os
import socket
import json
import requests
# Third Party
import carb
import numpy as np
import transforms3d as t3d
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


############################################################
def unity2zup_right_frame(pos_quat):
        pos_quat*=np.array([1,-1,1,1,-1,1,-1])
        rot_mat = t3d.quaternions.quat2mat(pos_quat[3:])
        pos_vec = pos_quat[:3]
        T=np.eye(4)
        T[:3,:3]= rot_mat
        T[:3,3]=pos_vec
        fit_mat = t3d.euler.axangle2mat([0,1,0],np.pi/2)
        fit_mat = fit_mat@t3d.euler.axangle2mat([0,0,1],-np.pi/2)
        target_rot_mat=fit_mat@rot_mat
        target_pos_vec=fit_mat@pos_vec
        target = np.array(target_pos_vec.tolist()+t3d.quaternions.mat2quat(target_rot_mat).tolist())
        return target

def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    r_past_pose = None
    l_past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    sim_robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = sim_robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = sim_robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot, robot_prim_path = add_robot_to_scene(sim_robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        trajopt_tsteps=40,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=0.03,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        collision_activation_distance=0.025,
        acceleration_scale=1.0,
        fixed_iters_trajopt=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    link_names = motion_gen.kinematics.link_names
    ee_link_name = motion_gen.kinematics.ee_link
    kin_state = motion_gen.kinematics.get_state(motion_gen.get_retract_config().view(1, -1))

    link_retract_pose = kin_state.link_pose
    target_links = {}
    names = []
    for i in link_names:
        if i != ee_link_name:
            k_pose = np.ravel(link_retract_pose[i].to_list())
            color = np.random.randn(3) * 0.2
            color[0] += 0.5
            color[1] = 0.5
            color[2] = 0.0
            target_links[i] = cuboid.VisualCuboid(
                "/World/target_" + i,
                position=np.array(k_pose[:3]),
                orientation=np.array(k_pose[3:]),
                color=color,
                scale=[0.01,0.01,0.05],
            )
            names.append("/World/target_" + i)



    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
        links_goal_pose={ln: Pose(
                    position=tensor_args.to_device(t.get_world_pose()[0]),
                    quaternion=tensor_args.to_device(t.get_world_pose()[1]),
                ) for (ln, t) in zip(target_links.keys(),target_links.values())}
    )


    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    target.set_world_pose(position=state.ee_pos_seq.cpu().numpy().flatten(),
                          orientation=state.ee_quat_seq.cpu().numpy().flatten())

    local_ip= "192.168.2.218"
    port=8082
    address = (local_ip, port)
    socket_obj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_obj.setblocking(0)
    socket_obj.bind(address)
    r_hand_q = np.zeros([1,24])
    l_hand_q = np.zeros([1,24])
    finger_joint_name = ['rh_WRJ2', 'rh_WRJ1',
      'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',
      'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
      'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
      'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
      'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1']
    l_finger_joint_name = ['lh_WRJ2', 'lh_WRJ1',
      'lh_THJ5', 'lh_THJ4', 'lh_THJ3', 'lh_THJ2', 'lh_THJ1',
      'lh_LFJ5', 'lh_LFJ4', 'lh_LFJ3', 'lh_LFJ2', 'lh_LFJ1',
      'lh_RFJ4', 'lh_RFJ3', 'lh_RFJ2', 'lh_RFJ1',
      'lh_MFJ4', 'lh_MFJ3', 'lh_MFJ2', 'lh_MFJ1',
      'lh_FFJ4', 'lh_FFJ3', 'lh_FFJ2', 'lh_FFJ1']
    r_tracking_state = False
    l_tracking_state = False
    r_start_sim_tcp = np.zeros(7)
    r_start_unity_tcp = np.zeros(7)
    l_start_sim_tcp = np.zeros(7)
    l_start_unity_tcp = np.zeros(7)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    spheres = None
    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 2:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                # only_paths=[obstacles_path],
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)
        
        if True and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        # position and orientation of target virtual cube:
        r_cube_position, r_cube_orientation = target.get_world_pose()
        l_cube_position, l_cube_orientation = list(target_links.values())[0].get_world_pose()
        try:
            data, _ = socket_obj.recvfrom(4096)
            s=json.loads(data)
            q = requests.post("http://127.0.0.1:8080/get_thumb_q",json.dumps(s['rightHand']))
            q = np.array(json.loads(q.content)).reshape(5,)
            r_hand_q = np.array(s['rightHand']['q'])
            r_hand_q[2:7] = q
            r_hand_q = r_hand_q.reshape([1,24])/180*np.pi

            l_hand_q = np.array(s['leftHand']['q'])
            l_hand_q = l_hand_q.reshape([1,24])/180*np.pi
            #print(hand_q)
            r_pos_from_unity = unity2zup_right_frame(np.array(s['rightHand']["pos"]+s['rightHand']["quat"]))
            l_pos_from_unity = unity2zup_right_frame(np.array(s['leftHand']["pos"]+s['leftHand']["quat"]))

            if s['rightHand']['cmd']==2:
                if not r_tracking_state:
                    r_tracking_state=True
                    r_start_sim_tcp = np.array(r_cube_position.tolist()+r_cube_orientation.tolist())
                    r_start_unity_tcp = r_pos_from_unity

            if s['leftHand']['cmd']==2:
                if not l_tracking_state:
                    l_tracking_state=True
                    l_start_sim_tcp = np.array(l_cube_position.tolist()+l_cube_orientation.tolist())
                    l_start_unity_tcp = l_pos_from_unity

            if s['rightHand']['cmd']==-2:
                if r_tracking_state:
                    r_tracking_state = False

            if s['leftHand']['cmd']==-2:
                if l_tracking_state:
                    l_tracking_state = False

            if r_tracking_state:
                r_target_tcp=np.zeros(7)
                r_target_tcp[:3]=r_pos_from_unity[:3] - r_start_unity_tcp[:3] + r_start_sim_tcp[:3]
                r_target_rot_mat = t3d.quaternions.quat2mat(r_pos_from_unity[3:]) \
                                @ np.linalg.inv(t3d.quaternions.quat2mat(r_start_unity_tcp[3:])) \
                                @ t3d.quaternions.quat2mat(r_start_sim_tcp[3:])
                r_target_tcp[3:]=t3d.quaternions.mat2quat(r_target_rot_mat).tolist()
                if np.linalg.norm(r_target_tcp[:3]-r_cube_position)>0.05:
                    print('right loss sync')
                    r_tracking_state=False
                else:
                    target.set_world_pose(position=r_target_tcp[:3],
                          orientation=r_target_tcp[3:])
                
            if l_tracking_state:
                l_target_tcp=np.zeros(7)
                l_target_tcp[:3]=l_pos_from_unity[:3] - l_start_unity_tcp[:3] + l_start_sim_tcp[:3]
                l_target_rot_mat = t3d.quaternions.quat2mat(l_pos_from_unity[3:]) \
                                @ np.linalg.inv(t3d.quaternions.quat2mat(l_start_unity_tcp[3:])) \
                                @ t3d.quaternions.quat2mat(l_start_sim_tcp[3:])
                l_target_tcp[3:]=t3d.quaternions.mat2quat(l_target_rot_mat).tolist()
                if np.linalg.norm(l_target_tcp[:3]-l_cube_position)>0.05:
                    print('left loss sync')
                    l_tracking_state=False
                else:
                    list(target_links.values())[0].set_world_pose(position=l_target_tcp[:3],
                          orientation=l_target_tcp[3:])
        except:
            pass
        
        r_cube_position, r_cube_orientation = target.get_world_pose()
        l_cube_position, l_cube_orientation = list(target_links.values())[0].get_world_pose()

        if r_past_pose is None:
            r_past_pose = r_cube_position + 1.0

        if l_past_pose is None:
            l_past_pose = l_cube_position + 1.0

        if (np.linalg.norm(r_cube_position - r_past_pose) > 1e-3 \
            or np.linalg.norm(l_cube_position - l_past_pose) > 1e-3):
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = r_cube_position
            ee_orientation_teleop_goal = r_cube_orientation
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            link_poses = {}
            for i in target_links.keys():
                #print(i)
                c_p, c_rot = target_links[i].get_world_pose()
                link_poses[i] = Pose(
                    position=tensor_args.to_device(c_p),
                    quaternion=tensor_args.to_device(c_rot),
                )
                goal_buffer.links_goal_pose[i].copy_(link_poses[i])
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            r_past_pose = r_cube_position
            l_past_pose = l_cube_position

        # if not changed don't call curobo:

        # get robot current state:
        sim_js = robot.get_joints_state()
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        common_js_names = []
        current_state.copy_(cu_js)

        mpc_result = mpc.step(current_state, max_attempts=2)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)
        for n in finger_joint_name + l_finger_joint_name:
            idx_list.append(robot.get_dof_index(n))

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state
        #print(cmd_state.position.cpu().numpy())
        art_action = ArticulationAction(
            np.concatenate([cmd_state.position.cpu().numpy(),r_hand_q,l_hand_q],axis=1),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(3):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()
