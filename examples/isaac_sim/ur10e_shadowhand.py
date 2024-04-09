import torch
a = torch.zeros(4, device="cuda:0")

import argparse
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
parser.add_argument("--robot", type=str, default="ur10e.yml", help="robot configuration to load")
parser.add_argument("--robot_curobo", type=str, default="ur10e.yml", help="robot configuration to load")
args = parser.parse_args()

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

import os
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")

from helper import add_extensions, add_robot_to_scene

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
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")

    usd_help = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    j_names_full = robot_cfg["kinematics"]["cspace"]["joint_names"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()


    curobo_robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot_curobo))["robot_cfg"]
    j_names = curobo_robot_cfg["kinematics"]["cspace"]["joint_names"]
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    world_cfg = WorldConfig()
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        curobo_robot_cfg,
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

    mpc_config = MpcSolverConfig.load_from_robot_config(
        curobo_robot_cfg,
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
        # links_goal_pose={ln: Pose(
        #             position=tensor_args.to_device(t.get_world_pose()[0]),
        #             quaternion=tensor_args.to_device(t.get_world_pose()[1]),
        #         ) for (ln, t) in zip(target_links.keys(),target_links.values())}
    )
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)
    past_pose = None

    target.set_world_pose(position=state.ee_pos_seq.cpu().numpy().flatten(),
                          orientation=state.ee_quat_seq.cpu().numpy().flatten())
    
    usd_help.load_stage(my_world.stage)
    init_world = False
    step = 0
    cmd_state_full = None
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
            idx_list = [robot.get_dof_index(x) for x in j_names_full]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

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

        sim_js = robot.get_joints_state()
        sim_js_names = robot.dof_names
        DOF = len(mpc.rollout_fn.joint_names)
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions[:DOF]),
            velocity=tensor_args.to_device(sim_js.velocities[:DOF]) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities[:DOF]) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities[:DOF]) * 0.0,
            joint_names=sim_js_names[:DOF],
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        
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

        cube_position, cube_orientation = target.get_world_pose()
        if past_pose is None:
            past_pose = cube_position + 1.0
        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            ik_goal = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )
            # link_poses = {}
            # for i in target_links.keys():
            #     print(i)
            #     c_p, c_rot = target_links[i].get_world_pose()
            #     link_poses[i] = Pose(
            #         position=tensor_args.to_device(c_p),
            #         quaternion=tensor_args.to_device(c_rot),
            #     )
            #     goal_buffer.links_goal_pose[i].copy_(link_poses[i])
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position
        #print(cu_js.position)
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
        mpc_result = mpc.step(cu_js, max_attempts=2)

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        cmd_state = cmd_state_full.get_ordered_joint_state(sim_js_names[:DOF])

        idx_list = []
        for n in sim_js_names:
            idx_list.append(robot.get_dof_index(n))
        
        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            #np.concatenate([cmd_state.position.cpu().numpy(),np.zeros([1,30-DOF])],axis=1),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(3):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")

if __name__ == "__main__":
    main()
    simulation_app.close()