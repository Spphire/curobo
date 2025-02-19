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
# Standard Library
from typing import Dict, Optional

# Third Party
import numpy as np
import yourdfpy
from lxml import etree

# CuRobo
from curobo.cuda_robot_model.kinematics_parser import KinematicsParser, LinkParams
from curobo.cuda_robot_model.types import JointType
from curobo.geom.types import Mesh as CuroboMesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_warn
from curobo.util_file import join_path


class UrdfKinematicsParser(KinematicsParser):
    def __init__(
        self,
        urdf_path,
        load_meshes: bool = False,
        mesh_root: str = "",
        extra_links: Optional[Dict[str, LinkParams]] = None,
    ) -> None:
        # load robot from urdf:
        self._robot = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=load_meshes,
            build_scene_graph=False,
            mesh_dir=mesh_root,
            filename_handler=yourdfpy.filename_handler_null,
        )
        super().__init__(extra_links)

    def build_link_parent(self):
        self._parent_map = {}
        for j in self._robot.joint_map:
            self._parent_map[self._robot.joint_map[j].child] = self._robot.joint_map[j].parent

    def _find_parent_joint_of_link(self, link_name):
        for j_idx, j in enumerate(self._robot.joint_map):
            if self._robot.joint_map[j].child == link_name:
                return j_idx, j
        log_error("Link is not attached to any joint")

    def _get_joint_name(self, idx):
        joint = self._robot.joint_names[idx]
        return joint

    def get_link_parameters(self, link_name: str, base=False) -> LinkParams:
        link_params = self._get_from_extra_links(link_name)
        if link_params is not None:
            return link_params
        body_params = {}
        body_params["link_name"] = link_name

        if base:
            body_params["parent_link_name"] = None
            joint_transform = np.eye(4)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_axis = None
            body_params["joint_id"] = 0
        else:
            body_params["parent_link_name"] = self._parent_map[link_name]

            jid, joint_name = self._find_parent_joint_of_link(link_name)
            body_params["joint_id"] = jid
            joint = self._robot.joint_map[joint_name]
            joint_transform = joint.origin
            if joint_transform is None:
                joint_transform = np.eye(4)
            joint_type = joint.type
            joint_limits = None
            joint_axis = None
            if joint_type != "fixed":
                if joint_type != "continuous":
                    joint_limits = {
                        "effort": joint.limit.effort,
                        "lower": joint.limit.lower,
                        "upper": joint.limit.upper,
                        "velocity": joint.limit.velocity,
                    }
                else:
                    log_warn("Converting continuous joint to revolute")
                    joint_type = "revolute"
                    joint_limits = {
                        "effort": joint.limit.effort,
                        "lower": -3.14 * 2,
                        "upper": 3.14 * 2,
                        "velocity": joint.limit.velocity,
                    }

                joint_axis = joint.axis

                body_params["joint_limits"] = [joint_limits["lower"], joint_limits["upper"]]
                body_params["joint_velocity_limits"] = [
                    -1.0 * joint_limits["velocity"],
                    joint_limits["velocity"],
                ]

        body_params["fixed_transform"] = joint_transform
        body_params["joint_name"] = joint_name

        body_params["joint_axis"] = joint_axis

        if joint_type == "fixed":
            joint_type = JointType.FIXED
        elif joint_type == "prismatic":
            if joint_axis[0] == 1:
                joint_type = JointType.X_PRISM
            if joint_axis[1] == 1:
                joint_type = JointType.Y_PRISM
            if joint_axis[2] == 1:
                joint_type = JointType.Z_PRISM
            if joint_axis[0] == -1:
                joint_type = JointType.X_PRISM_NEG
            if joint_axis[1] == -1:
                joint_type = JointType.Y_PRISM_NEG
            if joint_axis[2] == -1:
                joint_type = JointType.Z_PRISM_NEG

        elif joint_type == "revolute":
            if joint_axis[0] == 1:
                joint_type = JointType.X_ROT
            if joint_axis[1] == 1:
                joint_type = JointType.Y_ROT
            if joint_axis[2] == 1:
                joint_type = JointType.Z_ROT
            if joint_axis[0] == -1:
                joint_type = JointType.X_ROT_NEG
            if joint_axis[1] == -1:
                joint_type = JointType.Y_ROT_NEG
            if joint_axis[2] == -1:
                joint_type = JointType.Z_ROT_NEG
        else:
            log_error("Joint type not supported")

        body_params["joint_type"] = joint_type
        link_params = LinkParams(**body_params)

        return link_params

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = ""):
        # read all link meshes and update their mesh paths by prepending mesh_dir
        links = self._robot.link_map
        for k in links.keys():
            # read visual and collision
            vis = links[k].visuals
            for i in range(len(vis)):
                m = vis[i].geometry.mesh
                if m is not None:
                    m.filename = join_path(mesh_dir, m.filename)
            col = links[k].collisions
            for i in range(len(col)):
                m = col[i].geometry.mesh
                if m is not None:
                    m.filename = join_path(mesh_dir, m.filename)

    def get_urdf_string(self):
        txt = etree.tostring(self._robot.write_xml(), method="xml", encoding="unicode")
        return txt

    def get_link_mesh(self, link_name):
        m = self._robot.link_map[link_name].visuals[0].geometry.mesh
        mesh_pose = self._robot.link_map[link_name].visuals[0].origin
        # read visual material:
        if mesh_pose is None:
            mesh_pose = [0, 0, 0, 1, 0, 0, 0]
        else:
            # convert to list:
            mesh_pose = Pose.from_matrix(mesh_pose).to_list()

        return CuroboMesh(
            name=link_name,
            pose=mesh_pose,
            scale=m.scale,
            file_path=m.filename,
        )
