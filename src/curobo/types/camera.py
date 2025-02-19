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
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch
from torch.profiler import record_function

# CuRobo
from curobo.geom.cv import (
    get_projection_rays,
    project_depth_using_rays,
    project_pointcloud_to_depth,
)
from curobo.types.math import Pose


@dataclass
class CameraObservation:
    name: str = "camera_image"
    #: rgb image format is BxHxWxchannels
    rgb_image: Optional[torch.Tensor] = None
    depth_image: Optional[torch.Tensor] = None
    image_segmentation: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    projection_rays: Optional[torch.Tensor] = None
    resolution: Optional[List[int]] = None
    pose: Optional[Pose] = None
    intrinsics: Optional[torch.Tensor] = None
    timestamp: float = 0.0

    def filter_depth(self, distance: float = 0.01):
        self.depth_image = torch.where(self.depth_image < distance, 0, self.depth_image)

    @property
    def shape(self):
        return self.rgb_image.shape

    @record_function("camera/copy_")
    def copy_(self, new_data: CameraObservation):
        if self.rgb_image is not None:
            self.rgb_image.copy_(new_data.rgb_image)
        if self.depth_image is not None:
            self.depth_image.copy_(new_data.depth_image)
        if self.image_segmentation is not None:
            self.image_segmentation.copy_(new_data.image_segmentation)
        if self.projection_matrix is not None:
            self.projection_matrix.copy_(new_data.projection_matrix)
        if self.projection_rays is not None:
            self.projection_rays.copy_(new_data.projection_rays)
        if self.pose is not None:
            self.pose.copy_(new_data.pose)
        self.resolution = new_data.resolution

    @record_function("camera/clone")
    def clone(self):

        return CameraObservation(
            depth_image=self.depth_image.clone() if self.depth_image is not None else None,
            rgb_image=self.rgb_image.clone() if self.rgb_image is not None else None,
            intrinsics=self.intrinsics.clone() if self.intrinsics is not None else None,
            resolution=self.resolution,
            pose=self.pose.clone() if self.pose is not None else None,
        )

    def to(self, device: torch.device):
        if self.rgb_image is not None:
            self.rgb_image = self.rgb_image.to(device=device)
        if self.depth_image is not None:
            self.depth_image = self.depth_image.to(device=device)
        return self

    def get_pointcloud(self):
        if self.projection_rays is None:
            self.update_projection_rays()
        depth_image = self.depth_image
        if len(self.depth_image.shape) == 2:
            depth_image = self.depth_image.unsqueeze(0)
        point_cloud = project_depth_using_rays(depth_image, self.projection_rays)
        return point_cloud

    def get_image_from_pointcloud(self, pointcloud, output_image: Optional[torch.Tensor] = None):
        if output_image is None:
            output_image = torch.zeros(
                (self.depth_image.shape[0], self.depth_image.shape[1]),
                dtype=pointcloud.dtype,
                device=pointcloud.device,
            )

        depth_image = project_pointcloud_to_depth(pointcloud, output_image=output_image)
        return depth_image

    def update_projection_rays(self):
        intrinsics = self.intrinsics
        if len(intrinsics.shape) == 2:
            intrinsics = intrinsics.unsqueeze(0)
        project_rays = get_projection_rays(
            self.depth_image.shape[-2], self.depth_image.shape[-1], intrinsics
        )

        if self.projection_rays is None:
            self.projection_rays = project_rays

        self.projection_rays.copy_(project_rays)
