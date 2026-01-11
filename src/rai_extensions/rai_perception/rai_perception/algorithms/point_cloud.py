# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Point cloud processing algorithm: depth_to_point_cloud.

Low-level algorithm for converting depth images to 3D point clouds.
"""

import numpy as np
from numpy.typing import NDArray


def depth_to_point_cloud(
    depth_image: NDArray[np.float32], fx: float, fy: float, cx: float, cy: float
) -> NDArray[np.float32]:
    """Convert depth image to 3D point cloud.

    Args:
        depth_image: Depth image as numpy array of shape (H, W)
        fx: Focal length in x direction
        fy: Focal length in y direction
        cx: Principal point x coordinate
        cy: Principal point y coordinate

    Returns:
        Point cloud as numpy array of shape (N, 3) where N is number of valid points.
        Points with zero depth are filtered out.
    """
    height, width = depth_image.shape
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    z = depth_image
    x = (x_grid - float(cx)) * z / float(fx)
    y = (y_grid - float(cy)) * z / float(fy)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points = points[points[:, 2] > 0]
    return points.astype(np.float32, copy=False)
