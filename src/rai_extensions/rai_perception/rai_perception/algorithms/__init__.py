# Copyright (C) 2025 Julia Jia
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

"""Low-level core algorithms for perception.

This module provides direct access to detection, segmentation, and point cloud
processing algorithms for expert-level users who need full control over parameters.
"""

from .boxer import Box, GDBoxer
from .point_cloud import depth_to_point_cloud
from .segmenter import GDSegmenter

__all__ = [
    "Box",
    "GDBoxer",
    "GDSegmenter",
    "depth_to_point_cloud",
]
