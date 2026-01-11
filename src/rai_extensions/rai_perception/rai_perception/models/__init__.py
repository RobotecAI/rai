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

"""Model registry for detection and segmentation algorithms.

This module provides centralized registries that map model names to algorithm classes
and their configuration paths. This enables dynamic model selection via ROS2 parameters
without hardcoding model-specific logic.

Example usage:
    # In a ROS2 service node, read model name from parameter
    from rai.communication.ros2 import get_param_value
    from rai_perception.models import get_detection_model

    model_name = get_param_value(node, "model_name", default="grounding_dino")
    AlgorithmClass, config_path = get_detection_model(model_name)
    algorithm = AlgorithmClass(weights_path, config_path=config_path)
"""

from .detection import get_model as get_detection_model
from .detection import list_available_models as list_detection_models
from .segmentation import get_model as get_segmentation_model
from .segmentation import list_available_models as list_segmentation_models

__all__ = [
    "get_detection_model",
    "get_segmentation_model",
    "list_detection_models",
    "list_segmentation_models",
]
