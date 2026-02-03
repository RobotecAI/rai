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

"""Segmentation model registry.

Maps model names to algorithm classes and config identifiers.
Enables model switching via ROS2 parameters without code changes.

Example:
    AlgorithmClass, config_path = get_model("grounded_sam")
    segmenter = AlgorithmClass(weights_path, config_path=config_path)
"""

from typing import Tuple, Type

from rai_perception.algorithms.segmenter import GDSegmenter

# Registry: model_name -> (AlgorithmClass, config_path)
# Config loading is model-specific:
# - Return full file path (str) for models that need it (e.g., GroundingDINO)
# - Return None for models that handle config internally (e.g., SAM2 uses Hydra config module)
# When adding a model, check how its library loads configs and return the appropriate identifier.
_SEGMENTATION_REGISTRY: dict[str, Tuple[Type, str | None]] = {
    "grounded_sam": (
        GDSegmenter,
        None,  # Uses Hydra config module, no file path needed
    ),
}


def get_model(name: str) -> Tuple[Type, str | None]:
    """Get segmentation model class and config path by name.

    Args:
        name: Model name (e.g., "grounded_sam")

    Returns:
        Tuple of (AlgorithmClass, config_path)
        - config_path is a full file path for models that need it, or None for models that handle
          config loading internally (e.g., via Hydra config module)
        - See registry comments for model-specific config loading requirements

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in _SEGMENTATION_REGISTRY:
        available = ", ".join(_SEGMENTATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown segmentation model '{name}'. Available models: {available}"
        )
    return _SEGMENTATION_REGISTRY[name]


def list_available_models() -> list[str]:
    """List all available segmentation model names.

    Returns:
        List of model names
    """
    return list(_SEGMENTATION_REGISTRY.keys())
