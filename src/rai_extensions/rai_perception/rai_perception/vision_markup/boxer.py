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

"""Deprecated: Use rai_perception.algorithms.boxer instead.

This module is deprecated and will be removed in a future version.
All classes delegate to rai_perception.algorithms.boxer.
"""

import warnings
from os import PathLike
from pathlib import Path

from rai_perception.algorithms.boxer import Box
from rai_perception.algorithms.boxer import GDBoxer as AlgorithmsGDBoxer


class GDBoxer(AlgorithmsGDBoxer):
    """Deprecated: Use rai_perception.algorithms.boxer.GDBoxer instead.

    This class is deprecated and will be removed in a future version.
    It delegates to rai_perception.algorithms.boxer.GDBoxer.
    """

    def __init__(
        self,
        weight_path: str | PathLike,
        use_cuda: bool = True,
    ):
        """Initialize GDBoxer (deprecated wrapper).

        Args:
            weight_path: Path to model weights file
            use_cuda: Whether to use CUDA if available
        """
        warnings.warn(
            "rai_perception.vision_markup.boxer.GDBoxer is deprecated. "
            "Use rai_perception.algorithms.boxer.GDBoxer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map old hardcoded config path to new default

        config_path = Path(__file__).parent.parent / "configs" / "gdino_config.py"
        super().__init__(weight_path, config_path=config_path, use_cuda=use_cuda)


# Re-export Box class for backward compatibility
__all__ = ["Box", "GDBoxer"]
