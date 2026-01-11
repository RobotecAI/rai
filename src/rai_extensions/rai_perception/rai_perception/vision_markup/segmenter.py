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

"""Deprecated: Use rai_perception.algorithms.segmenter instead.

This module is deprecated and will be removed in a future version.
All classes delegate to rai_perception.algorithms.segmenter.
"""

import warnings
from os import PathLike

from rai_perception.algorithms.segmenter import GDSegmenter as AlgorithmsGDSegmenter


class GDSegmenter(AlgorithmsGDSegmenter):
    """Deprecated: Use rai_perception.algorithms.segmenter.GDSegmenter instead.

    This class is deprecated and will be removed in a future version.
    It delegates to rai_perception.algorithms.segmenter.GDSegmenter.
    """

    def __init__(
        self,
        weight_path: str | PathLike,
        config_path: str | PathLike | None = None,
        use_cuda: bool = True,
    ):
        """Initialize GDSegmenter (deprecated wrapper).

        Args:
            weight_path: Path to model weights file
            config_path: Ignored (kept for API compatibility, SAM2 uses Hydra config module)
            use_cuda: Whether to use CUDA if available
        """
        warnings.warn(
            "rai_perception.vision_markup.segmenter.GDSegmenter is deprecated. "
            "Use rai_perception.algorithms.segmenter.GDSegmenter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Delegate to algorithms version (config_path is ignored for SAM2 anyway)
        super().__init__(weight_path, config_path=config_path, use_cuda=use_cuda)


__all__ = ["GDSegmenter"]
