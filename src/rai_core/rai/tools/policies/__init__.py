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


"""Learned policies (VLAs such as pi0, GR00T, SmolVLA) as cancellable RAI tools.

The clients need the ``policies`` extra (``pip install 'rai_core[policies]'``);
`RunPolicyTool` and its input/output models need ROS 2.
"""

import importlib
from typing import Any

from .client import (
    ActionChunk,
    Observation,
    PolicyClient,
    PolicyConnectionError,
    PolicyError,
    PolicyTimeoutError,
)
from .gr00t import Gr00tClient
from .http_client import HttpPolicyClient
from .openpi import OpenPIClient

_ROS = {
    "ArrayInput",
    "ImageInput",
    "JointStateInput",
    "JointTrajectoryOutput",
    "PolicyProgress",
    "RunPolicyTool",
    "TwistOutput",
}


def __getattr__(name: str) -> Any:  # the tool imports ROS; the clients do not
    if name in _ROS:
        return getattr(importlib.import_module(".tool", __name__), name)
    if name == "FakePolicyServer":
        return importlib.import_module(".fake", __name__).FakePolicyServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ActionChunk",
    "ArrayInput",
    "FakePolicyServer",
    "Gr00tClient",
    "HttpPolicyClient",
    "ImageInput",
    "JointStateInput",
    "JointTrajectoryOutput",
    "Observation",
    "OpenPIClient",
    "PolicyClient",
    "PolicyConnectionError",
    "PolicyError",
    "PolicyProgress",
    "PolicyTimeoutError",
    "RunPolicyTool",
    "TwistOutput",
]
