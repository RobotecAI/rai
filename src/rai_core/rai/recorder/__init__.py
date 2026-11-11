# Copyright (C) 2026 Robotec.AI
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

"""Agent flight recorder: LLM turns, tool calls, hook decisions, observations
and selected ROS topics in one MCAP file, on one clock, for Foxglove and replay.

Needs the ``recorder`` extra (``pip install 'rai_core[recorder]'``)::

    from rai.recorder import FlightRecorder

    with FlightRecorder("run.mcap", agent=agent, topics=["/cmd_vel", "/camera/image_raw"]):
        rai.run(agent)

    from rai.recorder import Recording
    for call in Recording("run.mcap").values("/rai/agent/tools"):
        ...
"""

from .reader import RecordedMessage, Recording
from .recorder import FlightRecorder, install
from .schemas import (
    DECISIONS,
    EVENTS,
    IMAGES,
    LLM,
    LOG,
    OBSERVATIONS,
    TOOLS,
    observation_images,
)

__all__ = [
    "DECISIONS",
    "EVENTS",
    "IMAGES",
    "LLM",
    "LOG",
    "OBSERVATIONS",
    "TOOLS",
    "FlightRecorder",
    "RecordedMessage",
    "Recording",
    "install",
    "observation_images",
]
