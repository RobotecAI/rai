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

"""Record/replay harness: run agents, tools and aggregators on recorded data,
without a robot (needs the ``recorder`` extra and ROS 2).

- :class:`ReplayPlayer` publishes recorded topics into a live ROS graph
  (optionally with ``/clock``), so the real connector runs unchanged.
- :class:`ReplayConnector` serves the ROS2Connector API from a recording on a
  virtual clock, offline, and records egress instead of sending it.
- :class:`ReplayChatModel` answers with recorded model responses;
  :class:`AgentReplay` re-runs a recorded agent session with both and reports
  every divergence from the recording.
- ``expect_*`` assertions for tests.

Recordings: F2 flight recordings (``FlightRecorder``), rosbag2 MCAP and
rosbag2 sqlite3 bags (file or directory)::

    replay = AgentReplay("session.mcap")
    agent = ReActAgent(replay.model, make_tools(replay.connector), name="pilot")
    report = await replay.run(agent)
    assert report.ok, report
"""

from .connector import (
    EgressRecord,
    ReplayConnector,
    ReplayEnded,
    ReplayGoalHandle,
    ReplaySubscription,
    ScriptedGoal,
)
from .expect import (
    ToolCall,
    expect_called,
    expect_goal,
    expect_no_motion,
    expect_nothing_sent,
    expect_published,
    expect_tool_called,
    tool_calls,
)
from .llm import ReplayChatModel, ReplayDivergence, ReplayExhausted
from .player import ReplayPlayer
from .session import AgentReplay, Divergence, ReplayReport
from .source import Bag, RawMessage, open_bag

__all__ = [
    "AgentReplay",
    "Bag",
    "Divergence",
    "EgressRecord",
    "RawMessage",
    "ReplayChatModel",
    "ReplayConnector",
    "ReplayDivergence",
    "ReplayEnded",
    "ReplayExhausted",
    "ReplayGoalHandle",
    "ReplayPlayer",
    "ReplayReport",
    "ReplaySubscription",
    "ScriptedGoal",
    "ToolCall",
    "expect_called",
    "expect_goal",
    "expect_no_motion",
    "expect_nothing_sent",
    "expect_published",
    "expect_tool_called",
    "open_bag",
    "tool_calls",
]
