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

"""Record on the robot, replay in CI.

On the robot (needs the robot's topics and an LLM)::

    python examples/replay_in_ci.py record session.mcap "Where is the robot?"

In CI (no robot, no LLM, no ROS graph; a few seconds)::

    python examples/replay_in_ci.py check session.mcap

``check`` re-runs the recorded session offline: the model answers with the
recorded responses, the tools read the recorded topics on a virtual clock, and
what they publish or call is recorded instead of sent. It exits with 1 and
prints where the run departed from the recording if a tool or prompt changed.
Use the same ``tools()`` in both modes: that is what CI checks.
"""

import argparse
import asyncio
import sys
from typing import Any

from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.tools.ros2.generic.topics import (
    GetROS2TopicsNamesAndTypesTool,
    GetROS2TransformTool,
    ReceiveROS2MessageTool,
)

SYSTEM_PROMPT = "You are a mobile robot. Answer using your tools."
TOPICS = ["/tf", "/tf_static", "/odom"]  # what the tools read


def tools(connector: Any) -> list[Any]:
    access = dict(connector=connector, readable=TOPICS)
    return [
        GetROS2TopicsNamesAndTypesTool(**access),
        ReceiveROS2MessageTool(**access),
        GetROS2TransformTool(**access),
    ]


@ROS2Context()
def record(path: str, task: str) -> None:
    from rai.recorder import FlightRecorder

    connector = ROS2Connector()
    agent = ReActAgent(
        tools=tools(connector), system_prompt=SYSTEM_PROMPT, name="robot"
    )
    with FlightRecorder(path, agent=agent, topics=TOPICS, connector=connector):
        answer = asyncio.run(agent.ainvoke(task))
    print(answer.content)


def check(path: str) -> int:
    from rai.replay import AgentReplay

    replay = AgentReplay(path, agent="robot")
    agent = ReActAgent(
        replay.model,
        tools(replay.connector),
        system_prompt=SYSTEM_PROMPT,
        name="robot",
    )
    report = asyncio.run(replay.run(agent))
    print(report)
    return 0 if report.ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record", help="run the agent on the robot and record it")
    rec.add_argument("path")
    rec.add_argument("task")
    chk = sub.add_parser("check", help="re-run a recorded session offline")
    chk.add_argument("path")
    args = parser.parse_args()
    if args.mode == "record":
        record(args.path, args.task)
        return 0
    return check(args.path)


if __name__ == "__main__":
    sys.exit(main())
