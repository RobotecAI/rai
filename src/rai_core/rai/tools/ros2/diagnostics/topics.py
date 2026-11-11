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

"""Topic endpoints with their QoS, and measured rate, jitter, bandwidth and delay."""

import asyncio
import statistics
import struct
import time
from typing import Any, List, Optional, Tuple, Type

from pydantic import BaseModel, Field
from rclpy.qos import (
    DurabilityPolicy,
    QoSCompatibility,
    QoSProfile,
    ReliabilityPolicy,
    qos_check_compatible,
)

from rai.communication.ros2.runtime import message_class
from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.diagnostics.common import (
    bounded,
    now_s,
    qos_text,
    seconds,
    size,
)

# Volatile matches every publisher, so a latched backlog does not count as traffic.
RATE_QOS = QoSProfile(
    depth=100,
    reliability=ReliabilityPolicy.BEST_AVAILABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _endpoint(info: Any, own: str) -> str:
    node = f"{info.node_namespace.rstrip('/')}/{info.node_name}"
    return f"{node} (this agent)" if node == own else node


def _topic_type(tool: BaseROS2Tool, topic: str) -> Tuple[str, List[str]]:
    name = tool.connector.resolve(topic)
    types = dict(tool.connector.get_topics_names_and_types()).get(name)
    if not types:
        raise ValueError(f"Topic {name} is not in the ROS graph")
    return name, types


class TopicInput(BaseModel):
    topic: str = Field(..., min_length=1, description="Topic name, e.g. /scan")


class GetROS2TopicInfoTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_args = {"topic": "topic"}
    name: str = "get_ros2_topic_info"
    description: str = (
        "A topic's type, its publishers and subscribers with their QoS, and QoS"
        " incompatibilities that stop messages from arriving"
    )
    args_schema: Type[TopicInput] = TopicInput

    async def _acall(self, topic: str) -> str:
        self.check_readable(topic)
        name, types = _topic_type(self, topic)
        publishers, subscribers = self.connector.get_topic_endpoints(name)
        own = self.connector.node_name
        lines = [f"{name} [{', '.join(types)}]"]
        for title, endpoints in (
            ("publishers", publishers),
            ("subscribers", subscribers),
        ):
            lines.append(f"{title} ({len(endpoints)}):")
            lines += bounded(
                (
                    f"  {_endpoint(e, own)}: {qos_text(e.qos_profile)}"
                    for e in endpoints
                ),
                30,
            )
        problems = []
        for pub in publishers:
            for sub in subscribers:
                compatibility, reason = qos_check_compatible(
                    pub.qos_profile, sub.qos_profile
                )
                if compatibility == QoSCompatibility.ERROR:
                    problems.append(
                        f"  {_endpoint(pub, own)} -> {_endpoint(sub, own)}:"
                        f" {reason.strip().rstrip(';')}"
                    )
        if problems:
            lines += ["QoS INCOMPATIBLE (no messages flow):", *bounded(problems, 20)]
        if len(types) > 1:
            lines.append("WARNING: several types on one topic; endpoints may not match")
        return "\n".join(lines)


def _has_header(cls: Any) -> bool:
    first = next(iter(cls.get_fields_and_field_types().items()), None)
    return first == ("header", "std_msgs/Header")


def _cdr_stamp(data: bytes) -> Optional[float]:
    """header.stamp of a serialized message whose first field is a Header: it
    follows the 4-byte CDR encapsulation, whose 2nd byte gives the byte order."""
    if len(data) < 12:
        return None
    sec, nanosec = struct.unpack_from("<iI" if data[1] & 1 else ">iI", data, 4)
    return sec + nanosec / 1e9


class MeasureROS2TopicRateInput(BaseModel):
    topic: str = Field(..., min_length=1, description="Topic name, e.g. /scan")
    window_s: float = Field(
        2.0, gt=0, le=30, description="How long to listen, in seconds"
    )


class MeasureROS2TopicRateTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_args = {"topic": "topic"}
    name: str = "measure_ros2_topic_rate"
    description: str = (
        "Listen to a topic for window_s seconds and report its rate (Hz), period"
        " jitter, bandwidth and, for stamped messages, header delay"
    )
    args_schema: Type[MeasureROS2TopicRateInput] = MeasureROS2TopicRateInput

    async def _acall(self, topic: str, window_s: float = 2.0) -> str:
        self.check_readable(topic)
        name, types = _topic_type(self, topic)
        cls = message_class(types[0])
        stamped = _has_header(cls)
        publishers, subscribers = self.connector.get_topic_endpoints(name)
        samples: List[Tuple[float, int, Optional[float]]] = []

        def on_message(data: bytes) -> None:
            stamp = _cdr_stamp(data) if stamped else None
            delay = now_s(self) - stamp if stamp else None
            samples.append((time.monotonic(), len(data), delay))

        subscription = self.connector.subscribe(
            name, on_message, msg_type=cls, qos=RATE_QOS, maxsize=None, raw=True
        )
        try:
            await asyncio.sleep(window_s)
        finally:
            subscription.close()
            self.connector.release_topic(name, raw=True)
        samples = samples[:]  # a late callback may still append
        arrivals = [s[0] for s in samples]
        sizes = [s[1] for s in samples]
        delays = [s[2] for s in samples if s[2] is not None]
        graph = f"publishers {len(publishers)}, subscribers {len(subscribers)}"
        if len(arrivals) < 2:
            return f"{name}: {len(arrivals)} messages in {window_s:g} s ({graph})"
        periods = [b - a for a, b in zip(arrivals, arrivals[1:])]
        mean = statistics.fmean(periods)
        parts = [
            f"{name}: {len(arrivals)} messages in {window_s:g} s, rate {1 / mean:.2f} Hz",
            f"period mean {seconds(mean)}, min {seconds(min(periods))},"
            f" max {seconds(max(periods))}, std {seconds(statistics.pstdev(periods))}",
            f"bandwidth {size(sum(sizes) / window_s)}/s"
            f" (mean {size(statistics.fmean(sizes))} per message)",
        ]
        if delays:
            parts.append(
                f"header delay mean {seconds(statistics.fmean(delays))},"
                f" max {seconds(max(delays))}"
            )
        return "; ".join([*parts, graph])
