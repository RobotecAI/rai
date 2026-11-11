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

"""Formatting shared by the diagnostics tools: bounded, single-line, with units."""

from typing import Any, Iterable, List

from rclpy.qos import HistoryPolicy

from rai.tools.ros2.base import BaseROS2Tool

MAX_LINES = 100


def clip(text: Any, limit: int = 200) -> str:
    """One line of at most limit characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def bounded(lines: Iterable[str], limit: int = MAX_LINES) -> List[str]:
    lines = list(lines)
    if len(lines) <= limit:
        return lines
    return [*lines[:limit], f"... {len(lines) - limit} more"]


def seconds(value: float) -> str:
    if abs(value) < 1:
        return f"{value * 1000:.1f} ms"
    return f"{value:.2f} s" if abs(value) < 100 else f"{value:.0f} s"


def size(value: float) -> str:
    for unit in ("B", "KB", "MB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def now_s(tool: BaseROS2Tool) -> float:
    """The connector's clock (sim time aware) in seconds."""
    return tool.connector.now().nanoseconds / 1e9


def stamp_s(stamp: Any) -> float:
    return stamp.sec + stamp.nanosec / 1e9


def node_name(name: str) -> str:
    """ "talker" -> "/talker"."""
    return name if name.startswith("/") else f"/{name}"


def qos_text(qos: Any) -> str:
    """A QoS profile in a few words; unset or infinite durations are left out."""
    parts = [qos.reliability.name.lower(), qos.durability.name.lower()]
    if qos.history == HistoryPolicy.KEEP_LAST and qos.depth:
        parts.append(f"depth {qos.depth}")
    for field in ("deadline", "lifespan", "liveliness_lease_duration"):
        ns = getattr(qos, field).nanoseconds
        if 0 < ns < 10**15:  # rmw reports "infinite" as about 9.2e18 ns
            parts.append(f"{field.split('_')[0]} {seconds(ns / 1e9)}")
    return ", ".join(parts)
