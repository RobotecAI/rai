"""Robot IO shared by both slices: a fake in-memory robot and a real one on ROS2Connector.

Tool bodies live here too, so the two framework slices differ only in how they wrap them.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

STATUS_TOPIC = "/d1/status"
CMD_VEL = "/cmd_vel"


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    ACTUATE = "actuate"


class RobotIO(Protocol):
    def latest(self, topic: str, timeout_s: float) -> str: ...
    def publish_twist(self, linear_x: float, angular_z: float) -> None: ...


@dataclass
class FakeRobot:
    """In-memory robot: fixed topic contents, records every Twist."""

    topics: dict[str, str] = field(
        default_factory=lambda: {STATUS_TOPIC: '{"data": "battery=87% mode=idle"}'}
    )
    published: list[tuple[float, float]] = field(default_factory=list)

    def latest(self, topic: str, timeout_s: float) -> str:
        if topic not in self.topics:
            raise TimeoutError(f"no message on {topic} within {timeout_s}s")
        return self.topics[topic]

    def publish_twist(self, linear_x: float, angular_z: float) -> None:
        self.published.append((linear_x, angular_z))


class ConnectorRobot:
    """Real robot IO on rai_core's existing ROS2Connector."""

    def __init__(self, connector) -> None:
        self.connector = connector

    def latest(self, topic: str, timeout_s: float) -> str:
        from rosidl_runtime_py.convert import message_to_ordereddict

        msg = self.connector.receive_message(topic, timeout_sec=timeout_s)
        return json.dumps(message_to_ordereddict(msg.payload))

    def publish_twist(self, linear_x: float, angular_z: float) -> None:
        from rai.communication.ros2 import ROS2Message

        payload = {"linear": {"x": linear_x}, "angular": {"z": angular_z}}
        self.connector.send_message(
            ROS2Message(payload=payload), CMD_VEL, msg_type="geometry_msgs/msg/Twist"
        )


# ---- tool bodies (framework-neutral, async-native) ----


async def get_latest_message(io: RobotIO, topic: str, timeout_s: float = 2.0) -> str:
    """Return the latest message on a ROS 2 topic as JSON."""
    return await asyncio.to_thread(io.latest, topic, timeout_s)


async def publish_twist(
    io: RobotIO, linear_x: float, angular_z: float, duration_s: float = 0.0
) -> str:
    """Drive the base: publish a Twist to /cmd_vel at 10 Hz for duration_s, then stop.

    Cancellation stops the loop at the next 0.1 s tick and always sends a zero Twist.
    """
    io.publish_twist(linear_x, angular_z)
    try:
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            io.publish_twist(linear_x, angular_z)
    finally:
        if duration_s > 0:
            io.publish_twist(0.0, 0.0)
    return (
        f"published Twist(linear.x={linear_x}, angular.z={angular_z}) for {duration_s}s"
    )


async def observe_status(io: RobotIO) -> str:
    """StateBased context: a fresh sensor summary for one model call."""
    try:
        return f"[robot state] {STATUS_TOPIC}: {await get_latest_message(io, STATUS_TOPIC, 0.5)}"
    except TimeoutError:
        return f"[robot state] {STATUS_TOPIC}: no data"


# ---- real ROS fixture pieces ----


class RobotSim:
    """A tiny ROS node standing in for a robot: publishes /d1/status, records /cmd_vel."""

    def __init__(self) -> None:
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String

        self.node = Node(f"d1_robot_sim_{int(time.time() * 1e6) % 10**9}")
        self.published: list[tuple[float, float]] = []
        pub = self.node.create_publisher(String, STATUS_TOPIC, 10)
        self.node.create_timer(
            0.05, lambda: pub.publish(String(data="battery=87% mode=idle"))
        )
        self.node.create_subscription(
            Twist,
            CMD_VEL,
            lambda m: self.published.append((m.linear.x, m.angular.z)),
            10,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
