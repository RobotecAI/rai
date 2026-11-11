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

"""The diagnostics toolkit against real nodes: publishers, a diagnostics
publisher, lifecycle nodes, parameters and a static + dynamic TF tree with a
stale and a disconnected frame."""

import re
import threading
import time
from typing import Any, Dict, Iterator

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.connectors.ros2_connector import _parse_frames
from rai.tools.ros2 import ROS2DiagnosticsToolkit
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup

DIAGNOSTICS = "/f5_diagnostics"


def _transform(parent: str, child: str, stamp: Any) -> TransformStamped:
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id, t.header.stamp = parent, child, stamp
    t.transform.rotation.w = 1.0
    return t


class Robot(Node):
    """Topics, logs, diagnostics, parameters and TF of a small fake robot."""

    def __init__(self) -> None:
        super().__init__("f5_talker")
        self.declare_parameter("speed", 1.5)
        self.declare_parameter(
            "mode", "auto", ParameterDescriptor(description="Driving mode")
        )
        self.chatter = self.create_publisher(String, "/f5_chatter", 10)
        self.pose = self.create_publisher(PoseStamped, "/f5_pose", 10)
        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.qos_pub = self.create_publisher(String, "/f5_qos", best_effort)
        self.qos_sub = self.create_subscription(
            String, "/f5_qos", lambda _: None, reliable
        )
        self.diagnostics = self.create_publisher(DiagnosticArray, DIAGNOSTICS, 10)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        now = self.get_clock().now().to_msg()
        self.static_tf.sendTransform(
            [
                _transform("base_link", "laser", now),
                _transform("island_root", "island", now),
            ]
        )
        battery = DiagnosticStatus(level=DiagnosticStatus.OK, name="battery")
        battery.hardware_id, battery.message = "bms", "charging"
        self.diagnostics.publish(DiagnosticArray(status=[battery]))  # once: goes stale
        self.ticks = 0
        self.create_timer(0.05, self.tick)

    def tick(self) -> None:
        self.ticks += 1
        now = self.get_clock().now()
        self.chatter.publish(String(data="hello"))
        pose = PoseStamped()
        pose.header.stamp = now.to_msg()
        self.pose.publish(pose)
        self.qos_pub.publish(String(data="lost"))
        self.tf.sendTransform(
            [
                _transform("map", "odom", now.to_msg()),
                _transform("odom", "base_link", now.to_msg()),
                _transform(
                    "base_link", "camera", (now - Duration(seconds=30)).to_msg()
                ),
            ]
        )
        if self.ticks % 10 == 1:
            self.get_logger().warning(f"wheel slip {self.ticks}")
            self.get_logger().error("motor overcurrent")
            motor = DiagnosticStatus(level=DiagnosticStatus.ERROR, name="motor")
            motor.hardware_id, motor.message = "can0", "overcurrent"
            motor.values = [KeyValue(key="current", value="12.3")]
            imu = DiagnosticStatus(level=DiagnosticStatus.OK, name="imu")
            imu.hardware_id, imu.message = "imu0", "ok"
            self.diagnostics.publish(DiagnosticArray(status=[motor, imu]))


@pytest.fixture(scope="module")
def tools(ros_setup: None) -> Iterator[Dict[str, Any]]:
    robot = Robot()
    configured = LifecycleNode("f5_lifecycle")
    idle = LifecycleNode("f5_lifecycle_idle")
    configured.trigger_configure()
    executor = SingleThreadedExecutor()
    for node in (robot, configured, idle):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    connector = ROS2Connector()
    deadline = time.monotonic() + 10
    while "/f5_lifecycle_idle" not in connector.get_node_names():
        assert time.monotonic() < deadline, "test nodes not discovered"
        time.sleep(0.05)
    toolkit = ROS2DiagnosticsToolkit(connector=connector)
    yield {"connector": connector, **{t.name: t for t in toolkit.get_tools()}}
    connector.close()
    executor.shutdown()
    thread.join(timeout=2)
    for node in (robot, configured, idle):
        node.destroy_node()


def run(tool: Any, **kwargs: Any) -> str:
    return tool.invoke(kwargs)


def test_list_nodes(tools) -> None:
    out = run(tools["list_ros2_nodes"])
    assert "/f5_talker" in out and "/f5_lifecycle" in out
    assert f"{tools['connector'].node_name} (this agent)" in out


def test_node_info(tools) -> None:
    out = run(tools["get_ros2_node_info"], node="f5_talker")
    assert "/f5_chatter [std_msgs/msg/String]" in out
    assert "/f5_qos [std_msgs/msg/String]" in out.split("subscribers")[1]
    assert "parameter services)" in out and "/f5_talker/get_parameters" not in out
    missing = run(tools["get_ros2_node_info"], node="/f5_talkr")
    assert "not in the ROS graph; did you mean /f5_talker" in missing


def test_topic_info_and_qos_incompatibility(tools) -> None:
    out = run(tools["get_ros2_topic_info"], topic="f5_chatter")
    assert out.startswith("/f5_chatter [std_msgs/msg/String]")
    assert "publishers (1):\n  /f5_talker: reliable, volatile" in out
    assert "INCOMPATIBLE" not in out
    out = run(tools["get_ros2_topic_info"], topic="/f5_qos")
    assert "QoS INCOMPATIBLE" in out and "/f5_talker -> /f5_talker" in out


def test_topic_rate(tools) -> None:
    connector = tools["connector"]
    out = run(tools["measure_ros2_topic_rate"], topic="/f5_chatter", window_s=1.5)
    rate = float(re.search(r"rate ([\d.]+) Hz", out).group(1))
    assert 12 < rate < 25, out  # published at 20 Hz
    assert "period mean" in out and "bandwidth" in out and "publishers 1" in out
    assert "header delay" not in out
    out = run(tools["measure_ros2_topic_rate"], topic="/f5_pose", window_s=1.0)
    delay = re.search(r"header delay mean ([\d.]+) (ms|s)", out)
    assert delay and delay.group(2) == "ms", out
    # The measurement's subscription is gone afterwards.
    own = connector.node_name.rsplit("/", 1)[-1]
    subscribers = connector.get_topic_endpoints("/f5_pose")[1]
    assert own not in [e.node_name for e in subscribers]
    out = run(tools["measure_ros2_topic_rate"], topic="/nope")
    assert "/nope is not in the ROS graph" in out


def test_parameters(tools) -> None:
    out = run(tools["get_ros2_parameters"], node="/f5_talker")
    assert "speed = 1.5" in out and "mode = 'auto'" in out
    out = run(
        tools["get_ros2_parameters"], node="/f5_talker", names=["mode"], describe=True
    )
    assert out == "/f5_talker: 1 parameters\nmode = 'auto'  # Driving mode"


def test_logs(tools) -> None:
    out = run(tools["get_ros2_logs"], min_level="warn", node="f5_talker", last_n=5)
    assert "WARN [f5_talker] wheel slip" in out and "ERROR [f5_talker] motor" in out
    assert len(out.splitlines()) <= 6
    out = run(tools["get_ros2_logs"], min_level="error", node="/f5_talker")
    assert "WARN" not in out.split("\n", 1)[1] and "motor overcurrent" in out
    out = run(tools["get_ros2_logs"], contains="wheel slip", since_s=60)
    assert "wheel slip" in out and "motor" not in out.split("\n", 1)[1]
    assert run(tools["get_ros2_logs"], node="nobody").startswith("0 matching")


def test_diagnostics(tools) -> None:
    tool = tools["get_ros2_diagnostics"]
    out = run(tool, topic=DIAGNOSTICS)
    assert out.splitlines()[0].startswith(f"{DIAGNOSTICS}: 2 statuses, worst ERROR")
    assert "ERROR motor [hw can0]: overcurrent | current=12.3" in out
    assert "OK imu [hw imu0]" in out
    assert "can0 ERROR" in out.splitlines()[-1]
    out = run(tool, topic=DIAGNOSTICS, stale_after_s=1.0, only_problems=True)
    assert "OK imu" not in out
    # battery was published once, before the subscription: never seen
    assert "battery" not in out
    time.sleep(1.2)
    out = run(tool, topic=DIAGNOSTICS, stale_after_s=0.6)
    assert "ERROR motor" in out  # republished every 0.5 s


def test_diagnostics_stale(tools) -> None:
    connector = tools["connector"]
    publisher = Node("f5_diag_once")
    pub = publisher.create_publisher(DiagnosticArray, DIAGNOSTICS, 10)
    try:
        status = DiagnosticStatus(level=DiagnosticStatus.WARN, name="fan")
        status.hardware_id, status.message = "fan0", "slow"
        deadline = time.monotonic() + 5
        while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
            time.sleep(0.05)
        pub.publish(DiagnosticArray(status=[status]))
        time.sleep(0.3)
        out = run(tools["get_ros2_diagnostics"], topic=DIAGNOSTICS)
        assert "WARN fan [hw fan0]: slow" in out
        time.sleep(1.2)
        out = run(tools["get_ros2_diagnostics"], topic=DIAGNOSTICS, stale_after_s=1.0)
        assert re.search(
            r"STALE fan \[hw fan0\]: no update for [\d.]+ s \(last WARN", out
        )
        assert "fan0 STALE" in out
    finally:
        publisher.destroy_node()
    assert connector.closed is False


def test_lifecycle_states(tools) -> None:
    tool = tools["get_ros2_lifecycle_states"]
    out = run(tool)
    assert "/f5_lifecycle: inactive" in out
    assert "/f5_lifecycle_idle: unconfigured" in out
    assert "/f5_talker" not in out
    assert (
        run(tool, node="f5_lifecycle") == "1 lifecycle nodes:\n/f5_lifecycle: inactive"
    )
    assert "is not a lifecycle node" in run(tool, node="/f5_talker")


def test_tf_tree(tools) -> None:
    tool = tools["get_ros2_tf_tree"]
    out = run(tool)
    assert "roots: island_root, map" in out
    assert re.search(r"\n  odom \([\d.]+ Hz, age [\d.]+ ms\)", out), out
    assert re.search(r"\n      laser \(static\)", out), out
    assert re.search(r"camera \([\d.]+ Hz, age 30\.\d+ s STALE\)", out), out
    assert "STALE (older than 1.00 s): base_link -> camera" in out
    out = run(tool, source_frame="laser", target_frame="map")
    assert "laser -> map: OK; chain laser - base_link - odom - map" in out
    out = run(tool, source_frame="camera", target_frame="map")
    assert "camera -> map: STALE link on the chain" in out
    out = run(tool, source_frame="island", target_frame="/map")
    assert "NOT CONNECTED: island is in the tree of island_root" in out
    out = run(tool, source_frame="ghost", target_frame="map")
    assert "MISSING: frame(s) ghost not in the TF tree" in out


def test_parse_frames() -> None:
    text = (
        "odom: \n  parent: 'map'\n  broadcaster: 'default_authority'\n"
        "  rate: 20.000\n  most_recent_transform: 12.500000\n"
        "  oldest_transform: 2.500000\n  buffer_length: 10.000\n"
        "laser: \n  parent: 'base_link'\n  most_recent_transform: 0.000000\n"
    )
    frames = _parse_frames(text)
    assert frames["odom"]["parent"] == "map" and frames["odom"]["rate"] == 20.0
    assert frames["laser"]["most_recent_transform"] == 0.0
    assert _parse_frames("[]") == {}


def test_interfaces(tools) -> None:
    out = run(tools["list_ros2_interfaces"], package="std_msgs")
    assert "std_msgs/msg/String" in out.splitlines()
    out = run(tools["list_ros2_interfaces"])
    assert re.search(r"^std_msgs: \d+ msg", out, re.M)
