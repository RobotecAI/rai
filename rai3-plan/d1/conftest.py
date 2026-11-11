import importlib.util
import os
import time

import pytest

from .common.robot import ConnectorRobot, FakeRobot, RobotSim

# Spike-only deps (see REPORT.md): skip a slice when `uv sync` has removed them.
collect_ignore = [
    d
    for d, mod in [
        ("langchain", "langchain_mcp_adapters"),
        ("pydanticai", "pydantic_ai"),
    ]
    if importlib.util.find_spec(mod) is None or importlib.util.find_spec("mcp") is None
]


@pytest.fixture(scope="session")
def ros():
    import rclpy

    assert os.environ.get("ROS_DOMAIN_ID"), "set ROS_DOMAIN_ID (31 for D1)"
    rclpy.init()
    sim = RobotSim()
    yield sim
    sim.close()
    rclpy.shutdown()


@pytest.fixture(params=["fake", "ros"])
def robot(request):
    """(io, published): the same tests run on a fake robot and on ROS2Connector + a sim node."""
    if request.param == "fake":
        io = FakeRobot()
        yield io, io.published
        return
    from rai.communication.ros2.connectors import ROS2Connector

    sim = request.getfixturevalue("ros")
    sim.published.clear()
    conn = ROS2Connector(executor_type="single_threaded")
    deadline = time.monotonic() + 5  # wait for discovery both ways
    while time.monotonic() < deadline and not (
        conn.node.count_publishers("/d1/status")
        and sim.node.count_publishers("/cmd_vel") >= 0
    ):
        time.sleep(0.05)
    yield ConnectorRobot(conn), sim.published
    conn.shutdown()


def wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()
