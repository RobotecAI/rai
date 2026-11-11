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

"""The spec file, rai_whoami input, merging and rendering, on a fake graph."""

import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError
from rai.manifest import (
    ManifestSource,
    RobotManifest,
    RobotSpec,
    build_manifest,
    is_noise,
    load_spec,
    load_whoami,
    manifest_settings,
)

SPEC = Path(__file__).with_name("robot.toml")
MOTION = frozenset({"geometry_msgs/msg/Twist"})


class FakeNode:
    def __init__(self, graph: "FakeConnector") -> None:
        self.graph = graph

    def get_fully_qualified_name(self) -> str:
        return "/rai"

    def get_node_names_and_namespaces(self) -> list[tuple[str, str]]:
        return list(self.graph.nodes)


class FakeConnector:
    """The graph queries of ROS2Connector, from lists."""

    def __init__(self, topics=(), services=(), actions=(), nodes=()) -> None:
        self.topics = list(topics)
        self.services = list(services)
        self.actions = list(actions)
        self.nodes = list(nodes)
        self._node = FakeNode(self)

    def resolve(self, name: str, kind: str = "topic") -> str:
        return name if name.startswith("/") else f"/{name}"

    def get_topics_names_and_types(self):
        return list(self.topics)

    def get_services_names_and_types(self):
        return list(self.services)

    def get_actions_names_and_types(self):
        return list(self.actions)


def robot_graph() -> FakeConnector:
    return FakeConnector(
        topics=[
            ("/cmd_vel", ["geometry_msgs/msg/Twist"]),
            ("/camera/image_raw", ["sensor_msgs/msg/Image"]),
            ("/rosout", ["rcl_interfaces/msg/Log"]),
            ("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]),
            ("/fibonacci/_action/status", ["action_msgs/msg/GoalStatusArray"]),
            ("/debug/raw", ["std_msgs/msg/String"]),
            ("/joint_effort", ["std_msgs/msg/Float64"]),
        ],
        services=[
            ("/driver/get_parameters", ["rcl_interfaces/srv/GetParameters"]),
            ("/driver/get_type_description", ["type_description_interfaces/srv/X"]),
            ("/fibonacci/_action/send_goal", ["example_interfaces/action/X"]),
            ("/factory_reset", ["std_srvs/srv/Trigger"]),
            ("/reset_odom", ["std_srvs/srv/Trigger"]),
        ],
        actions=[("/fibonacci", ["example_interfaces/action/Fibonacci"])],
        nodes=[("driver", "/"), ("rai", "/"), ("_ros2cli_daemon", "/"), ("cam", "/a")],
    )


def test_spec_file_loads():
    spec = load_spec(SPEC)
    assert spec.name == "testbot"
    assert spec.limits == {
        "max_linear_speed_mps": 0.5,
        "max_angular_speed_radps": 1.5,
        "can_climb_stairs": False,
    }
    assert [c.name for c in spec.capabilities] == ["drive", "count", "dock"]
    assert spec.frames["base"] == "base_link"


@pytest.mark.parametrize(
    "data, error",
    [
        ({"nmae": "typo"}, "Extra inputs"),
        ({"capabilities": [{"name": "a"}, {"name": "a"}]}, "Duplicate capability"),
        ({"capabilities": [{"name": "a", "topics": ["bad name"]}]}, "valid ROS 2 name"),
        ({"capabilities": [{"name": "a", "topics": ["/a//b"]}]}, "valid ROS 2 name"),
        ({"capabilities": [{"name": ""}]}, "at least 1"),
        ({"hide": ["a b"]}, "pattern"),
        ({"motion_types": ["Twist"]}, "pkg/msg"),
        ({"limits": {"speed": [1, 2]}}, "limits"),
    ],
)
def test_spec_validation(data, error):
    with pytest.raises(ValidationError, match=error):
        RobotSpec.model_validate(data)


def test_settings_section(tmp_path, monkeypatch):
    config = tmp_path / "rai.toml"
    config.write_text('[manifest]\nspec = "robot.toml"\nmax_chars = 3000\n')
    settings = manifest_settings(config)
    assert settings.spec == Path("robot.toml") and settings.max_chars == 3000
    monkeypatch.setenv("RAI_MANIFEST__WHOAMI", "/opt/whoami")
    assert manifest_settings(config).whoami == Path("/opt/whoami")


def test_whoami_degrades_gracefully(tmp_path, caplog):
    assert load_whoami(None) is None
    assert load_whoami(tmp_path / "missing") is None
    assert "No rai_whoami description" in caplog.text
    generated = tmp_path / "generated"
    generated.mkdir()
    info = {"description": "A small rover.", "rules": ["be gentle"], "images": ["x"]}
    (generated / "info.json").write_text(json.dumps(info))
    who = load_whoami(tmp_path)
    assert who is not None and who.description == "A small rover."
    assert who.rules == ["be gentle"]

    class EmbodimentInfo:  # duck-typed rai_whoami model
        def model_dump(self, exclude=None):
            return {"description": "from the model", "capabilities": ["see"]}

    assert load_whoami(EmbodimentInfo()).capabilities == ["see"]


def test_noise_is_hidden():
    assert is_noise("topic", "/rosout") and is_noise("topic", "/x/_action/status")
    assert is_noise("service", "/n/set_parameters") and is_noise("node", "/_hidden")
    assert not is_noise("topic", "/cmd_vel") and not is_noise("service", "/reset")


def test_merge_annotates_the_graph():
    connector = robot_graph()
    manifest = build_manifest(connector, load_spec(SPEC), motion_types=MOTION)
    assert [t.name for t in manifest.topics] == [
        "/camera/image_raw",
        "/cmd_vel",
        "/joint_effort",
    ]
    assert [s.name for s in manifest.services] == ["/factory_reset", "/reset_odom"]
    assert manifest.nodes == ["/a/cam", "/driver"]  # not rai itself, not hidden
    topics = {t.name: t for t in manifest.topics}
    assert topics["/cmd_vel"].motion and topics["/cmd_vel"].capabilities == ["drive"]
    assert topics["/joint_effort"].motion  # spec motion_types
    assert not topics["/camera/image_raw"].motion
    assert manifest.services[0].forbidden
    caps = {c.name: c for c in manifest.capabilities}
    assert caps["drive"].topics == ["/cmd_vel"] and not caps["drive"].missing
    assert caps["dock"].missing == ["/dock"]
    assert manifest.actions[0].capabilities == ["count"]


def test_render_is_deterministic_and_complete():
    a = robot_graph()
    b = robot_graph()
    for lists in (b.topics, b.services, b.nodes):
        random.Random(7).shuffle(lists)
    spec = load_spec(SPEC)
    text = build_manifest(a, spec, motion_types=MOTION).render()
    assert text == build_manifest(b, spec, motion_types=MOTION).render()
    for expected in (
        "# Robot manifest: testbot",
        "- Stop before any person closer than 0.5 m.",
        "- max_linear_speed_mps: 0.5",
        "- can_climb_stairs: false",
        "- base: base_link",
        "- dock: Docking station. Uses actions /dock (missing).",
        "/cmd_vel geometry_msgs/msg/Twist [MOTION, cap:drive]",
        "/factory_reset std_srvs/srv/Trigger [FORBIDDEN]",
        "### Topics (3)",
    ):
        assert expected in text, expected
    assert "/rosout" not in text and "get_parameters" not in text
    assert "debug" not in text


def test_render_stays_within_the_size_bound():
    connector = robot_graph()
    connector.topics += [
        (f"/sensors/s{i:03}/data", ["sensor_msgs/msg/Range"]) for i in range(500)
    ]
    connector.nodes += [(f"sensor_{i:03}", "/") for i in range(200)]
    manifest = build_manifest(connector, load_spec(SPEC), motion_types=MOTION)
    for max_chars in (1500, 3000, 6000):
        text = manifest.render(max_chars)
        assert len(text) <= max_chars
        assert "/cmd_vel geometry_msgs/msg/Twist [MOTION" in text  # important first
        assert "Omitted for length:" in text
        assert "## Safety" in text
    assert len(manifest.render(200)) <= 200


def test_empty_graph_renders():
    text = RobotManifest().render()
    assert "unnamed robot" in text and "nothing discovered yet" in text


def test_source_refreshes_only_when_the_graph_changes():
    connector = robot_graph()
    source = ManifestSource(connector, spec=SPEC, motion_types=MOTION)
    assert source.changed()
    first = source.current()
    assert not source.changed() and source.refresh() is None
    assert source.current() is first
    connector.topics.append(("/n/_hidden", ["std_msgs/msg/String"]))  # noise
    connector.services.append(("/new/list_parameters", ["x/srv/Y"]))  # noise
    assert not source.changed()
    connector.topics.append(("/debug/more", ["std_msgs/msg/String"]))  # spec hide
    assert not source.changed()
    connector.actions.append(("/dock", ["x/action/Dock"]))
    assert source.changed()
    second = source.refresh()
    assert second is not None and second.fingerprint != first.fingerprint
    assert not {c.name: c for c in second.capabilities}["dock"].missing
