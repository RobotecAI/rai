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

"""Writing small bags with exact stamps for replay tests."""

from pathlib import Path
from typing import Any

from rai.communication.ros2.egress import type_name
from rai.recorder.schemas import ros2_schema
from rclpy.serialization import serialize_message

S = 1_000_000_000
T0 = 1_000 * S  # recordings start at 1000 s

Messages = list[tuple[str, Any, int]]  # topic, message, stamp (ns)


def write_mcap(path: Path, messages: Messages) -> Path:
    """An MCAP laid out like rosbag2's (ros2msg schemas, CDR channels)."""
    from mcap.writer import Writer

    with path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="ros2", library="tests")
        schemas: dict[str, int] = {}
        channels: dict[str, int] = {}
        for topic, msg, stamp in sorted(messages, key=lambda m: m[2]):
            type_ = type_name(type(msg))
            if type_ not in schemas:
                encoding, text = ros2_schema(type_)
                schemas[type_] = writer.register_schema(type_, encoding, text.encode())
            if topic not in channels:
                channels[topic] = writer.register_channel(topic, "cdr", schemas[type_])
            writer.add_message(
                channels[topic],
                log_time=stamp,
                publish_time=stamp,
                data=serialize_message(msg),
            )
        writer.finish()
    return path


def write_rosbag2(path: Path, messages: Messages, storage: str) -> Path:
    """A real rosbag2 bag directory (rosbag2_py), storage "mcap" or "sqlite3"."""
    import rosbag2_py

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    created: set[str] = set()
    for i, (topic, msg, _) in enumerate(messages):
        if topic in created:
            continue
        created.add(topic)
        kwargs = dict(name=topic, type=type_name(type(msg)), serialization_format="cdr")
        try:
            metadata = rosbag2_py.TopicMetadata(id=i, **kwargs)
        except TypeError:  # older rosbag2_py: no id
            metadata = rosbag2_py.TopicMetadata(**kwargs)
        writer.create_topic(metadata)
    for topic, msg, stamp in sorted(messages, key=lambda m: m[2]):
        writer.write(topic, serialize_message(msg), stamp)
    del writer  # closes the bag
    return path
