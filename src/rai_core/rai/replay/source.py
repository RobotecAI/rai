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

"""Recorded ROS topics as serialized CDR: F2 flight recordings, rosbag2 MCAP
and rosbag2 sqlite3 bags behind one interface."""

import heapq
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from rai.recorder import Recording


@dataclass(frozen=True)
class RawMessage:
    topic: str
    type: str  # "pkg/msg/Name"
    stamp: int  # ns on the recording's clock
    data: bytes  # CDR


class Bag:
    """The ROS topics of a recording, read in stamp order.

    Opens an F2 flight recording or a rosbag2 MCAP file (``.mcap``), a
    rosbag2 sqlite3 file (``.db3``), or a rosbag2 directory (every ``.mcap``
    or ``.db3`` file in it, merged). Only CDR topics are ROS topics: the JSON
    channels of a flight recording (``/rai/...``) are left out.
    """

    def __init__(self, path: Union[str, Path, Recording]) -> None:
        if isinstance(path, Recording):
            self.recording: Optional[Recording] = path
            self.path = path.path
            files = [path.path]
        else:
            self.path = Path(path)
            self.recording = None
            if self.path.is_dir():
                files = sorted(
                    p for p in self.path.iterdir() if p.suffix in (".mcap", ".db3")
                )
                if not files:
                    raise FileNotFoundError(f"No .mcap or .db3 file in {self.path}")
            else:
                files = [self.path]
        self._files: list[_File] = []
        for file in files:
            if file.suffix == ".db3":
                self._files.append(_Sqlite(file))
            else:
                if self.recording is None and len(files) == 1:
                    self.recording = Recording(file)
                recording = self.recording if len(files) == 1 else Recording(file)
                assert recording is not None
                self._files.append(_Mcap(recording))
        self.topics: dict[str, str] = {}
        for f in self._files:
            self.topics.update(f.topics)

    @property
    def session(self) -> dict[str, str]:
        """The ``rai.session`` metadata of a flight recording (empty for bags)."""
        if self.recording is None:
            return {}
        return self.recording.metadata.get("rai.session", {})

    @property
    def clock(self) -> str:
        """What the stamps are: "ros" (sim time) or "wall" for flight
        recordings, "bag" (receive time, wall clock) for rosbag2."""
        return self.session.get("clock", "bag")

    def messages(
        self,
        topics: Optional[Iterable[str]] = None,
        *,
        start: Optional[int] = None,
    ) -> Iterator[RawMessage]:
        """Messages of topics (every ROS topic if None) in stamp order, from
        start (ns) on."""
        wanted = set(self.topics) if topics is None else set(topics) & set(self.topics)
        streams = [f.messages(wanted, start) for f in self._files]
        if len(streams) == 1:
            return streams[0]
        return heapq.merge(*streams, key=lambda m: m.stamp)

    def bounds(self) -> tuple[int, int]:
        """(first, last) stamp of the ROS messages; (0, 0) if there are none."""
        first = last = None
        for message in self.messages():
            if first is None:
                first = message.stamp
            last = message.stamp
        return (first or 0, last or 0)


class _File:
    topics: dict[str, str]

    def messages(self, topics: set[str], start: Optional[int]) -> Iterator[RawMessage]:
        raise NotImplementedError


class _Mcap(_File):
    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.topics = {
            c.topic: recording.schemas[c.schema_id].name
            for c in recording.channels.values()
            if c.message_encoding == "cdr" and c.schema_id in recording.schemas
        }

    def messages(self, topics: set[str], start: Optional[int]) -> Iterator[RawMessage]:
        if not topics:
            return
        if self.recording.complete:  # indexed: stream in log_time order
            from mcap.reader import make_reader

            with self.recording.path.open("rb") as f:
                reader = make_reader(f)
                try:  # mcap >= 1.3 deprecates start_time
                    found = reader.iter_messages(
                        topics=sorted(topics), starting_at=start, log_time_order=True
                    )
                except TypeError:
                    found = reader.iter_messages(
                        topics=sorted(topics), start_time=start, log_time_order=True
                    )
                for schema, channel, message in found:
                    if schema is None:  # schemaless: not a ROS message
                        continue
                    yield RawMessage(
                        channel.topic, schema.name, message.log_time, message.data
                    )
            return
        for m in self.recording.messages(topics, decode=False):  # a cut file
            if start is None or m.log_time >= start:
                yield RawMessage(m.topic, m.schema, m.log_time, m.data)


class _Sqlite(_File):
    """A rosbag2 sqlite3 file: tables topics(id, name, type, ...) and
    messages(topic_id, timestamp, data)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, name, type, serialization_format FROM topics"
            ).fetchall()
        self._ids = {name: id_ for id_, name, _, fmt in rows if fmt == "cdr"}
        self.topics = {name: type_ for _, name, type_, fmt in rows if fmt == "cdr"}

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def messages(self, topics: set[str], start: Optional[int]) -> Iterator[RawMessage]:
        ids = {self._ids[t]: t for t in topics if t in self._ids}
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        query = (
            f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({marks})"
            " AND timestamp >= ? ORDER BY timestamp, id"
        )
        params: list[Any] = [*ids, start if start is not None else -(2**63)]
        db = self._connect()
        try:
            for topic_id, stamp, data in db.execute(query, params):
                topic = ids[topic_id]
                yield RawMessage(topic, self.topics[topic], stamp, bytes(data))
        finally:
            db.close()


def open_bag(source: Union[str, Path, Recording, Bag]) -> Bag:
    return source if isinstance(source, Bag) else Bag(source)
