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

"""Reading flight recordings back (tests, replay, analysis)."""

import base64
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from . import schemas as s
from .encode import restore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordedMessage:
    topic: str
    schema: str  # "rai.ToolCall", "geometry_msgs/msg/Twist", ...
    encoding: str  # "json" or "cdr"
    log_time: int  # ns on the recording's clock (log_time == publish_time)
    publish_time: int
    sequence: int
    data: bytes
    value: Any  # the decoded message: a dict for JSON, a ROS message for CDR


class Recording:
    """An MCAP file written by FlightRecorder (or any MCAP with JSON and ROS 2
    channels). Reads linearly, so a file whose writer died (no summary, a cut
    last chunk) still gives everything up to the last flush."""

    def __init__(self, path: Union[str, Path]) -> None:
        try:
            from mcap.stream_reader import StreamReader
        except ImportError as e:
            raise ImportError(
                "rai.recorder needs the 'recorder' extra:"
                " pip install 'rai_core[recorder]'"
            ) from e
        self.path = Path(path)
        self._stream_reader = StreamReader
        self.schemas: dict[int, Any] = {}
        self.channels: dict[int, Any] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.attachments: dict[str, bytes] = {}
        self.complete = False  # the file ends with a footer
        self._images: Optional[dict[str, str]] = None
        self._decoders: dict[int, Callable[[bytes], Any]] = {}
        for _ in self._records():  # schemas, channels, metadata, attachments
            pass

    def _records(self) -> Iterator[Any]:
        from mcap.records import (
            Attachment,
            Channel,
            Footer,
            Message,
            Metadata,
            Schema,
        )

        with self.path.open("rb") as f:
            reader = self._stream_reader(f, skip_magic=False)
            try:
                for record in reader.records:
                    if isinstance(record, Message):
                        yield record
                    elif isinstance(record, Schema):
                        self.schemas[record.id] = record
                    elif isinstance(record, Channel):
                        self.channels[record.id] = record
                    elif isinstance(record, Metadata):
                        self.metadata[record.name] = dict(record.metadata)
                    elif isinstance(record, Attachment):
                        self.attachments[record.name] = record.data
                    elif isinstance(record, Footer):
                        self.complete = True
            except Exception as e:  # a cut file: keep what was read
                logger.debug("%s ends early: %r", self.path, e)

    @property
    def topics(self) -> dict[str, str]:
        """topic -> schema name."""
        return {
            c.topic: self.schemas[c.schema_id].name if c.schema_id else ""
            for c in self.channels.values()
        }

    def messages(
        self, topics: Optional[Iterable[str]] = None, *, decode: bool = True
    ) -> list[RecordedMessage]:
        """Messages (of topics, or all) in log_time order."""
        wanted = None if topics is None else set(topics)
        out = []
        for record in self._records():
            channel = self.channels[record.channel_id]
            if wanted is not None and channel.topic not in wanted:
                continue
            schema = self.schemas.get(channel.schema_id)
            out.append(
                RecordedMessage(
                    channel.topic,
                    schema.name if schema else "",
                    channel.message_encoding,
                    record.log_time,
                    record.publish_time,
                    record.sequence,
                    record.data,
                    self._decode(channel, record.data) if decode else None,
                )
            )
        out.sort(key=lambda m: m.log_time)
        return out

    def values(self, topic: str) -> list[Any]:
        return [m.value for m in self.messages([topic])]

    def _decode(self, channel: Any, data: bytes) -> Any:
        if channel.message_encoding == "json":
            return json.loads(data)
        decoder = self._decoders.get(channel.id)
        if decoder is None:
            from mcap_ros2.decoder import DecoderFactory

            schema = self.schemas.get(channel.schema_id)
            decoder = DecoderFactory().decoder_for(channel.message_encoding, schema)
            if decoder is None:
                return data
            self._decoders[channel.id] = decoder
        return decoder(data)

    # ------------------------------------------------------------ RAI content

    def media(self, kind: str, ref: str) -> str:
        """The base64 data of a ``rai-image:<ref>`` or ``rai-audio:<ref>``."""
        if kind == "audio":
            for name, data in self.attachments.items():
                if name.startswith(f"{ref}."):
                    return base64.b64encode(data).decode("ascii")
            raise KeyError(ref)
        if self._images is None:
            self._images = {}
            image_topics = [
                t for t, schema in self.topics.items()
                if schema == "foxglove.CompressedImage"
            ]  # fmt: skip
            for message in self.messages(image_topics):
                self._images[message.value["frame_id"]] = message.value["data"]
        return self._images[ref]

    def restore(self, value: Any) -> Any:
        """value with every media token replaced by its base64 data."""
        return restore(value, self.media)

    def prompts(self, agent: Optional[str] = None) -> list[dict[str, Any]]:
        """Every model request (of one agent) with its full prompt: the
        LangChain message dicts (``messages_from_dict`` turns them back into
        messages), media restored. Each item is the /rai/agent/llm request
        with ``messages`` replaced by the full list."""
        last: dict[str, list[Any]] = {}
        out = []
        for request in self.values(s.LLM):
            if request["phase"] != "request":
                continue
            if agent is not None and request["agent"] != agent:
                continue
            previous = last.get(request["agent"], [])
            full = previous[: request["prefix"]] + request["messages"]
            last[request["agent"]] = full
            out.append(request | {"messages": self.restore(full)})
        return out
