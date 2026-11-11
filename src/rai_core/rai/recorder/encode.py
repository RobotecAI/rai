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

"""JSON for recorded content, with media moved out of line.

Every base64 image or WAV string found in a value (content blocks, data URLs,
tool artifacts, RAI ``Image`` and ``Audio``) is stored once and replaced in
place by a token, ``rai-image:<ref>`` or ``rai-audio:<ref>``, so a JPEG seen
in fifty prompts is written once. :func:`restore` puts the base64 back.
"""

import base64
import dataclasses
import enum
import hashlib
import re
from collections.abc import Callable, Mapping
from typing import Any

IMAGE_TOKEN = "rai-image:"
AUDIO_TOKEN = "rai-audio:"
_TOKEN = re.compile(r"rai-(image|audio):([0-9a-f]{20})")
_MIN_MEDIA = 128  # shorter base64 strings are left in place

# (kind, format) of base64 media by their first characters
_MAGIC = (
    ("/9j/", "image", "jpeg"),
    ("iVBORw0KGgo", "image", "png"),
    ("R0lGOD", "image", "gif"),
    ("UklGR", None, None),  # RIFF: WAV audio or WebP image
)

Media = Callable[[str, str, str], str]
"""``media(kind, format, base64) -> ref`` stores media and returns its ref."""


def ref_of(data: str) -> str:
    return hashlib.blake2b(data.encode("ascii"), digest_size=10).hexdigest()


def _media_kind(s: str) -> tuple[str, str] | None:
    if len(s) < _MIN_MEDIA:
        return None
    for prefix, kind, fmt in _MAGIC:
        if s.startswith(prefix):
            if kind is not None and fmt is not None:
                return kind, fmt
            head = base64.b64decode(s[:16])
            if head[8:12] == b"WAVE":
                return "audio", "wav"
            if head[8:12] == b"WEBP":
                return "image", "webp"
    return None


def _string(s: str, media: Media) -> str:
    if s.startswith("data:") and ";base64," in s[:64]:
        head, data = s.split(",", 1)
        if (found := _media_kind(data)) is not None:
            return f"{head},{_token(found, data, media)}"
        return s
    if (found := _media_kind(s)) is not None:
        return _token(found, s, media)
    return s


def _token(found: tuple[str, str], data: str, media: Media) -> str:
    kind, fmt = found
    return f"rai-{kind}:{media(kind, fmt, data)}"


def prepare(value: Any, media: Media) -> Any:
    """A JSON-ready copy of value, media replaced by tokens."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _string(value, media)
    if isinstance(value, Mapping):
        return {str(k): prepare(v, media) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [prepare(v, media) for v in value]
    if isinstance(value, enum.Enum):
        return prepare(value.value, media)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    return _object(value, media)


def _object(value: Any, media: Media) -> Any:
    from langchain_core.messages import BaseMessage, message_to_dict

    from rai.content import Audio, Image, Text

    if isinstance(value, BaseMessage):
        return prepare(message_to_dict(value), media)
    if isinstance(value, Image):
        return _token(("image", "jpeg"), value.encode(), media)
    if isinstance(value, Audio):
        wav = base64.b64encode(value.to_wav()).decode("ascii")
        return _token(("audio", "wav"), wav, media)
    if isinstance(value, Text):
        return value.text
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return prepare(
            {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}, media
        )
    if (model_dump := getattr(value, "model_dump", None)) is not None:  # pydantic
        return prepare(model_dump(mode="json"), media)
    if (tolist := getattr(value, "tolist", None)) is not None:  # numpy
        return prepare(tolist(), media)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    return repr(value)


def restore(value: Any, media: Callable[[str, str], str]) -> Any:
    """Undo :func:`prepare`'s media tokens: ``media(kind, ref) -> base64``."""
    if isinstance(value, str):
        return _TOKEN.sub(lambda m: media(m.group(1), m.group(2)), value)
    if isinstance(value, dict):
        return {k: restore(v, media) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v, media) for v in value]
    return value
