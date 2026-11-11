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

"""RAI's content types: :class:`Text`, :class:`Image` and :class:`Audio`.

They hold numpy arrays and know nothing about LLM frameworks or ROS;
``rai.messages`` converts them to LangChain content blocks at the LLM boundary.
"""

import base64
import io
import wave
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

# Image encodings (ROS names) -> (dtype, channels)
ENCODINGS: dict[str, tuple[type[np.generic], int]] = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "8UC3": (np.uint8, 3),
    "8UC4": (np.uint8, 4),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}
_CANONICAL = {"8UC1": "mono8", "8UC3": "bgr8", "8UC4": "bgra8", "16UC1": "mono16"}
# Encodings that cv2.cvtColor converts into each other.
_COLOR_NAMES = {
    "rgb8": "RGB",
    "bgr8": "BGR",
    "rgba8": "RGBA",
    "bgra8": "BGRA",
    "mono8": "GRAY",
}


def convert_encoding(image: NDArray[Any], src: str, dst: str) -> NDArray[Any]:
    """Convert a numpy image between encodings. "passthrough" keeps it as is."""
    src, dst = _CANONICAL.get(src, src), _CANONICAL.get(dst, dst)
    if dst in ("passthrough", src):
        return image
    if src not in _COLOR_NAMES or dst not in _COLOR_NAMES:
        raise ValueError(f"Cannot convert image from {src} to {dst}")
    import cv2

    code = getattr(cv2, f"COLOR_{_COLOR_NAMES[src]}2{_COLOR_NAMES[dst]}")
    return cv2.cvtColor(image, code)


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True, eq=False)
class Image:
    """An (H, W) or (H, W, C) array and its encoding ("rgb8", "bgr8", "mono8",
    "mono16", "32FC1", ...; the ROS names). The array may be a view of a
    received message: copy it to change it."""

    data: NDArray[Any]
    encoding: str = "rgb8"

    def __post_init__(self) -> None:
        if self.encoding not in ENCODINGS:
            raise ValueError(f"Unsupported image encoding: {self.encoding}")
        dtype, channels = ENCODINGS[self.encoding]
        got = 1 if self.data.ndim == 2 else self.data.shape[2]
        if self.data.ndim not in (2, 3) or self.data.dtype != dtype or got != channels:
            raise ValueError(
                f"{self.encoding} needs (H, W{'' if channels == 1 else f', {channels}'})"
                f" {np.dtype(dtype)}, got {self.data.shape} {self.data.dtype}"
            )

    def to(self, encoding: str) -> "Image":
        """The image in another color encoding (rgb8, bgr8, rgba8, bgra8, mono8)."""
        return Image(convert_encoding(self.data, self.encoding, encoding), encoding)

    def encode(
        self, *, max_side: int = 1024, format: str = "jpeg", quality: int = 85
    ) -> str:
        """Base64 JPEG (or "png") for the LLM, the one encode path. Images larger
        than max_side are downscaled; depth images are scaled to 8 bits."""
        import cv2

        image = self.data
        if image.dtype != np.uint8:  # depth: show the range that is there
            finite = np.nan_to_num(image.astype(np.float32), posinf=0.0, neginf=0.0)
            image = cv2.normalize(  # pyright: ignore[reportCallIssue]
                finite,
                None,  # pyright: ignore[reportArgumentType]  # allocates dst; the stubs require a Mat
                0,
                255,
                cv2.NORM_MINMAX,
                cv2.CV_8U,
            )
        elif image.ndim == 3:
            src = _CANONICAL.get(self.encoding, self.encoding)
            image = convert_encoding(image, src, "bgra8" if src[-2] == "a" else "bgr8")
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        if format == "png":
            ok, data = cv2.imencode(".png", image)
        else:
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError(f"Cannot encode the image as {format}")
        return base64.b64encode(data.tobytes()).decode("ascii")

    @classmethod
    def decode(cls, data: Union[str, bytes]) -> "Image":
        """An Image (rgb8, rgba8 or mono8) from encoded bytes or their base64."""
        import cv2

        if isinstance(data, str):
            data = base64.b64decode(data)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Cannot decode the image")
        if image.ndim == 2:
            return cls(image, "mono8" if image.dtype == np.uint8 else "mono16")
        return cls(image, "bgra8" if image.shape[2] == 4 else "bgr8").to(
            "rgba8" if image.shape[2] == 4 else "rgb8"
        )


@dataclass(frozen=True, eq=False)
class Audio:
    """Samples shaped (n, channels), int16 or float32 in [-1, 1], and their
    sample rate. A 1-D array is taken as mono."""

    samples: NDArray[Any]
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.dtype not in (np.int16, np.float32):
            raise ValueError(f"Audio is int16 or float32, not {self.samples.dtype}")
        if self.samples.ndim == 1:
            object.__setattr__(self, "samples", self.samples.reshape(-1, 1))
        elif self.samples.ndim != 2:
            raise ValueError(f"Audio is shaped (n, channels), not {self.samples.shape}")

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def as_int16(self) -> "Audio":
        if self.samples.dtype == np.int16:
            return self
        scaled = np.clip(self.samples * 32768.0, -32768, 32767)
        return Audio(scaled.astype(np.int16), self.sample_rate)

    def as_float32(self) -> "Audio":
        if self.samples.dtype == np.float32:
            return self
        return Audio(self.samples.astype(np.float32) / 32768.0, self.sample_rate)

    def to_wav(self) -> bytes:
        """16-bit PCM WAV."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as f:
            f.setnchannels(self.channels)
            f.setsampwidth(2)
            f.setframerate(self.sample_rate)
            f.writeframes(self.as_int16().samples.astype("<i2").tobytes())
        return buffer.getvalue()

    @classmethod
    def from_wav(cls, data: bytes) -> "Audio":
        """An int16 Audio from 16-bit PCM WAV bytes."""
        with wave.open(io.BytesIO(data), "rb") as f:
            if f.getsampwidth() != 2:
                raise ValueError("Only 16-bit PCM WAV is supported")
            frames = f.readframes(f.getnframes())
            samples = np.frombuffer(frames, "<i2").astype(np.int16)
            return cls(samples.reshape(-1, f.getnchannels()), f.getframerate())


Content = Union[Text, Image, Audio]

__all__ = ["ENCODINGS", "Audio", "Content", "Image", "Text", "convert_encoding"]
