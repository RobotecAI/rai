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

"""sensor_msgs Image <-> numpy conversion, replacing cv_bridge."""

import sys
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Header

from rai import content
from rai.content import ENCODINGS, convert_encoding


def imgmsg_to_numpy(msg: Image, encoding: str = "passthrough") -> NDArray:
    """Return a copy of the image data as an (H, W) or (H, W, C) array."""
    if msg.encoding not in ENCODINGS:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    dtype, channels = ENCODINGS[msg.encoding]
    dt = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    row_bytes = msg.width * channels * dt.itemsize
    image = data[:, :row_bytes].copy().view(dt).astype(dt.newbyteorder("="), copy=False)
    shape = (
        (msg.height, msg.width) if channels == 1 else (msg.height, msg.width, channels)
    )
    return convert_encoding(image.reshape(shape), msg.encoding, encoding)


def numpy_to_imgmsg(
    image: NDArray, encoding: str, header: Optional[Header] = None
) -> Image:
    """Build an Image message from an (H, W) or (H, W, C) array in the given encoding."""
    if encoding not in ENCODINGS:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    dtype, channels = ENCODINGS[encoding]
    image_channels = 1 if image.ndim == 2 else image.shape[2]
    if image.dtype != dtype or image_channels != channels:
        raise ValueError(
            f"Encoding {encoding} needs {channels} channel(s) of {np.dtype(dtype)}, "
            f"got {image_channels} of {image.dtype}"
        )
    image = np.ascontiguousarray(image)
    msg = Image(
        height=image.shape[0],
        width=image.shape[1],
        encoding=encoding,
        is_bigendian=sys.byteorder == "big",
        step=image.shape[1] * channels * image.itemsize,
        data=image.tobytes(),
    )
    if header is not None:
        msg.header = header
    return msg


def compressed_imgmsg_to_numpy(
    msg: CompressedImage, encoding: str = "passthrough"
) -> NDArray:
    """Decode a CompressedImage. Color images decode as bgr8 or bgra8, like cv2."""
    buffer = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode compressed image ({msg.format})")
    if image.ndim == 2:
        src = "mono8" if image.dtype == np.uint8 else "mono16"
    else:
        src = "bgr8" if image.shape[2] == 3 else "bgra8"
    return convert_encoding(image, src, encoding)


def encode_image(
    image: NDArray | Image | CompressedImage | content.Image,
    *,
    max_side: int = 1024,
    format: str = "jpeg",
    quality: int = 85,
) -> str:
    """Base64 JPEG (or "png") of an image for the LLM, see
    :meth:`rai.content.Image.encode`. Arrays are RGB(A) or mono."""
    if isinstance(image, CompressedImage):
        image = compressed_imgmsg_to_numpy(image, "rgb8")
    if isinstance(image, Image):
        image = content.Image(imgmsg_to_numpy(image), image.encoding)
    elif not isinstance(image, content.Image):
        mono = {np.uint8: "mono8", np.uint16: "mono16", np.float32: "32FC1"}
        encoding = "rgba8" if image.ndim == 3 and image.shape[2] == 4 else "rgb8"
        image = content.Image(
            image, mono[image.dtype.type] if image.ndim == 2 else encoding
        )
    return image.encode(max_side=max_side, format=format, quality=quality)
