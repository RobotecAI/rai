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

import cv2
import numpy as np
import pytest
from rai.communication.ros2.images import (
    ENCODINGS,
    compressed_imgmsg_to_numpy,
    imgmsg_to_numpy,
    numpy_to_imgmsg,
)
from sensor_msgs.msg import CompressedImage, Image


def _random_image(encoding: str, h: int = 4, w: int = 5) -> np.ndarray:
    dtype, channels = ENCODINGS[encoding]
    shape = (h, w) if channels == 1 else (h, w, channels)
    rng = np.random.default_rng(0)
    if dtype is np.float32:
        return rng.random(shape, dtype=np.float32)
    return rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)


@pytest.mark.parametrize("encoding", sorted(ENCODINGS))
def test_round_trip(encoding):
    image = _random_image(encoding)
    msg = numpy_to_imgmsg(image, encoding)
    assert (msg.height, msg.width, msg.encoding) == (4, 5, encoding)
    assert msg.step == image.shape[1] * image.itemsize * ENCODINGS[encoding][1]
    out = imgmsg_to_numpy(msg)
    assert out.dtype == image.dtype
    np.testing.assert_array_equal(out, image)
    out[...] = 0  # the result is a writable copy


def test_row_padding_and_big_endian():
    image = _random_image("mono16")
    padded = np.zeros((4, 8), dtype=">u2")
    padded[:, :5] = image
    msg = Image(
        height=4,
        width=5,
        encoding="mono16",
        is_bigendian=True,
        step=16,
        data=padded.tobytes(),
    )
    out = imgmsg_to_numpy(msg)
    assert out.dtype == np.uint16 and out.dtype.isnative
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize(
    "src,dst,code",
    [
        ("rgb8", "bgr8", cv2.COLOR_RGB2BGR),
        ("bgr8", "rgb8", cv2.COLOR_BGR2RGB),
        ("bgra8", "rgb8", cv2.COLOR_BGRA2RGB),
        ("rgba8", "bgr8", cv2.COLOR_RGBA2BGR),
        ("mono8", "rgb8", cv2.COLOR_GRAY2RGB),
        ("rgb8", "mono8", cv2.COLOR_RGB2GRAY),
    ],
)
def test_color_conversion(src, dst, code):
    image = _random_image(src)
    out = imgmsg_to_numpy(numpy_to_imgmsg(image, src), dst)
    np.testing.assert_array_equal(out, cv2.cvtColor(image, code))


def test_unsupported():
    with pytest.raises(ValueError):
        imgmsg_to_numpy(Image(encoding="yuv422", height=1, width=1, step=2, data=b"00"))
    with pytest.raises(ValueError):
        imgmsg_to_numpy(numpy_to_imgmsg(_random_image("mono16"), "mono16"), "rgb8")
    with pytest.raises(ValueError):
        numpy_to_imgmsg(_random_image("rgb8"), "mono8")


def test_compressed():
    bgr = _random_image("bgr8", 8, 8)
    msg = CompressedImage(format="png", data=cv2.imencode(".png", bgr)[1].tobytes())
    np.testing.assert_array_equal(compressed_imgmsg_to_numpy(msg), bgr)
    np.testing.assert_array_equal(
        compressed_imgmsg_to_numpy(msg, "rgb8"), bgr[..., ::-1]
    )
    gray = _random_image("mono8", 8, 8)
    msg = CompressedImage(format="png", data=cv2.imencode(".png", gray)[1].tobytes())
    np.testing.assert_array_equal(compressed_imgmsg_to_numpy(msg), gray)
    with pytest.raises(ValueError):
        compressed_imgmsg_to_numpy(CompressedImage(format="png", data=b"garbage"))
