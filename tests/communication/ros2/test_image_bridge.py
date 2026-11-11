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

import cv2
import numpy as np
import pytest
from rai.communication.ros2.image_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image


@pytest.fixture
def bridge() -> CvBridge:
    return CvBridge()


def _random(shape, dtype, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return rng.random(shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, shape, dtype=dtype, endpoint=True)


@pytest.mark.parametrize(
    "encoding, shape, dtype",
    [
        ("rgb8", (4, 6, 3), np.uint8),
        ("bgra8", (4, 6, 4), np.uint8),
        ("mono8", (4, 6), np.uint8),
        ("mono16", (4, 6), np.uint16),
        ("16UC1", (4, 6), np.uint16),
        ("32FC1", (4, 6), np.float32),
        ("64FC3", (4, 6, 3), np.float64),
        ("8SC1", (4, 6), np.int8),
    ],
)
def test_round_trip_passthrough(bridge, encoding, shape, dtype):
    image = _random(shape, dtype)
    msg = bridge.cv2_to_imgmsg(image, encoding)
    assert (msg.height, msg.width, msg.encoding) == (4, 6, encoding)
    assert msg.step == 6 * image.itemsize * (shape[2] if len(shape) == 3 else 1)
    out = bridge.imgmsg_to_cv2(msg)
    assert out.dtype == image.dtype
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize(
    "shape, dtype, expected",
    [
        ((2, 3, 3), np.uint8, "8UC3"),
        ((2, 3), np.uint16, "16UC1"),
        ((2, 3), np.float32, "32FC1"),
        ((2, 3, 4), np.int16, "16SC4"),
    ],
)
def test_cv2_to_imgmsg_passthrough_uses_opencv_type(bridge, shape, dtype, expected):
    assert bridge.cv2_to_imgmsg(np.zeros(shape, dtype)).encoding == expected


def test_cv2_to_imgmsg_keeps_header(bridge):
    msg = Image()
    msg.header.frame_id = "camera"
    out = bridge.cv2_to_imgmsg(np.zeros((2, 2), np.uint8), header=msg.header)
    assert out.header.frame_id == "camera"


@pytest.mark.parametrize(
    "image, encoding",
    [
        (np.zeros((4, 4), np.uint8), "rgb8"),  # wrong channel count
        (np.zeros((4, 4, 3), np.uint8), "mono8"),
        (np.zeros((4, 4), np.uint8), "16UC1"),  # wrong depth
        (np.zeros((4, 4), bool), "passthrough"),  # no OpenCV type for bool
        (np.zeros((4, 4), np.int64), "passthrough"),
    ],
)
def test_cv2_to_imgmsg_rejects_incompatible_images(bridge, image, encoding):
    with pytest.raises(CvBridgeError):
        bridge.cv2_to_imgmsg(image, encoding)


def test_cv2_to_imgmsg_rejects_non_arrays(bridge):
    with pytest.raises(TypeError):
        bridge.cv2_to_imgmsg([[1, 2], [3, 4]])


def test_color_conversions(bridge):
    rgb = _random((3, 5, 3), np.uint8)
    msg = bridge.cv2_to_imgmsg(rgb, "rgb8")
    np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "bgr8"), rgb[..., ::-1])
    np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg, "rgb8"), rgb)

    rgba = _random((3, 5, 4), np.uint8)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(rgba, "rgba8"), "rgb8")
    np.testing.assert_array_equal(out, rgba[..., :3])

    mono = _random((3, 5), np.uint8)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(mono, "mono8"), "rgb8")
    np.testing.assert_array_equal(out, np.repeat(mono[..., None], 3, axis=2))


def test_bit_depth_is_rescaled_between_8_and_16_bits(bridge):
    mono16 = np.array([[0, 257, 25700, 65535]], np.uint16)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(mono16, "mono16"), "mono8")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 1, 100, 255]])

    mono8 = np.array([[0, 1, 255]], np.uint8)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(mono8, "mono8"), "mono16")
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, [[0, 257, 65535]])

    rgb16 = np.full((2, 2, 3), 65535, np.uint16)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(rgb16, "rgb16"), "bgr8")
    assert out.dtype == np.uint8 and (out == 255).all()


def test_invalid_float_pixels_follow_opencv_saturate_cast(bridge):
    # NaN, +-inf and values beyond +-2**31 become INT_MIN before saturation, as in
    # OpenCV/cv_bridge, so invalid depth pixels do not turn into the maximum value.
    depth = np.array([[np.inf, 1e20, 12.5, np.nan, -np.inf, 3.5]], np.float32)
    msg = bridge.cv2_to_imgmsg(depth, "32FC1")
    np.testing.assert_array_equal(
        bridge.imgmsg_to_cv2(msg, "16UC1"), [[0, 0, 12, 0, 0, 4]]
    )
    np.testing.assert_array_equal(
        bridge.imgmsg_to_cv2(msg, "8SC1"), [[-128, -128, 12, -128, -128, 4]]
    )
    wide = np.array([[2147483647.0, 2147483648.0, -2147483649.0]], np.float64)
    out = bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(wide, "64FC1"), "32SC1")
    np.testing.assert_array_equal(out, [[2147483647, -2147483648, -2147483648]])


def test_bayer_follows_ros_pattern_names(bridge):
    raw = _random((4, 6), np.uint8)
    msg = bridge.cv2_to_imgmsg(raw, "bayer_rggb8")
    np.testing.assert_array_equal(
        bridge.imgmsg_to_cv2(msg, "rgb8"), cv2.cvtColor(raw, cv2.COLOR_BayerBG2RGB)
    )


@pytest.mark.parametrize(
    "src, dst",
    [
        ("32FC1", "rgb8"),  # data without colour information cannot become colour
        ("rgb8", "8UC1"),  # colour to a generic type needs the same channel count
        ("16UC1", "32FC3"),
        ("bayer_rggb8", "rgba8"),  # not a conversion cv_bridge offers
        ("8U", "8UC1"),  # bare depths are not accepted in conversions
    ],
)
def test_unsupported_conversions_raise(bridge, src, dst):
    depth, channels = bridge.encoding_to_dtype_with_channels(src)
    shape = (4, 6) if channels == 1 else (4, 6, channels)
    msg = bridge.cv2_to_imgmsg(np.zeros(shape, depth), src)
    with pytest.raises(CvBridgeError):
        bridge.imgmsg_to_cv2(msg, dst)


def test_unknown_encoding_raises(bridge):
    msg = Image(height=1, width=1, encoding="not-an-encoding", step=1, data=[0])
    with pytest.raises(CvBridgeError):
        bridge.imgmsg_to_cv2(msg)


def test_padded_rows_are_cropped(bridge):
    image = _random((3, 5, 3), np.uint8)
    padding = np.full((3, 7), 0xAB, np.uint8)
    msg = Image(height=3, width=5, encoding="rgb8", step=5 * 3 + 7)
    msg.data = np.hstack([image.reshape(3, -1), padding]).tobytes()
    np.testing.assert_array_equal(bridge.imgmsg_to_cv2(msg), image)


def test_step_shorter_than_a_row_raises(bridge):
    msg = Image(height=2, width=4, encoding="rgb8", step=11)  # a row needs 12 bytes
    msg.data = bytes(24)
    with pytest.raises(CvBridgeError):
        bridge.imgmsg_to_cv2(msg)


def test_data_shorter_than_the_image_raises(bridge):
    msg = Image(height=4, width=4, encoding="mono8", step=4)
    msg.data = bytes(10)
    with pytest.raises(CvBridgeError):
        bridge.imgmsg_to_cv2(msg)


@pytest.mark.parametrize(
    "encoding, dtype", [("16UC1", np.uint16), ("32FC1", np.float32)]
)
def test_big_endian_data_is_returned_in_native_order(bridge, encoding, dtype):
    image = _random((3, 4), dtype)
    msg = Image(height=3, width=4, encoding=encoding, is_bigendian=1)
    msg.data = image.astype(np.dtype(dtype).newbyteorder(">")).tobytes()
    msg.step = len(msg.data) // 3
    out = bridge.imgmsg_to_cv2(msg)
    assert out.dtype.isnative
    np.testing.assert_array_equal(out, image)


def test_compressed_round_trip(bridge):
    bgr = _random((6, 8, 3), np.uint8)
    msg = bridge.cv2_to_compressed_imgmsg(bgr, "png")
    assert isinstance(msg, CompressedImage) and msg.format == "png"
    np.testing.assert_array_equal(bridge.compressed_imgmsg_to_cv2(msg), bgr)
    # OpenCV decodes colour as BGR, so other encodings are converted from bgr8.
    np.testing.assert_array_equal(
        bridge.compressed_imgmsg_to_cv2(msg, "rgb8"), bgr[..., ::-1]
    )


def test_undecodable_compressed_image(bridge):
    msg = CompressedImage(format="png")
    msg.data = b"not an image"
    assert bridge.compressed_imgmsg_to_cv2(msg) is None
    with pytest.raises(CvBridgeError):
        bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
