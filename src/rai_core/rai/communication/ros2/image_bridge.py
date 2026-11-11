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

"""Conversions between ``sensor_msgs`` images and numpy arrays without ``cv_bridge``.

``CvBridge`` here is a drop-in replacement for the ``cv_bridge.CvBridge`` methods RAI
uses. It is pure numpy + OpenCV, so it works under both numpy 1.x and numpy 2.x.

ROS 2 Jazzy's apt ``cv_bridge`` cannot be used under numpy 2: its compiled helper
(``cv_bridge_boost``) is built against Ubuntu 24.04's numpy 1.26, ``import cv_bridge``
silently swallows the resulting import failure, and the first conversion then crashes
the interpreter (segmentation fault). Big-endian images additionally hit
``ndarray.newbyteorder()``, which numpy 2 removed.

Behaviour follows ``cv_bridge`` (``cv_bridge/core.py`` and the C++ ``cvtColor`` it
wraps): the same encodings, the same colour conversions, the same 8 <-> 16 bit
scaling and the same refusals. Failures raise :class:`CvBridgeError`.

One deliberate difference: rows are read with the message ``step`` as the exact row
stride. cv_bridge assumes row padding is a whole number of pixels and scrambles the
image when it is not (e.g. a 5-pixel ``rgb8`` row padded to 16 bytes); here any
padding works, and a ``step`` shorter than one row raises instead of cropping.
"""

import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

__all__ = ["CvBridge", "CvBridgeError"]


class CvBridgeError(TypeError):
    """Raised when an image conversion is not possible (mirrors ``cv_bridge.CvBridgeError``)."""


# OpenCV depth name <-> numpy dtype name.
_CV_DEPTH_TO_NUMPY = {
    "8U": "uint8",
    "8S": "int8",
    "16U": "uint16",
    "16S": "int16",
    "32S": "int32",
    "32F": "float32",
    "64F": "float64",
}
_NUMPY_TO_CV_DEPTH = {v: k for k, v in _CV_DEPTH_TO_NUMPY.items()}

# Named encodings from sensor_msgs/image_encodings.hpp: name -> (OpenCV depth, channels).
_NAMED_ENCODINGS: Dict[str, Tuple[str, int]] = {
    "rgb8": ("8U", 3),
    "rgba8": ("8U", 4),
    "rgb16": ("16U", 3),
    "rgba16": ("16U", 4),
    "bgr8": ("8U", 3),
    "bgra8": ("8U", 4),
    "bgr16": ("16U", 3),
    "bgra16": ("16U", 4),
    "mono8": ("8U", 1),
    "mono16": ("16U", 1),
    "yuv422": ("8U", 2),
    "yuv422_yuy2": ("8U", 2),
}
for _pattern in ("rggb", "bggr", "gbrg", "grbg"):
    _NAMED_ENCODINGS[f"bayer_{_pattern}8"] = ("8U", 1)
    _NAMED_ENCODINGS[f"bayer_{_pattern}16"] = ("16U", 1)

_GENERIC_ENCODING = re.compile(r"(8U|8S|16U|16S|32S|32F|64F)C([0-9]+)")
# A bare depth (e.g. "8U") is a valid single-channel message encoding, but cv_bridge
# refuses it in colour conversions (its channel lookup only knows "8UC<n>").
_GENERIC_DEPTH_ONLY = re.compile(r"(8U|8S|16U|16S|32S|32F|64F)")

# Colour families used to look up conversions (cv_bridge's ``getEncoding``).
_FAMILY = {
    "mono8": "GRAY",
    "mono16": "GRAY",
    "rgb8": "RGB",
    "rgb16": "RGB",
    "bgr8": "BGR",
    "bgr16": "BGR",
    "rgba8": "RGBA",
    "rgba16": "RGBA",
    "bgra8": "BGRA",
    "bgra16": "BGRA",
    "yuv422": "YUV422",
    "yuv422_yuy2": "YUV422_YUY2",
}
for _pattern in ("rggb", "bggr", "gbrg", "grbg"):
    _FAMILY[f"bayer_{_pattern}8"] = f"BAYER_{_pattern.upper()}"
    _FAMILY[f"bayer_{_pattern}16"] = f"BAYER_{_pattern.upper()}"

# Marker for "same channel layout, convert the bit depth only" (cv_bridge's SAME_FORMAT).
_SAME_FORMAT = -1


def _conversion_codes() -> Dict[Tuple[str, str], List[int]]:
    """Colour conversions cv_bridge supports, as OpenCV ``cvtColor`` codes."""
    codes: Dict[Tuple[str, str], List[int]] = {}
    for family in ("GRAY", "RGB", "BGR", "RGBA", "BGRA"):
        codes[(family, family)] = [_SAME_FORMAT]
    pairs = {
        ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
        ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
        ("GRAY", "RGBA"): cv2.COLOR_GRAY2RGBA,
        ("GRAY", "BGRA"): cv2.COLOR_GRAY2BGRA,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("RGB", "RGBA"): cv2.COLOR_RGB2RGBA,
        ("RGB", "BGRA"): cv2.COLOR_RGB2BGRA,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("BGR", "RGBA"): cv2.COLOR_BGR2RGBA,
        ("BGR", "BGRA"): cv2.COLOR_BGR2BGRA,
        ("RGBA", "GRAY"): cv2.COLOR_RGBA2GRAY,
        ("RGBA", "RGB"): cv2.COLOR_RGBA2RGB,
        ("RGBA", "BGR"): cv2.COLOR_RGBA2BGR,
        ("RGBA", "BGRA"): cv2.COLOR_RGBA2BGRA,
        ("BGRA", "GRAY"): cv2.COLOR_BGRA2GRAY,
        ("BGRA", "RGB"): cv2.COLOR_BGRA2RGB,
        ("BGRA", "BGR"): cv2.COLOR_BGRA2BGR,
        ("BGRA", "RGBA"): cv2.COLOR_BGRA2RGBA,
        ("YUV422", "GRAY"): cv2.COLOR_YUV2GRAY_UYVY,
        ("YUV422", "RGB"): cv2.COLOR_YUV2RGB_UYVY,
        ("YUV422", "BGR"): cv2.COLOR_YUV2BGR_UYVY,
        ("YUV422", "RGBA"): cv2.COLOR_YUV2RGBA_UYVY,
        ("YUV422", "BGRA"): cv2.COLOR_YUV2BGRA_UYVY,
        ("YUV422_YUY2", "GRAY"): cv2.COLOR_YUV2GRAY_YUY2,
        ("YUV422_YUY2", "RGB"): cv2.COLOR_YUV2RGB_YUY2,
        ("YUV422_YUY2", "BGR"): cv2.COLOR_YUV2BGR_YUY2,
        ("YUV422_YUY2", "RGBA"): cv2.COLOR_YUV2RGBA_YUY2,
        ("YUV422_YUY2", "BGRA"): cv2.COLOR_YUV2BGRA_YUY2,
    }
    # ROS names Bayer patterns by the top-left 2x2 block; OpenCV by the second row.
    bayer = {
        "BAYER_RGGB": ("BayerBG", cv2.COLOR_BayerBG2GRAY),
        "BAYER_BGGR": ("BayerRG", cv2.COLOR_BayerRG2GRAY),
        "BAYER_GBRG": ("BayerGR", cv2.COLOR_BayerGR2GRAY),
        "BAYER_GRBG": ("BayerGB", cv2.COLOR_BayerGB2GRAY),
    }
    for family, (cv_name, to_gray) in bayer.items():
        pairs[(family, "GRAY")] = to_gray
        pairs[(family, "RGB")] = getattr(cv2, f"COLOR_{cv_name}2RGB")
        pairs[(family, "BGR")] = getattr(cv2, f"COLOR_{cv_name}2BGR")
    for key, code in pairs.items():
        codes[key] = [code]
    return codes


_CONVERSIONS = _conversion_codes()


def _encoding_depth_and_channels(encoding: str) -> Tuple[str, int]:
    """OpenCV depth name and channel count of a ROS image encoding (``getCvType``)."""
    if encoding in _NAMED_ENCODINGS:
        return _NAMED_ENCODINGS[encoding]
    match = _GENERIC_ENCODING.fullmatch(encoding)
    if match:
        return match.group(1), int(match.group(2))
    match = _GENERIC_DEPTH_ONLY.fullmatch(encoding)
    if match:
        return match.group(1), 1
    raise CvBridgeError(f"Unrecognized image encoding [{encoding}]")


def _is_color_format(encoding: str) -> bool:
    return encoding in _FAMILY


def _bit_depth(encoding: str) -> int:
    return (
        np.dtype(_CV_DEPTH_TO_NUMPY[_encoding_depth_and_channels(encoding)[0]]).itemsize
        * 8
    )


def _conversion_steps(src_encoding: str, dst_encoding: str) -> List[int]:
    """The conversion steps from one encoding to another (cv_bridge's ``getConversionCode``)."""
    for encoding in (src_encoding, dst_encoding):
        if _GENERIC_DEPTH_ONLY.fullmatch(encoding):
            raise CvBridgeError(f"Unknown encoding {encoding}")
    src_channels = _encoding_depth_and_channels(src_encoding)[1]
    dst_channels = _encoding_depth_and_channels(dst_encoding)[1]
    same_channels = src_channels == dst_channels
    src_is_color = _is_color_format(src_encoding)
    dst_is_color = _is_color_format(dst_encoding)

    if not src_is_color:
        if dst_is_color:
            raise CvBridgeError(
                f"[{src_encoding}] is not a color format. but [{dst_encoding}] is. "
                "The conversion does not make sense"
            )
        if not same_channels:
            raise CvBridgeError(
                f"[{src_encoding}] and [{dst_encoding}] do not have the same number of channel"
            )
        return [_SAME_FORMAT]

    if not dst_is_color:
        if not same_channels:
            raise CvBridgeError(
                f"[{src_encoding}] is a color format but [{dst_encoding}] is not so they must "
                "have the same OpenCV type, CV_8UC3, CV16UC1 ...."
            )
        return [_SAME_FORMAT]

    steps = _CONVERSIONS.get((_FAMILY[src_encoding], _FAMILY[dst_encoding]))
    if steps is None:
        raise CvBridgeError(
            f"Unsupported conversion from [{src_encoding}] to [{dst_encoding}]"
        )
    steps = list(steps)
    if _bit_depth(src_encoding) != _bit_depth(dst_encoding) and (
        _FAMILY[src_encoding] != _FAMILY[dst_encoding]
    ):
        steps.append(_SAME_FORMAT)
    return steps


def _convert_depth(
    image: np.ndarray, src_encoding: str, dst_encoding: str
) -> np.ndarray:
    """Change the bit depth like ``cv::Mat::convertTo`` does in cv_bridge.

    8 <-> 16 bit conversions are rescaled between [0, 255] and [0, 65535]. Integer
    results follow OpenCV's ``saturate_cast``: the value is rounded half-to-even to a
    32-bit integer, anything that cannot be represented that way (NaN, +-inf, beyond
    +-2**31) becomes INT_MIN, and the result is then saturated to the destination type.
    So invalid float pixels become 0 in unsigned images rather than the maximum.
    """
    dst_dtype = np.dtype(
        _CV_DEPTH_TO_NUMPY[_encoding_depth_and_channels(dst_encoding)[0]]
    )
    src_bits, dst_bits = _bit_depth(src_encoding), _bit_depth(dst_encoding)
    scale = 1.0
    if src_bits == 8 and dst_bits == 16:
        scale = 65535.0 / 255.0
    elif src_bits == 16 and dst_bits == 8:
        scale = 255.0 / 65535.0
    if image.dtype == dst_dtype and scale == 1.0:
        return image.copy()
    scaled = image.astype(np.float64) * scale
    if np.issubdtype(dst_dtype, np.integer):
        info = np.iinfo(dst_dtype)
        int32 = np.iinfo(np.int32)
        with np.errstate(invalid="ignore"):
            rounded = np.rint(scaled)
            invalid = (
                ~np.isfinite(rounded) | (rounded < int32.min) | (rounded > int32.max)
            )
        rounded[invalid] = int32.min
        scaled = np.clip(rounded, info.min, info.max)
    return scaled.astype(dst_dtype)


def _convert_color(
    image: np.ndarray, src_encoding: str, dst_encoding: str
) -> np.ndarray:
    """Convert an image between encodings (cv_bridge's ``cvtColor2``)."""
    if src_encoding == dst_encoding:
        return image.copy()
    result = image
    for step in _conversion_steps(src_encoding, dst_encoding):
        if step == _SAME_FORMAT:
            result = _convert_depth(result, src_encoding, dst_encoding)
        else:
            try:
                result = cv2.cvtColor(result, step)
            except cv2.error as e:
                raise CvBridgeError(str(e)) from e
    return result


class CvBridge:
    """Converts between ``sensor_msgs`` images and numpy (OpenCV) images.

    Same method signatures as ``cv_bridge.CvBridge``; works under numpy 1.x and 2.x.
    """

    def encoding_to_dtype_with_channels(self, encoding: str) -> Tuple[str, int]:
        """The numpy dtype name and channel count of a ROS image encoding."""
        depth, channels = _encoding_depth_and_channels(encoding)
        return _CV_DEPTH_TO_NUMPY[depth], channels

    def dtype_with_channels_to_cvtype2(self, dtype: np.dtype, n_channels: int) -> str:
        """The OpenCV type name (e.g. ``8UC3``) of a numpy dtype and channel count."""
        try:
            return f"{_NUMPY_TO_CV_DEPTH[np.dtype(dtype).name]}C{n_channels}"
        except KeyError as e:
            raise CvBridgeError(
                f"Unsupported image dtype [{np.dtype(dtype).name}]"
            ) from e

    def imgmsg_to_cv2(
        self, img_msg: Image, desired_encoding: str = "passthrough"
    ) -> np.ndarray:
        """Convert a ``sensor_msgs/Image`` to a numpy array.

        With ``desired_encoding="passthrough"`` the array keeps the message's layout;
        otherwise it is converted to the given ROS image encoding. Single-channel
        images are 2-D (height, width).
        """
        dtype_name, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
        dtype = np.dtype(dtype_name).newbyteorder(">" if img_msg.is_bigendian else "<")
        data = img_msg.data
        buffer = bytes(data) if isinstance(data, list) else data
        pixel_bytes = dtype.itemsize * n_channels
        if img_msg.step < img_msg.width * pixel_bytes:
            raise CvBridgeError(
                f"Image step {img_msg.step} is shorter than a row of {img_msg.width} "
                f"{img_msg.encoding} pixels ({img_msg.width * pixel_bytes} bytes)"
            )
        # Rows are read with ``step`` as the exact row stride, so padding of any size
        # works. (cv_bridge assumes the padding is a whole number of pixels and returns a
        # scrambled image otherwise.)
        if n_channels == 1:
            shape: Tuple[int, ...] = (img_msg.height, img_msg.width)
            strides: Tuple[int, ...] = (img_msg.step, dtype.itemsize)
        else:
            shape = (img_msg.height, img_msg.width, n_channels)
            strides = (img_msg.step, pixel_bytes, dtype.itemsize)
        try:
            image = np.ndarray(shape=shape, dtype=dtype, buffer=buffer, strides=strides)
        except (TypeError, ValueError) as e:
            raise CvBridgeError(
                f"Image data ({len(buffer)} bytes) is too small for {img_msg.height} rows "
                f"of step {img_msg.step}"
            ) from e
        image = np.ascontiguousarray(image)
        if img_msg.is_bigendian == (sys.byteorder == "little"):
            # Bring foreign-endian data to native byte order (numpy 2 has no
            # ndarray.newbyteorder(), which cv_bridge uses here).
            image = image.astype(image.dtype.newbyteorder("="))
        if desired_encoding == "passthrough":
            return image
        return _convert_color(image, img_msg.encoding, desired_encoding)

    def compressed_imgmsg_to_cv2(
        self, cmprs_img_msg: CompressedImage, desired_encoding: str = "passthrough"
    ) -> Optional[np.ndarray]:
        """Decode a ``sensor_msgs/CompressedImage`` to a numpy array.

        OpenCV decodes colour images as BGR, so a non-passthrough ``desired_encoding``
        is converted from ``bgr8``, as cv_bridge does.
        """
        buffer = np.frombuffer(bytes(cmprs_img_msg.data), dtype=np.uint8).reshape(1, -1)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if desired_encoding == "passthrough":
            return image
        if image is None:
            raise CvBridgeError("Could not decode the compressed image")
        return _convert_color(image, "bgr8", desired_encoding)

    def cv2_to_imgmsg(
        self, cvim: np.ndarray, encoding: str = "passthrough", header=None
    ) -> Image:
        """Convert a numpy array to a ``sensor_msgs/Image``.

        With ``encoding="passthrough"`` the message encoding is the array's OpenCV type
        (e.g. ``8UC3``); otherwise the array must already match the given encoding.
        """
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        img_msg = Image()
        img_msg.height = cvim.shape[0]
        img_msg.width = cvim.shape[1]
        if header is not None:
            img_msg.header = header
        n_channels = 1 if len(cvim.shape) < 3 else cvim.shape[2]
        cv_type = self.dtype_with_channels_to_cvtype2(cvim.dtype, n_channels)
        if encoding == "passthrough":
            img_msg.encoding = cv_type
        else:
            img_msg.encoding = encoding
            depth, channels = _encoding_depth_and_channels(encoding)
            if f"{depth}C{channels}" != cv_type:
                raise CvBridgeError(
                    f"encoding specified as {encoding}, but image has incompatible type {cv_type}"
                )
        if cvim.dtype.byteorder == ">":
            img_msg.is_bigendian = True
        img_msg.data.frombytes(cvim.tobytes())
        img_msg.step = len(img_msg.data) // img_msg.height
        return img_msg

    def cv2_to_compressed_imgmsg(
        self, cvim: np.ndarray, dst_format: str = "jpg"
    ) -> CompressedImage:
        """Encode a numpy array as a ``sensor_msgs/CompressedImage`` (e.g. ``jpg``, ``png``)."""
        if not isinstance(cvim, (np.ndarray, np.generic)):
            raise TypeError("Your input type is not a numpy array")
        cmprs_img_msg = CompressedImage()
        cmprs_img_msg.format = dst_format
        try:
            ok, encoded = cv2.imencode("." + dst_format, cvim)
        except cv2.error as e:
            raise CvBridgeError(str(e)) from e
        if not ok:
            raise CvBridgeError(f"Could not encode the image as {dst_format}")
        cmprs_img_msg.data.frombytes(np.asarray(encoded).tobytes())
        return cmprs_img_msg
