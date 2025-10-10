# Copyright (C) 2024 Robotec.AI
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


import base64
import os
from io import BytesIO
from typing import Callable, Union

import numpy as np
import requests
from PIL import Image as PILImage
from PIL.Image import Image


def preprocess_image(
    image: Union[Image, str, bytes, np.ndarray],
    encoding_function: Callable[[bytes], str] = lambda b: base64.b64encode(b).decode(
        "utf-8"
    ),
) -> str:
    """Convert various image inputs into a base64-encoded PNG string.

    Parameters
    ----------
    image : PIL.Image.Image or str or bytes or numpy.ndarray
        Supported inputs:
        - PIL Image
        - Path to a file, ``file://`` URL, or HTTP(S) URL
        - Raw bytes containing image data
        - ``numpy.ndarray`` with dtype ``uint8`` or ``float32``/``float64``;
          grayscale or 3/4-channel arrays are supported.
    encoding_function : callable, optional
        Function that converts PNG bytes to the final string representation.
        By default, returns base64-encoded UTF-8 string.

    Returns
    -------
    str
        Base64-encoded PNG string.

    Raises
    ------
    FileNotFoundError
        If a file path (or ``file://`` URL) does not exist.
    TypeError
        If the input type is not supported.
    requests.HTTPError
        If fetching an HTTP(S) URL fails with a non-2xx response.
    requests.RequestException
        If a network error occurs while fetching an HTTP(S) URL.
    OSError
        If the input cannot be decoded as an image by Pillow.

    Notes
    -----
    - All inputs are decoded and re-encoded to PNG to guarantee consistent output.
    - Float arrays are assumed to be in [0, 1] and are scaled to ``uint8``.
    - Network requests use a timeout of ``(5, 15)`` seconds (connect, read).

    Examples
    --------
    >>> b64_png = preprocess_image("path/to/image.jpg")
    >>> import numpy as np
    >>> arr = np.random.rand(64, 64, 3).astype(np.float32)
    >>> b64_png = preprocess_image(arr)
    """

    def _to_pil_from_ndarray(arr: np.ndarray) -> Image:
        a = arr
        if a.dtype in (np.float32, np.float64):
            a = np.clip(a, 0.0, 1.0)
            a = (a * 255.0).round().astype(np.uint8)
        a = np.ascontiguousarray(a)
        return PILImage.fromarray(a)

    def _ensure_pil(img: Union[Image, str, bytes, np.ndarray]) -> Image:
        if isinstance(img, Image):
            return img
        if isinstance(img, np.ndarray):  # type: ignore
            return _to_pil_from_ndarray(img)
        if isinstance(img, str):
            if img.startswith(("http://", "https://")):
                response = requests.get(img, timeout=(5, 15))
                response.raise_for_status()
                return PILImage.open(BytesIO(response.content))
            if img.startswith("file://"):
                file_path = img[len("file://") :]
            else:
                # fallback to file path if not marked with file://
                file_path = img
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            return PILImage.open(file_path)
        if isinstance(img, bytes):
            return PILImage.open(BytesIO(img))
        raise TypeError(f"Unsupported image type: {type(img).__name__}")

    pil_image = _ensure_pil(image)

    # Normalize to PNG bytes
    with BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

    return encoding_function(png_bytes)
