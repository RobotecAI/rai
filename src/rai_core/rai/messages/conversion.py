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

    Supported inputs:
    - PIL Image.Image
    - str path, str file URL (file://...), or HTTP(S) URL
    - bytes containing image data
    - numpy.ndarray (uint8 or float32/float64; grayscale or 3/4-channel)

    All inputs are decoded and re-encoded to PNG to guarantee consistent output.
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
