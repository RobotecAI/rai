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
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from rai.messages import preprocess_image


def decode_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


@pytest.mark.parametrize(
    "test_image",
    [
        np.zeros((300, 300), dtype=np.uint8),
        np.zeros((300, 300, 3), dtype=np.uint8),
        np.zeros((300, 300, 4), dtype=np.uint8),
        np.random.rand(300, 300).astype(np.float32),
        np.random.rand(300, 300, 3).astype(np.float32),
        np.random.rand(300, 300, 4).astype(np.float32),
        np.random.rand(300, 300).astype(np.float64),
        np.random.rand(300, 300, 3).astype(np.float64),
        np.random.rand(300, 300, 4).astype(np.float64),
        "tests/resources/image.png",
    ],
)
def test_preprocess_image_always_png(test_image):
    base64_image = preprocess_image(test_image)
    img = decode_image(base64_image)
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0


def test_preprocess_image_from_bytes_and_file_url(tmp_path: Path):
    # Create a temporary PNG file
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    file_path = tmp_path / "tmp.png"
    pil_img.save(file_path, format="PNG")

    # bytes input
    with open(file_path, "rb") as f:
        raw_bytes = f.read()
    b64_bytes = preprocess_image(raw_bytes)
    img_from_bytes = decode_image(b64_bytes)
    assert img_from_bytes.format == "PNG"
    assert img_from_bytes.size == (32, 32)

    # file:// URL input
    file_url = f"file://{file_path}"
    b64_file_url = preprocess_image(file_url)
    img_from_url = decode_image(b64_file_url)
    assert img_from_url.format == "PNG"
    assert img_from_url.size == (32, 32)


def test_preprocess_image_unsupported_type():
    class NotAnImage:
        pass

    with pytest.raises(TypeError):
        _ = preprocess_image(NotAnImage())


def test_preprocess_image_http_url():
    resources_dir = Path(__file__).resolve().parents[1] / "resources"
    image_path = resources_dir / "image.png"
    assert image_path.exists(), "tests/resources/image.png must exist for this test"

    handler = partial(SimpleHTTPRequestHandler, directory=str(resources_dir))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)

    try:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        url = f"http://127.0.0.1:{port}/image.png"
        base64_image = preprocess_image(url)
        img = decode_image(base64_image)
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0
    finally:
        httpd.shutdown()
        httpd.server_close()
