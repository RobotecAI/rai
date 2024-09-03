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
#

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from rai.messages.utils import preprocess_image


def decode_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return image


def test_preprocess_image():
    link = "https://upload.wikimedia.org/wikipedia/commons/6/63/Wikipedia-logo.png"
    numpy_image = np.zeros((300, 300, 3))
    local_image = "tests/resources/image.png"

    test_images = [link, numpy_image, local_image]
    for test_image in test_images:
        base64_image = preprocess_image(test_image)
        image = decode_image(base64_image)  # noqa: F841
