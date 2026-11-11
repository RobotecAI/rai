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

"""RAI content types and their conversion at the LLM boundary."""

import base64

import cv2
import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rai.communication import HRIMessage
from rai.content import Audio, Image, Text
from rai.messages import from_blocks, human, to_blocks, with_tool_images

RNG = np.random.default_rng(0)
PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

# ------------------------------------------------------------------- Image


@pytest.mark.parametrize(
    "encoding, shape, dtype",
    [
        ("rgb8", (4, 6, 3), np.uint8),
        ("bgr8", (4, 6, 3), np.uint8),
        ("rgba8", (4, 6, 4), np.uint8),
        ("mono8", (4, 6), np.uint8),
        ("mono16", (4, 6), np.uint16),
        ("32FC1", (4, 6), np.float32),
    ],
)
def test_image_accepts_matching_arrays(encoding, shape, dtype):
    data = np.zeros(shape, dtype)
    assert Image(data, encoding).data is data  # no copy


@pytest.mark.parametrize(
    "encoding, data",
    [
        ("rgb8", np.zeros((4, 6), np.uint8)),
        ("mono8", np.zeros((4, 6, 3), np.uint8)),
        ("rgb8", np.zeros((4, 6, 3), np.float32)),
        ("yuv422", np.zeros((4, 6, 2), np.uint8)),
    ],
)
def test_image_rejects_mismatched_arrays(encoding, data):
    with pytest.raises(ValueError):
        Image(data, encoding)


def test_image_color_conversion_round_trip():
    rgb = Image(RNG.integers(0, 255, (8, 8, 3), dtype=np.uint8))
    bgr = rgb.to("bgr8")
    assert np.array_equal(bgr.data, rgb.data[..., ::-1])
    assert np.array_equal(bgr.to("rgb8").data, rgb.data)


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "rgba8", "bgra8", "mono8"])
def test_image_png_encode_decode_is_lossless(encoding):
    channels = {"rgba8": 4, "bgra8": 4, "mono8": 0}.get(encoding, 3)
    shape = (8, 10, channels) if channels else (8, 10)
    image = Image(RNG.integers(0, 255, shape, dtype=np.uint8), encoding)
    decoded = Image.decode(image.encode(format="png"))
    expected = {"bgr8": "rgb8", "bgra8": "rgba8"}.get(encoding, encoding)
    assert decoded.encoding == expected
    assert np.array_equal(decoded.data, image.to(expected).data)


def test_image_jpeg_encode_keeps_colors_and_downscales():
    data = np.zeros((600, 2000, 3), np.uint8)
    data[..., 0] = 255  # red in rgb8
    decoded = Image.decode(Image(data).encode(max_side=1000))
    assert decoded.data.shape == (300, 1000, 3)
    assert decoded.data[..., 0].mean() > 240 and decoded.data[..., 2].mean() < 15


@pytest.mark.parametrize(
    "encoding, data",
    [
        ("mono16", np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000),
        ("32FC1", np.linspace(0, 5, 64, dtype=np.float32).reshape(8, 8)),
    ],
)
def test_depth_images_encode_as_8_bit_range(encoding, data):
    data = data.copy()
    if data.dtype == np.float32:
        data[0, 0] = np.nan
    decoded = Image.decode(Image(data, encoding).encode(format="png"))
    assert decoded.encoding == "mono8"
    assert decoded.data.max() == 255


# ------------------------------------------------------------------- Audio


def test_audio_mono_1d_becomes_n_by_1():
    audio = Audio(np.zeros(160, np.int16), 16000)
    assert audio.samples.shape == (160, 1)
    assert (audio.channels, audio.duration) == (1, 0.01)


@pytest.mark.parametrize("bad", [np.zeros(8, np.int32), np.zeros((2, 2, 2), np.int16)])
def test_audio_rejects_other_dtypes_and_shapes(bad):
    with pytest.raises(ValueError):
        Audio(bad, 16000)


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_int16_wav_round_trip_is_exact(channels):
    samples = RNG.integers(-32768, 32767, (500, channels), dtype=np.int16)
    audio = Audio.from_wav(Audio(samples, 44100).to_wav())
    assert audio.samples.dtype == np.int16
    assert (audio.sample_rate, audio.channels) == (44100, channels)
    assert np.array_equal(audio.samples, samples)


@pytest.mark.parametrize("channels", [1, 2])
def test_audio_float32_int16_conversion(channels):
    samples = RNG.uniform(-1, 1, (500, channels)).astype(np.float32)
    audio = Audio(samples, 22050)
    back = audio.as_int16().as_float32()
    assert back.samples.dtype == np.float32
    assert np.abs(back.samples - samples).max() < 1 / 16384
    assert audio.as_float32() is audio
    loud = Audio(np.array([[1.5], [-1.5]], np.float32), 8000).as_int16()
    assert loud.samples.ravel().tolist() == [32767, -32768]  # clipped, not wrapped


# ------------------------------------------------------------ LLM boundary


def test_to_blocks_and_back():
    image = Image(RNG.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    audio = Audio(RNG.integers(-1000, 1000, (100, 2), dtype=np.int16), 16000)
    blocks = to_blocks(["hello", Text("world"), image, audio])
    assert [b["type"] for b in blocks] == ["text", "text", "image", "audio"]
    assert blocks[2]["mime_type"] == "image/jpeg"
    assert blocks[3]["mime_type"] == "audio/wav"
    text1, text2, image2, audio2 = from_blocks(blocks)
    assert (text1, text2) == (Text("hello"), Text("world"))
    assert isinstance(image2, Image) and image2.data.shape == (16, 16, 3)
    assert isinstance(audio2, Audio) and np.array_equal(audio2.samples, audio.samples)


def test_to_blocks_rejects_other_objects():
    with pytest.raises(TypeError):
        to_blocks([b"bytes"])  # type: ignore[list-item]


def test_from_blocks_reads_provider_formats():
    wav = base64.b64encode(Audio(np.zeros(10, np.int16), 8000).to_wav()).decode()
    content = from_blocks(
        [
            "plain",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"},
            },
            {"type": "image_url", "image_url": "https://example.com/a.png"},
            {"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}},
            {"type": "tool_use", "id": "x"},
        ]
    )
    assert [type(c).__name__ for c in content] == ["Text", "Image", "Audio"]
    assert from_blocks("") == [] and from_blocks("hi") == [Text("hi")]


def test_human_builds_text_or_content_blocks():
    assert human("hi") == HumanMessage("hi")
    jpeg = base64.b64encode(
        cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))[1]
    ).decode()
    message = human("look", [PNG_1PX, jpeg], [Audio(np.zeros(4, np.int16), 8000)])
    types = [(b["type"], b.get("mime_type")) for b in message.content]
    assert types == [
        ("text", None),
        ("image", "image/png"),
        ("image", "image/jpeg"),
        ("audio", "audio/wav"),
    ]
    assert message.text == "look"
    assert [b["type"] for b in human(images=[PNG_1PX]).content] == ["image"]


def test_hri_message_to_the_model_and_back():
    image = Image(np.full((8, 8, 3), 200, np.uint8))
    message = HRIMessage("what is this?", [image], communication_id="c")
    lc = human(message.text, message.images, message.audios)
    text, decoded = from_blocks(lc.content)
    assert text == Text("what is this?")
    assert np.abs(decoded.data.astype(int) - 200).max() <= 2  # JPEG


def test_with_tool_images_adds_one_message_per_tool_run():
    image = Image(np.zeros((4, 4, 3), np.uint8))
    call = {"name": "cam", "args": {}, "id": "1"}
    messages = [
        HumanMessage("go"),
        AIMessage("", tool_calls=[call, {**call, "id": "2"}]),
        ToolMessage("a", tool_call_id="1", artifact={"images": [image]}),
        ToolMessage("b", tool_call_id="2", artifact={"images": [PNG_1PX]}),
        AIMessage("done"),
    ]
    out = with_tool_images(messages)
    assert [m.type for m in out] == ["human", "ai", "tool", "tool", "human", "ai"]
    assert [b["type"] for b in out[4].content] == ["text", "image", "image"]
    assert with_tool_images(messages[:3])[-1].type == "human"
    assert with_tool_images(messages[:2]) == messages[:2]
