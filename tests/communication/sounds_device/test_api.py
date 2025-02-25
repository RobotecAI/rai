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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice
from rai.communication.sound_device import (
    SoundDeviceAPI,
    SoundDeviceConfig,
    SoundDeviceError,
)


@pytest.fixture
def mock_sd():
    mock_play = MagicMock()
    mock_rec = MagicMock(
        return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
    )  # Simulated recorded data
    mock_open = MagicMock()
    mock_stop = MagicMock()
    mock_wait = MagicMock()

    with (
        patch.object(sounddevice, "play", mock_play),
        patch.object(sounddevice, "rec", mock_rec),
        patch.object(sounddevice, "open", mock_open),
        patch.object(sounddevice, "stop", mock_stop),
        patch.object(sounddevice, "wait", mock_wait),
    ):
        yield {
            "play": mock_play,
            "rec": mock_rec,
            "open": mock_open,
            "stop": mock_stop,
            "wait": mock_wait,
        }


@pytest.fixture
def input_device_id():
    device = sounddevice.query_devices(kind="input")
    if type(device) is dict:
        return int(device["index"])
    elif isinstance(device, list):
        return int(device[0]["index"])  # type: ignore
    raise AssertionError("No input device found")


@pytest.mark.parametrize(
    "stream, block_size, dtype, channels, consumer_sampling_rate, is_input, is_output",
    [
        (
            True,
            1024,
            "float32",
            2,
            44100,
            True,
            True,
        ),  # Standard input/output config
        (
            False,
            512,
            "int16",
            1,
            22050,
            True,
            False,
        ),  # Read-only device
        (
            True,
            2048,
            "float64",
            2,
            48000,
            False,
            True,
        ),  # Write-only device
    ],
)
def test_init(
    input_device_id,
    mock_sd,
    stream,
    block_size,
    dtype,
    channels,
    consumer_sampling_rate,
    is_input,
    is_output,
):
    """Test different configurations of SoundDeviceAPI"""
    config = SoundDeviceConfig(
        stream=stream,
        block_size=block_size,
        dtype=dtype,
        channels=channels,
        consumer_sampling_rate=consumer_sampling_rate,
        device_number=input_device_id,
        device_name=None,
        is_input=is_input,
        is_output=is_output,
    )

    if not mock_sd:
        with pytest.raises(
            SoundDeviceError, match="SoundDeviceAPI requires sound_device module!"
        ):
            SoundDeviceAPI(config)
    else:
        api = SoundDeviceAPI(config)
        assert api.config == config


@pytest.mark.parametrize("is_output", [True, False])
def test_write_unsupported(input_device_id, mock_sd, is_output):
    """Ensure writing raises an error if output is not supported."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=True,
        is_output=is_output,
    )
    api = SoundDeviceAPI(config)

    if not is_output:
        with pytest.raises(SoundDeviceError, match="does not support writing!"):
            api.write(np.array([0.0, 1.0]))
    else:
        api.write(np.array([0.0, 1.0]), blocking=True)
        mock_sd["play"].assert_called_once()


@pytest.mark.parametrize("is_input", [True, False])
def test_read_unsupported(input_device_id, mock_sd, is_input):
    """Ensure reading raises an error if input is not supported."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=is_input,
        is_output=True,
    )
    api = SoundDeviceAPI(config)

    if not is_input:
        with pytest.raises(SoundDeviceError, match="does not support reading!"):
            api.read(1.0)
    else:
        mock_sd["rec"].return_value = np.array([[0.0], [1.0]])
        result = api.read(1.0, blocking=True)
        np.testing.assert_array_equal(result, np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("method", ["stop", "wait"])
def test_control_methods(input_device_id, mock_sd, method):
    """Test stop and wait methods."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=True,
        is_output=True,
    )
    api = SoundDeviceAPI(config)

    getattr(api, method)()
    mock_sd[method].assert_called_once()


@pytest.mark.parametrize("is_output", [True, False])
def test_open_write_stream_unsupported(input_device_id, mock_sd, is_output):
    """Ensure opening a write stream raises an error if not supported."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=True,
        is_output=is_output,
    )
    api = SoundDeviceAPI(config)

    if not is_output:
        with pytest.raises(
            SoundDeviceError, match="does not support streaming writing!"
        ):
            api.open_write_stream(lambda x, y, z, w: None)


@pytest.mark.parametrize("is_input", [True, False])
def test_open_read_stream_unsupported(input_device_id, mock_sd, is_input):
    """Ensure opening a read stream raises an error if not supported."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=is_input,
        is_output=True,
    )
    api = SoundDeviceAPI(config)

    if not is_input:
        with pytest.raises(
            SoundDeviceError, match="does not support streaming reading!"
        ):
            api.open_read_stream(lambda: None)  # type: ignore


@pytest.mark.parametrize("has_stream", [True, False])
def test_close_read_stream(input_device_id, mock_sd, has_stream):
    """Test closing an active read stream."""
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=2,
        consumer_sampling_rate=44100,
        device_number=input_device_id,
        device_name=None,
        is_input=True,
        is_output=True,
    )
    api = SoundDeviceAPI(config)

    if has_stream:
        api.in_stream = MagicMock()
        api.close_read_stream()
        assert api.in_stream is None
    else:
        api.close_read_stream()  # Should not raise an error
