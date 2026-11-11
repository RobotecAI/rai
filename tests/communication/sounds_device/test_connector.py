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
from rai.content import Audio

from rai_s2s.sound_device import SoundDeviceConfig, SoundDeviceConnector


@pytest.fixture
def mock_sound_device_api():
    with patch("rai_s2s.sound_device.api.SoundDeviceAPI") as mock:
        mock_instance = MagicMock()
        mock_instance.write = MagicMock()
        mock_instance.rec = MagicMock()
        mock_instance.stop = MagicMock()
        mock_instance.close_write_stream = MagicMock()
        mock.return_value = mock_instance
    return mock_instance


@pytest.fixture
def connector(mock_sound_device_api):
    device = sounddevice.query_devices(kind="input")
    if type(device) is dict:
        device_id = int(device["index"])
    elif isinstance(device, list):
        device_id = int(device[0]["index"])  # type: ignore
    else:
        raise AssertionError("No input device found")
    config = SoundDeviceConfig(
        stream=True,
        block_size=1024,
        dtype="float32",
        channels=1,
        consumer_sampling_rate=1600,
        device_number=device_id,
        device_name=None,
        is_input=True,
        is_output=True,
    )
    targets = [("speaker", config)]
    sources = [("microphone", config)]
    ret = SoundDeviceConnector(targets, sources)
    ret.devices["speaker"] = mock_sound_device_api
    ret.devices["microphone"] = mock_sound_device_api

    return ret


@pytest.fixture
def sine_wav():
    frequency = 440
    duration = 2.0
    sample_rate = 44100
    amplitude = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return Audio(np.int16(wave * 32767), sample_rate)


def test_play(connector, mock_sound_device_api, sine_wav):
    connector.play("speaker", sine_wav)
    mock_sound_device_api.write.assert_called_once_with(sine_wav, blocking=False)


def test_play_blocking(connector, mock_sound_device_api, sine_wav):
    connector.play("speaker", sine_wav, blocking=True)
    mock_sound_device_api.write.assert_called_once_with(sine_wav, blocking=True)


def test_stop(connector, mock_sound_device_api):
    connector.stop("speaker")
    mock_sound_device_api.stop.assert_called_once()


def test_record(connector, mock_sound_device_api, sine_wav):
    mock_sound_device_api.read.return_value = sine_wav
    assert connector.record("microphone", 1.0) is sine_wav
    mock_sound_device_api.read.assert_called_once_with(1.0, blocking=True)


def test_open_input_stream(connector, mock_sound_device_api):
    on_samples = MagicMock()
    connector.open_input_stream("microphone", on_samples)
    mock_sound_device_api.open_read_stream.assert_called_once()
    assert mock_sound_device_api.open_read_stream.call_args[0][0] is on_samples


def test_open_output_stream(connector, mock_sound_device_api):
    fill = MagicMock()
    connector.open_output_stream("speaker", fill, sample_rate=22050, channels=2)
    args = mock_sound_device_api.open_write_stream.call_args[0]
    assert args[0] is fill and args[2:] == (22050, 2)


def test_close_streams(connector, mock_sound_device_api):
    connector.close_streams("speaker")
    mock_sound_device_api.close_read_stream.assert_called_once()
    mock_sound_device_api.close_write_stream.assert_called_once()


def test_get_audio_params(connector, mock_sound_device_api):
    mock_sound_device_api.sample_rate = 16000
    mock_sound_device_api.in_channels = 1
    mock_sound_device_api.out_channels = 2
    assert connector.get_audio_params("speaker") == (16000, 1, 2)
