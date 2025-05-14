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
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice
from pydub import AudioSegment
from scipy.io import wavfile

from rai_s2s.sound_device import (
    SoundDeviceConfig,
    SoundDeviceConnector,
    SoundDeviceError,
    SoundDeviceMessage,
)


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
    wave_int16 = np.int16(wave * 32767)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, wave_int16)
    audio = AudioSegment.from_wav(wav_buffer)
    return audio


@pytest.fixture
def binary_audio():
    frequency = 440
    duration = 2.0
    sample_rate = 44100
    amplitude = 0.5
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    wave_int16 = np.int16(wave * 32767)
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, sample_rate, wave_int16)

    wav_binary = wav_buffer.getvalue()
    return wav_binary


def test_send_message_play_audio(connector, mock_sound_device_api, sine_wav):
    message = SoundDeviceMessage(
        text="",
        audios=[sine_wav],
        message_author="human",
    )
    connector.send_message(message, "speaker")
    connector.devices["speaker"].write.assert_called_once()


def test_send_message_stop_audio(connector, mock_sound_device_api):
    message = SoundDeviceMessage(stop=True)
    connector.send_message(message, "speaker")
    connector.devices["speaker"].stop.assert_called_once()


def test_send_message_read_error(connector):
    message = SoundDeviceMessage(read=True)
    with pytest.raises(
        SoundDeviceError,
        match="For recording use start_action or service_call with read=True",
    ):
        connector.send_message(message, "speaker")


def test_service_call_play_audio(connector, mock_sound_device_api, sine_wav):
    message = SoundDeviceMessage(text="", audios=[sine_wav], message_author="ai")
    result = connector.service_call(message, "speaker")
    mock_sound_device_api.write.assert_called_once()
    assert isinstance(result, SoundDeviceMessage)


def test_service_call_read_audio(connector, mock_sound_device_api, sine_wav):
    mock_sound_device_api.read.return_value = sine_wav
    message = SoundDeviceMessage(read=True)
    result = connector.service_call(message, "microphone")
    mock_sound_device_api.read.assert_called_once_with(1.0, blocking=True)

    assert result.audios == [sine_wav]


def test_service_call_stop_error(connector):
    message = SoundDeviceMessage(stop=True)
    with pytest.raises(
        SoundDeviceError, match="For stopping use send_message with stop=True."
    ):
        connector.service_call(message, "speaker")


def test_start_action_read(connector, mock_sound_device_api):
    message = SoundDeviceMessage(read=True)
    handle = connector.start_action(message, "microphone", MagicMock(), MagicMock())
    mock_sound_device_api.open_read_stream.assert_called_once()
    assert handle in connector.action_handles


def test_start_action_write(connector, mock_sound_device_api):
    message = SoundDeviceMessage()
    handle = connector.start_action(message, "speaker", MagicMock(), MagicMock())
    mock_sound_device_api.open_write_stream.assert_called_once()
    assert handle in connector.action_handles


def test_terminate_action(connector, mock_sound_device_api):
    connector.action_handles["test_handle"] = ("speaker", False)
    connector.terminate_action("test_handle")
    mock_sound_device_api.close_write_stream.assert_called_once()
    assert "test_handle" not in connector.action_handles
