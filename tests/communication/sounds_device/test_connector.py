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

from rai.communication import HRIPayload
from rai.communication.sound_device import SoundDeviceConfig, SoundDeviceError
from rai.communication.sound_device.connector import (  # Replace with actual module name
    SoundDeviceConnector,
    SoundDeviceMessage,
)


@pytest.fixture
def mock_sound_device_api():
    with patch("rai.communication.sound_device.api.SoundDeviceAPI") as mock:
        mock_instance = MagicMock()
        mock_instance.play = MagicMock()
        mock_instance.rec = MagicMock()
        mock_instance.stop = MagicMock()
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
    return SoundDeviceConnector(targets, sources)


def test_send_message_play_audio(connector, mock_sound_device_api):
    message = SoundDeviceMessage(
        payload=HRIPayload(text="", audios=[np.array([1, 2, 3])])
    )
    connector.send_message(message, "speaker")
    connector.devices["speaker"].assert_called_once_with(b"test_audio")


def test_send_message_stop_audio(connector, mock_sound_device_api):
    message = SoundDeviceMessage(stop=True)
    connector.send_message(message, "speaker")
    connector.devices["speaker"].assert_called_once()


def test_send_message_read_error(connector):
    message = SoundDeviceMessage(read=True)
    with pytest.raises(
        SoundDeviceError,
        match="For recording use start_action or service_call with read=True",
    ):
        connector.send_message(message, "speaker")


def test_service_call_play_audio(connector, mock_sound_device_api):
    message = SoundDeviceMessage(payload=HRIPayload(text="", audios=["test_audio"]))
    result = connector.service_call(message, "speaker")
    mock_sound_device_api.play.assert_called_once_with(b"test_audio", blocking=True)
    assert isinstance(result, SoundDeviceMessage)


def test_service_call_read_audio(connector, mock_sound_device_api):
    mock_sound_device_api.record.return_value = b"recorded_audio"
    message = SoundDeviceMessage(read=True, duration=2.0)
    result = connector.service_call(message, "microphone")
    mock_sound_device_api.record.assert_called_once_with(2.0, blocking=True)
    assert result.payload.audios == [b"recorded_audio"]


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
    mock_sound_device_api.out_stream.stop.assert_called_once()
    assert "test_handle" not in connector.action_handles
