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


import logging
from dataclasses import dataclass
from threading import Event, Thread
from typing import TYPE_CHECKING, Optional
from typing_extensions import Self
from uuid import uuid4

from numpy._typing import NDArray
from pydub import AudioSegment
from rai.agents.base import BaseAgent
from rai.communication.ros2 import (
    IROS2Message,
    ROS2HRIConnector,
    ROS2HRIMessage,
    TopicConfig,
)
from rai.communication.sound_device import SoundDeviceConfig, SoundDeviceConnector
from rai.communication.sound_device.connector import SoundDeviceMessage
from std_msgs.msg import String

from rai_interfaces.msg._hri_message import HRIMessage
from rai_tts.models.base import TTSModel

from .initialization import load_config

if TYPE_CHECKING:
    from rai.communication.sound_device.api import SoundDeviceConfig

from queue import Empty, Queue

import numpy as np

# This file contains every concurrent programming antipattern known to man
# The words callback hell are insufficient to describe the cacophony of function calls
# wreathing havoc along the 9 circles of threads
# Ye who enter here abandon all hope
#
# It works tho


@dataclass
class PlayData:
    playing: bool = False
    current_segment: Optional[AudioSegment] = None
    data: Optional[NDArray] = None
    channels: int = 1
    current_frame: int = 0


class TextToSpeechAgent(BaseAgent):
    """
    Agent responsible for converting text to speech and handling audio playback.

    Parameters
    ----------
    speaker_config : SoundDeviceConfig
        Configuration for the sound device used for playback.
    ros2_name : str
        Name of the ROS2 node.
    tts : TTSModel
        Text-to-speech model used for generating audio.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages, by default None.
    max_speech_history : int, optional
        Maximum amount of speech ids to remember, by default 64
    """

    def __init__(
        self,
        speaker_config: SoundDeviceConfig,
        ros2_name: str,
        tts: TTSModel,
        logger: Optional[logging.Logger] = None,
        max_speech_history=64,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        speaker = SoundDeviceConnector(
            targets=[("speaker", speaker_config)], sources=[]
        )
        sample_rate, _, out_channels = speaker.get_audio_params("speaker")
        tts.sample_rate = sample_rate
        tts.channels = out_channels

        self.node_base_name = ros2_name
        self.model = tts
        ros2_connector = self._setup_ros2_connector()
        self.connectors = {"ros2": ros2_connector, "speaker": speaker}
        super().__init__()

        self.current_transcription_id = str(uuid4())[0:8]
        self.current_speech_id = None
        self.text_queues: dict[str, Queue] = {self.current_transcription_id: Queue()}
        self.audio_queues: dict[str, Queue] = {self.current_transcription_id: Queue()}
        self.remembered_speech_ids: list[str] = []

        self.tog_play_event = Event()
        self.stop_event = Event()
        self.current_audio = None

        self.terminate_agent = Event()
        self.transcription_thread = None
        self.running = False

        self.playback_data = PlayData()

    @classmethod
    def from_config(cls, cfg_path: Optional[str] = None) -> Self:
        cfg = load_config(cfg_path)
        config = SoundDeviceConfig(
            stream=True,
            is_output=True,
            device_name=cfg.speaker.device_name,
        )
        match cfg.text_to_speech.model_type:
            case "ElevenLabs":
                from rai_tts.models import ElevenLabsTTS

                if cfg.text_to_speech.voice != "":
                    model = ElevenLabsTTS(voice=cfg.text_to_speech.voice)
                else:
                    raise ValueError("ElevenLabs [tts] vendor required voice to be set")
            case "OpenTTS":
                from rai_tts.models import OpenTTS

                if cfg.text_to_speech.voice != "":
                    model = OpenTTS(voice=cfg.text_to_speech.voice)
                else:
                    model = OpenTTS()
            case _:
                raise ValueError(f"Unknown model_type: {cfg.text_to_speech.model_type}")
        return cls(config, "rai_auto_tts", model)

    def __call__(self):
        self.run()

    def run(self):
        """
        Start the text-to-speech agent, initializing playback and launching the transcription thread.
        """
        self.running = True
        self.logger.info("TextToSpeechAgent started")
        self.transcription_thread = Thread(target=self._transcription_thread)
        self.transcription_thread.start()

        msg = SoundDeviceMessage(read=False)
        assert isinstance(self.connectors["speaker"], SoundDeviceConnector)
        self.connectors["speaker"].start_action(
            msg,
            "speaker",
            on_feedback=self._speaker_callback,
            on_done=lambda: None,
        )

    def _speaker_callback(self, outdata, frames, time, status_dict):
        set_flags = [flag for flag, status in status_dict.items() if status]

        if set_flags:
            self.logger.warning("Flags set:" + ", ".join(set_flags))
        if self.playback_data.playing:
            if self.playback_data.current_segment is None:
                try:
                    self.playback_data.current_segment = self.audio_queues[
                        self.current_transcription_id
                    ].get(block=False)
                    self.playback_data.data = np.array(
                        self.playback_data.current_segment.get_array_of_samples()  # type: ignore
                    ).reshape(-1, self.playback_data.channels)
                except Empty:
                    pass
                except KeyError:
                    pass
            if self.playback_data.data is not None:
                current_frame = self.playback_data.current_frame
                chunksize = min(len(self.playback_data.data) - current_frame, frames)
                outdata[:chunksize] = self.playback_data.data[
                    current_frame : current_frame + chunksize
                ]
                if chunksize < frames:
                    outdata[chunksize:] = 0
                    self.playback_data.current_frame = 0
                    self.playback_data.current_segment = None
                    self.playback_data.data = None
                else:
                    self.playback_data.current_frame += chunksize

        if not self.playback_data.playing:
            outdata[:] = np.zeros(outdata.size).reshape(outdata.shape)

    def stop(self):
        """
        Clean exit the text-to-speech agent, terminating playback and joining the transcription thread.
        """
        self.logger.info("Stopping TextToSpeechAgent")
        self.terminate_agent.set()
        if self.transcription_thread is not None:
            self.transcription_thread.join()

    def _transcription_thread(self):
        while not self.terminate_agent.wait(timeout=0.01):
            if self.current_transcription_id in self.text_queues:
                try:
                    data = self.text_queues[self.current_transcription_id].get(
                        block=False
                    )
                except Empty:
                    continue
                audio = self.model.get_speech(data)
                try:
                    self.audio_queues[self.current_transcription_id].put(audio)
                except KeyError as e:
                    self.logger.error(
                        f"Could not find queue for {self.current_transcription_id}: queuse: {self.audio_queues.keys()}"
                    )
                    raise e

    def _setup_ros2_connector(self):
        to_human = TopicConfig(
            msg_type="rai_interfaces/msg/HRIMessage",
            auto_qos_matching=True,
            is_subscriber=True,
            subscriber_callback=self._on_to_human_message,
            source_author="ai",
        )
        voice_commands = TopicConfig(
            msg_type="std_msgs/msg/String",
            auto_qos_matching=True,
            is_subscriber=True,
            subscriber_callback=self._on_command_message,
            source_author="human",
        )
        return ROS2HRIConnector(
            node_name=self.node_base_name,
            sources=[("/to_human", to_human), ("/voice_commands", voice_commands)],
        )

    def _on_to_human_message(self, message: IROS2Message):
        assert isinstance(message, HRIMessage)
        msg = ROS2HRIMessage.from_ros2(message, "ai")
        self.logger.info(f"Receieved message from human: {message.text}")
        self.logger.warning(
            f"Starting playback, current id: {self.current_transcription_id}"
        )
        if (
            self.current_speech_id is None
            and msg.conversation_id is not None
            and msg.conversation_id not in self.remembered_speech_ids
        ):
            self.current_speech_id = msg.conversation_id
            self.remembered_speech_ids.append(self.current_speech_id)
            if len(self.remembered_speech_ids) > 64:
                self.remembered_speech_ids.pop(0)
        if self.current_speech_id == msg.conversation_id:
            self.text_queues[self.current_transcription_id].put(msg.text)
        self.playback_data.playing = True

    def _on_command_message(self, message: IROS2Message):
        assert isinstance(message, String)
        self.logger.debug(f"Receieved status message: {message}")
        if message.data == "tog_play":
            self.playback_data.playing = not self.playback_data.playing
        elif message.data == "play":
            self.playback_data.playing = True
        elif message.data == "pause":
            self.playback_data.playing = False
        elif message.data == "stop":
            self.current_speech_id = None
            self.playback_data.playing = False
            previous_id = self.current_transcription_id
            self.logger.warning(f"Stopping playback, previous id: {previous_id}")
            self.current_transcription_id = str(uuid4())[0:8]
            self.audio_queues[self.current_transcription_id] = Queue()
            self.text_queues[self.current_transcription_id] = Queue()
            try:
                del self.audio_queues[previous_id]
                del self.text_queues[previous_id]
            except KeyError:
                pass
            self.playback_data.data = None
            self.playback_data.current_frame = 0
            self.playback_data.current_segment = None

        self.logger.debug(f"Current status is: {self.playback_data.playing}")
