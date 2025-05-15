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
import time
from threading import Event, Lock, Thread
from typing import Any, List, Optional, TypedDict
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from rai.agents.base import BaseAgent
from rai.communication.ros2 import (
    ROS2Connector,
    ROS2HRIConnector,
    ROS2HRIMessage,
    ROS2Message,
)
from typing_extensions import Self

from rai_s2s.asr.agents.initialization import load_config
from rai_s2s.asr.models import BaseTranscriptionModel, BaseVoiceDetectionModel
from rai_s2s.sound_device import (
    SoundDeviceConfig,
    SoundDeviceConnector,
    SoundDeviceMessage,
)


class ThreadData(TypedDict):
    thread: Thread
    event: Event
    transcription: str
    joined: bool


class SpeechRecognitionAgent(BaseAgent):
    """
    Agent responsible for voice recognition, transcription, and processing voice activity.

    Parameters
    ----------
    microphone_config : SoundDeviceConfig
        Configuration for the microphone device used for audio input.
    ros2_name : str
        Name of the ROS2 node.
    transcription_model : BaseTranscriptionModel
        Model used for transcribing audio input to text.
    vad : BaseVoiceDetectionModel
        Voice activity detection model used to determine when speech is present.
    grace_period : float, optional
        Time in seconds to wait before stopping recording after speech ends, by default 1.0.
    logger : Optional[logging.Logger], optional
        Logger instance for logging messages, by default None.
    """

    def __init__(
        self,
        microphone_config: SoundDeviceConfig,
        ros2_name: str,
        transcription_model: BaseTranscriptionModel,
        vad: BaseVoiceDetectionModel,
        grace_period: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.microphone = SoundDeviceConnector(
            targets=[], sources=[("microphone", microphone_config)]
        )
        self.ros2_hri_connector = ROS2HRIConnector(ros2_name)
        self.ros2_connector = ROS2Connector(ros2_name + "ari")
        super().__init__()
        self.should_record_pipeline: List[BaseVoiceDetectionModel] = []
        self.should_stop_pipeline: List[BaseVoiceDetectionModel] = []

        self.transcription_model = transcription_model
        self.transcription_lock = Lock()

        self.vad: BaseVoiceDetectionModel = vad

        self.grace_period = grace_period
        self.grace_period_start = 0

        self.recording_started = False
        self.ran_setup = False

        self.sample_buffer = []
        self.sample_buffer_lock = Lock()
        self.active_thread = ""
        self.transcription_threads: dict[str, ThreadData] = {}
        self.transcription_buffers: dict[str, list[NDArray]] = {}
        self.is_playing = True

    @classmethod
    def from_config(cls, cfg_path: Optional[str] = None) -> Self:
        cfg = load_config(cfg_path)
        microphone_configuration = SoundDeviceConfig(
            stream=True,
            channels=1,
            device_name=cfg.microphone.device_name,
            block_size=1280,
            consumer_sampling_rate=16000,
            dtype="int16",
            device_number=None,
            is_input=True,
            is_output=False,
        )
        match cfg.transcribe.model_type:
            case "LocalWhisper (Free)":
                from rai_s2s.asr.models import LocalWhisper

                model = LocalWhisper(
                    cfg.transcribe.model_name, 16000, language=cfg.transcribe.language
                )
            case "FasterWhisper (Free)":
                from rai_s2s.asr.models import FasterWhisper

                model = FasterWhisper(
                    cfg.transcribe.model_name, 16000, language=cfg.transcribe.language
                )
            case "OpenAI (Cloud)":
                from rai_s2s.asr.models import OpenAIWhisper

                model = OpenAIWhisper(
                    cfg.transcribe.model_name, 16000, language=cfg.transcribe.language
                )
            case _:
                raise ValueError(f"Unknown model name f{cfg.transcribe.model_name}")

        match cfg.voice_activity_detection.model_name:
            case "SileroVAD":
                from rai_s2s.asr.models import SileroVAD

                vad = SileroVAD(16000, cfg.voice_activity_detection.threshold)

        agent = cls(microphone_configuration, "rai_auto_asr_agent", model, vad)
        if cfg.wakeword.is_used:
            match cfg.wakeword.model_type:
                case "OpenWakeWord":
                    from rai_s2s.asr.models import OpenWakeWord

                    agent.add_detection_model(
                        OpenWakeWord(cfg.wakeword.model_name, cfg.wakeword.threshold)
                    )
        return agent

    def __call__(self):
        self.run()

    def add_detection_model(
        self, model: BaseVoiceDetectionModel, pipeline: str = "record"
    ):
        """
        Add a voice detection model to the specified processing pipeline.

        Parameters
        ----------
        model : BaseVoiceDetectionModel
            The voice detection model to be added.
        pipeline : str, optional
            The pipeline where the model should be added, either 'record' or 'stop'.
            Default is 'record'.

        Raises
        ------
        ValueError
            If the specified pipeline is not 'record' or 'stop'.
        """

        if pipeline == "record":
            self.should_record_pipeline.append(model)
        elif pipeline == "stop":
            self.should_stop_pipeline.append(model)
        else:
            raise ValueError("Pipeline should be either 'record' or 'stop'")

    def run(self):
        """
        Start the voice recognition agent, initializing the microphone and handling incoming audio samples.
        """
        self.running = True
        msg = SoundDeviceMessage(read=True)
        self.listener_handle = self.microphone.start_action(
            action_data=msg,
            target="microphone",
            on_feedback=self._on_new_sample,
            on_done=lambda: None,
        )
        self.logger.info("Started Voice Agent")

    def stop(self):
        """
        Clean exit the voice recognition agent, ensuring all transcription threads finish before termination.
        """
        self.logger.info("Stopping Voice Agent")
        self.running = False
        self.microphone.terminate_action(self.listener_handle)
        self.ros2_hri_connector.shutdown()
        self.ros2_connector.shutdown()
        while not all(
            [thread["joined"] for thread in self.transcription_threads.values()]
        ):
            for thread_id in self.transcription_threads:
                if self.transcription_threads[thread_id]["event"].is_set():
                    self.transcription_threads[thread_id]["thread"].join()
                    self.transcription_threads[thread_id]["joined"] = True
                else:
                    self.logger.info(
                        f"Waiting for transcription of {thread_id} to finish..."
                    )
        self.logger.info("Voice agent stopped")

    def _on_new_sample(self, indata: np.ndarray, status_flags: dict[str, Any]):
        sample_time = time.time()
        with self.sample_buffer_lock:
            self.sample_buffer.append(indata)
            if not self.recording_started and len(self.sample_buffer) > 5:
                self.sample_buffer = self.sample_buffer[-5:]

        # attempt to join finished threads:
        for thread_id in self.transcription_threads:
            if self.transcription_threads[thread_id]["event"].is_set():
                self.transcription_threads[thread_id]["thread"].join()
                self.transcription_threads[thread_id]["joined"] = True

        voice_detected, output_parameters = self.vad(indata, {})
        self.logger.debug(f"Voice detected: {voice_detected}: {output_parameters}")
        should_record = False
        if voice_detected and not self.recording_started:
            should_record = self._should_record(indata, output_parameters)

        if should_record:
            self.logger.info("starting recording...")
            self.recording_started = True
            thread_id = str(uuid4())[0:8]
            transcription_thread = Thread(
                target=self._transcription_thread,
                args=[thread_id],
            )
            transcription_finished = Event()
            self.active_thread = thread_id
            self.transcription_threads[thread_id] = {
                "thread": transcription_thread,
                "event": transcription_finished,
                "transcription": "",
                "joined": False,
            }

        if voice_detected:
            self.logger.debug("Voice detected... resetting grace period")
            self.grace_period_start = sample_time
            self._send_ros2_message("pause", "/voice_commands")
            self.is_playing = False
        if (
            self.recording_started
            and sample_time - self.grace_period_start > self.grace_period
        ):
            self.logger.info(
                "Grace period ended... stopping recording, starting transcription"
            )
            self.recording_started = False
            self.grace_period_start = 0
            with self.sample_buffer_lock:
                self.transcription_buffers[self.active_thread] = self.sample_buffer
                self.sample_buffer = []
            self.transcription_threads[self.active_thread]["thread"].start()
            self.active_thread = ""
            self._send_ros2_message("stop", "/voice_commands")
            self.is_playing = False
        elif not self.is_playing and (
            sample_time - self.grace_period_start > self.grace_period
        ):
            self._send_ros2_message("play", "/voice_commands")
            self.is_playing = True

    def _should_record(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> bool:
        if len(self.should_record_pipeline) == 0:
            return True
        for model in self.should_record_pipeline:
            detected, output = model(audio_data, input_parameters)
            self.logger.debug(f"detected {detected}, output {output}")
            if detected:
                model.reset()
                return True
        return False

    def _transcription_thread(self, identifier: str):
        self.logger.info(f"transcription thread {identifier} started")
        audio_data = np.concatenate(self.transcription_buffers[identifier])

        # NOTE: this is only necessary for the local model, but it seems to cause no relevant performance drops in case of cloud models
        with self.transcription_lock:
            transcription = self.transcription_model.transcribe(audio_data)
        self._send_ros2_message(transcription, "/from_human")
        self.transcription_threads[identifier]["transcription"] = transcription
        self.transcription_threads[identifier]["event"].set()

    def _send_ros2_message(self, data: str, topic: str):
        self.logger.debug(f"Sending message to {topic}: {data}")
        if topic == "/voice_commands":
            msg = ROS2Message(payload={"data": data})
            try:
                self.ros2_connector.send_message(
                    msg, topic, msg_type="std_msgs/msg/String"
                )
            except Exception as e:
                self.logger.error(f"Error sending message to {topic}: {e}")
        else:
            msg = ROS2HRIMessage(
                text=data,
                message_author="human",
                communication_id=ROS2HRIMessage.generate_conversation_id(),
            )
            self.ros2_hri_connector.send_message(msg, topic)
