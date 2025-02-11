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


import logging
import time
from threading import Event, Lock, Thread
from typing import Any, List, Optional, TypedDict
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from rai.agents.base import BaseAgent
from rai.communication import (
    ROS2ARIConnector,
    ROS2ARIMessage,
    SoundDeviceConfig,
    SoundDeviceConnector,
    SoundDeviceMessage,
)
from rai_asr.models import BaseTranscriptionModel, BaseVoiceDetectionModel


class ThreadData(TypedDict):
    thread: Thread
    event: Event
    transcription: str
    joined: bool


class VoiceRecognitionAgent(BaseAgent):
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
        microphone = SoundDeviceConnector(
            targets=[], sources=[("microphone", microphone_config)]
        )
        ros2_connector = ROS2ARIConnector(ros2_name)
        super().__init__(connectors={"microphone": microphone, "ros2": ros2_connector})
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

    def __call__(self):
        self.run()

    def add_detection_model(
        self, model: BaseVoiceDetectionModel, pipeline: str = "record"
    ):
        if pipeline == "record":
            self.should_record_pipeline.append(model)
        elif pipeline == "stop":
            self.should_stop_pipeline.append(model)
        else:
            raise ValueError("Pipeline should be either 'record' or 'stop'")

    def run(self):
        self.running = True
        assert isinstance(self.connectors["microphone"], SoundDeviceConnector)
        msg = SoundDeviceMessage(read=True)
        self.listener_handle = self.connectors["microphone"].start_action(
            action_data=msg,
            target="microphone",
            on_feedback=self.on_new_sample,
            on_done=lambda: None,
        )

    def stop(self):
        self.logger.info("Stopping voice agent")
        self.running = False
        self.connectors["microphone"].terminate_action(self.listener_handle)
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

    def on_new_sample(self, indata: np.ndarray, status_flags: dict[str, Any]):
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
        should_record = False
        # TODO: second condition is temporary
        if voice_detected and not self.recording_started:
            should_record = self.should_record(indata, output_parameters)

        if should_record:
            self.logger.info("starting recording...")
            self.recording_started = True
            thread_id = str(uuid4())[0:8]
            transcription_thread = Thread(
                target=self.transcription_thread,
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
        elif sample_time - self.grace_period_start > self.grace_period:
            self._send_ros2_message("play", "/voice_commands")

    def should_record(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> bool:
        for model in self.should_record_pipeline:
            detected, output = model(audio_data, input_parameters)
            self.logger.info(f"detected {detected}, output {output}")
            if detected:
                return True
        return False

    def transcription_thread(self, identifier: str):
        self.logger.info(f"transcription thread {identifier} started")
        audio_data = np.concatenate(self.transcription_buffers[identifier])
        with (
            self.transcription_lock
        ):  # this is only necessary for the local model... TODO: fix this somehow
            transcription = self.transcription_model.transcribe(audio_data)
        self._send_ros2_message(transcription, "/from_human")
        self.transcription_threads[identifier]["transcription"] = transcription
        self.transcription_threads[identifier]["event"].set()

    def _send_ros2_message(self, data: str, topic: str):
        self.connectors["ros2"].send_message(
            ROS2ARIMessage({"data": data}, {"msg_type": "std_msgs/msg/String"}),
            topic,
            msg_type="std_msgs/msg/String",
        )
