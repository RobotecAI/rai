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
    AudioInputDeviceConfig,
    ROS2ARIConnector,
    ROS2ARIMessage,
    StreamingAudioInputDevice,
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
        microphone_device_id: int,  # TODO: Change to name based instead of id based identification
        microphone_config: AudioInputDeviceConfig,
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
        microphone = StreamingAudioInputDevice()
        microphone.configure_device(
            target=str(microphone_device_id), config=microphone_config
        )
        ros2_connector = ROS2ARIConnector(ros2_name)
        super().__init__(connectors={"microphone": microphone, "ros2": ros2_connector})
        self.microphone_device_id = str(microphone_device_id)
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
        self.listener_handle = self.connectors["microphone"].start_action(
            action_data=None,
            target=self.microphone_device_id,
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

        voice_detected, output_parameters = self.vad.detected(indata, {})
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

    def should_record(
        self, audio_data: NDArray, input_parameters: dict[str, Any]
    ) -> bool:
        for model in self.should_record_pipeline:
            detected, output = model.detected(audio_data, input_parameters)
            if detected:
                return True
        return False

    def transcription_thread(self, identifier: str):
        self.logger.info(f"transcription thread {identifier} started")
        audio_data = np.concatenate(self.transcription_buffers[identifier])
        with self.transcription_lock:  # this is only necessary for the local model... TODO: fix this somehow
            transcription = self.transcription_model.transcribe(audio_data)
        self.connectors["ros2"].send_message(
            ROS2ARIMessage(
                {"data": transcription}, {"msg_type": "std_msgs/msg/String"}
            ),
            "/from_human",
        )
        self.transcription_threads[identifier]["transcription"] = transcription
        self.transcription_threads[identifier]["event"].set()
