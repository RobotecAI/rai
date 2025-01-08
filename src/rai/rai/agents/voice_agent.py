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
from typing import Any, List, Optional, cast

import numpy as np
from numpy.typing import NDArray
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai.agents.base import BaseAgent
from rai.communication import AudioInputDeviceConfig, StreamingAudioInputDevice
from rai_asr.models.base import BaseTranscriptionModel, BaseVoiceDetectionModel


class VoiceRecognitionAgent(BaseAgent):
    def __init__(
        self,
        logger: Optional[RcutilsLogger | logging.Logger] = None,
    ):
        super().__init__(connectors={"microphone": StreamingAudioInputDevice()})
        self.should_record_pipeline: List[BaseVoiceDetectionModel] = []
        self.should_stop_pipeline: List[BaseVoiceDetectionModel] = []
        self.transcription_lock = Lock()
        self.shared_samples = []
        self.recording_started = False
        self.ran_setup = False

        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

    def __call__(self):
        self.run()

    def setup(
        self,
        microphone_device_id: int,  # TODO: Change to name based instead of id based identification
        microphone_config: AudioInputDeviceConfig,
        transcription_model: BaseTranscriptionModel,
        grace_period: float = 0.5,
    ):
        self.connectors["microphone"] = cast(
            StreamingAudioInputDevice, self.connectors["microphone"]
        )
        self.microphone_device_id = str(microphone_device_id)
        self.connectors["microphone"].configure_device(
            target=self.microphone_device_id, config=microphone_config
        )
        self.transcription_model = transcription_model
        self.ran_setup = True
        self.running = False
        self.grace_period = grace_period
        self.grace_period_start = None
        self.start_transcription = Event()

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
            self.microphone_device_id, self.on_new_sample
        )
        self.transcription_thread = Thread(target=self._transcription_function)
        self.transcription_thread.start()

    def stop(self):
        self.running = False
        self.connectors["microphone"].terminate_action(self.listener_handle)
        self.transcription_thread.join()

    def on_new_sample(self, indata: np.ndarray, status_flags: dict[str, Any]):
        should_stop = self.should_stop_recording(indata)
        sample_time = time.time()
        if self.grace_period_start is None:
            self.grace_period_start = sample_time
        grace_period_elapsed = sample_time - self.grace_period_start > self.grace_period
        if self.recording_started:
            print(sample_time - self.grace_period_start)
        should_stop = should_stop if grace_period_elapsed else False
        if not self.recording_started:
            if self.should_start_recording(indata):
                self.recording_started = True
                self.grace_period_start = time.time()
                self.start_transcription.clear()
                print("Recording started")
            # self._logger.info("Recording started")
        if self.recording_started:
            # self._logger.info(f"Grace period elapsed: {grace_period_elapsed}")
            # self._logger.info(f"Should stop: {should_stop}")
            print(f"Grace period elapsed: {grace_period_elapsed}")
            print(f"Should stop: {should_stop}")
            if should_stop:
                # self._logger.info("Recording stopped")
                print("Recording stopped")
                self.start_transcription.set()
                self.recording_started = False
            else:
                # self._logger.info("Recording...")
                print("Recording...")
                with self.transcription_lock:
                    self.shared_samples.extend(indata)

    def should_start_recording(self, audio_data: NDArray[np.int16]) -> bool:
        output_parameters = {}
        should_listen = False
        for model in self.should_record_pipeline:
            should_listen, output_parameters = model.detected(
                audio_data, output_parameters
            )
            print(output_parameters)
            if not should_listen:
                return False
        return should_listen

    def should_stop_recording(self, audio_data: NDArray[np.int16]) -> bool:
        output_parameters = {}
        should_listen = False
        for model in self.should_stop_pipeline:
            should_listen, output_parameters = model.detected(
                audio_data, output_parameters
            )
            if should_listen:
                return False
        return not should_listen

    def _transcription_function(self):
        while self.running:
            time.sleep(0.1)
            # critical section for samples
            with self.transcription_lock:
                if len(self.shared_samples) == 0:
                    continue
                samples = np.array(self.shared_samples)
                self.shared_samples = []
            # end critical section for samples
            self.transcription_model.add_samples(samples)
            print(f"trascription to be run on {len(samples)} new samples")

            # if self.start_transcription.is_set():
            #     transcription = (
            #         self.transcription_model.transcribe()
            #     )  # TODO: this should be async
            #     self.on_transcription(transcription)

    def on_transcription(self, transcription: str):
        print(transcription)
