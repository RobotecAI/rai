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


from threading import Lock, Thread
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

from rai.agents.base import BaseAgent
from rai.communication import AudioInputDeviceConfig, StreamingAudioInputDevice
from rai_asr.models.base import BaseVoiceDetectionModel


class VoiceRecognitionAgent(BaseAgent):
    def __init__(self):
        super().__init__(connectors={"microphone": StreamingAudioInputDevice()})
        self.should_record_pipeline: List[BaseVoiceDetectionModel] = []
        self.should_stop_pipeline: List[BaseVoiceDetectionModel] = []
        self.transcription_lock = Lock()
        self.shared_samples = []
        self.recording_started = False
        self.ran_setup = False

    def __call__(self):
        self.run()

    def setup(
        self, microphone_device_id: int, microphone_config: AudioInputDeviceConfig
    ):
        assert isinstance(self.connectors["microphone"], StreamingAudioInputDevice)
        self.microphone_device_id = str(microphone_device_id)
        self.connectors["microphone"].configure_device(
            target=self.microphone_device_id, config=microphone_config
        )
        self.ran_setup = True

    def run(self):
        self.listener_handle = self.connectors["microphone"].start_action(
            self.microphone_device_id, self.on_new_sample
        )
        self.transcription_thread = Thread(target=self._transcription_function)
        self.transcription_thread.start()

    def stop(self):
        self.connectors["microphone"].terminate_action(self.listener_handle)
        self.transcription_thread.join()

    def on_new_sample(self, indata: np.ndarray, status_flags: dict[str, Any]):
        should_stop, should_cancel = self.should_stop_recording(indata)
        print(indata)
        if should_cancel:
            self.cancel_task()
        if (self.recording_started and not should_stop) or (
            self.should_start_recording(indata)
        ):
            with self.transcription_lock:
                self.shared_samples.extend(indata)

    def should_start_recording(self, audio_data: NDArray[np.int16]) -> bool:
        output_parameters = {}
        for model in self.should_record_pipeline:
            should_listen, output_parameters = model.detected(
                audio_data, output_parameters
            )
            if not should_listen:
                return False
        return True

    def should_stop_recording(self, audio_data: NDArray[np.int16]) -> Tuple[bool, bool]:
        output_parameters = {}
        for model in self.should_stop_pipeline:
            should_listen, output_parameters = model.detected(
                audio_data, output_parameters
            )
            # TODO: Add handling output parametrs for checking if should cancel
            if should_listen:
                return False, False
        return True, False

    def _transcription_function(self):
        with self.transcription_lock:
            samples = np.array(self.shared_samples)
            print(samples)
            self.shared_samples = []
