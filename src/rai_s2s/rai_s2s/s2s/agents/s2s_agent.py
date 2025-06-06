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
from abc import abstractmethod
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, List, Literal, Optional
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from rai.agents.base import BaseAgent
from rai.communication import HRIConnector, HRIMessage

from rai_s2s.asr.agents.asr_agent import ThreadData
from rai_s2s.asr.models import BaseTranscriptionModel, BaseVoiceDetectionModel
from rai_s2s.sound_device import (
    SoundDeviceConfig,
    SoundDeviceConnector,
    SoundDeviceMessage,
)
from rai_s2s.tts.agents.tts_agent import PlayData
from rai_s2s.tts.models.base import TTSModel


class SpeechToSpeechAgent(BaseAgent):
    def __init__(
        self,
        from_human_topic: str,
        to_human_topic: str,
        *,
        microphone_config: SoundDeviceConfig,
        speaker_config: SoundDeviceConfig,
        transcription_model: BaseTranscriptionModel,
        vad: BaseVoiceDetectionModel,
        tts: TTSModel,
        grace_period: float = 1.0,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__()
        if logger is not None:
            self.logger = logger
        self.sound_connector = SoundDeviceConnector(
            targets=[("speaker", speaker_config)],
            sources=[("microphone", microphone_config)],
        )

        self.from_human_topic = from_human_topic
        self.to_human_topic = to_human_topic

        sample_rate, _, out_channels = self.sound_connector.get_audio_params("speaker")
        tts.sample_rate = sample_rate
        tts.channels = out_channels

        self.playback_data = PlayData()

        self.should_record_pipeline: List[BaseVoiceDetectionModel] = []
        self.should_stop_pipeline: List[BaseVoiceDetectionModel] = []

        self.current_transcription_id = str(uuid4())[0:8]
        self.current_speech_id = None
        self.text_queues: dict[str, Queue] = {self.current_transcription_id: Queue()}

        self.terminate_agent = Event()

        self.audio_generating_thread = Thread(target=self._audio_gen_thread)
        self.audio_generating_thread.start()
        self.audio_queues: dict[str, Queue] = {self.current_transcription_id: Queue()}
        self.remembered_speech_ids: list[str] = []

        self.tts_model = tts

        self.transcription_model = transcription_model

        self.vad: BaseVoiceDetectionModel = vad
        self.grace_period = grace_period
        self.grace_period_start = 0

        self.sample_buffer = []
        self.sample_buffer_lock = Lock()
        self.transcription_lock = Lock()
        self.active_thread = ""
        self.transcription_threads: dict[str, ThreadData] = {}
        self.transcription_buffers: dict[str, list[NDArray]] = {}
        self.is_playing = False

        self.recording_started = False
        # self.ran_setup = False

        self.hri_connector: HRIConnector = self._setup_hri_connector()

        self.microphone_samples: Optional[np.ndarray] = None
        self.save_flag = False

    @abstractmethod
    def _setup_hri_connector(self) -> HRIConnector: ...

    def _audio_gen_thread(self):
        while not self.terminate_agent.wait(timeout=0.01):
            if self.current_transcription_id in self.text_queues:
                try:
                    data = self.text_queues[self.current_transcription_id].get(
                        block=False
                    )
                except Empty:
                    continue
                audio = self.tts_model.get_speech(data)
                try:
                    self.audio_queues[self.current_transcription_id].put(audio)
                except KeyError as e:
                    self.logger.error(
                        f"Could not find queue for {self.current_transcription_id}: queuse: {self.audio_queues.keys()}"
                    )
                    raise e

    def run(self):
        """
        Start the text-to-speech agent, initializing playback and launching the transcription thread.
        """
        self.running = True
        self.logger.info("Starting SpeechToSpeechAgent...")

        msg = SoundDeviceMessage(read=False)
        self.player_handle = self.sound_connector.start_action(
            action_data=msg,
            target="speaker",
            on_feedback=self._speaker_callback,
            on_done=lambda: None,
        )
        msg = SoundDeviceMessage(read=True)
        self.listener_handle = self.sound_connector.start_action(
            action_data=msg,
            target="microphone",
            on_feedback=self._on_microphone_sample,
            on_done=lambda: None,
        )
        self.logger.info("SpeechToSpeechAgent Started!")

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

    def _on_microphone_sample(self, indata: np.ndarray, status_flags: dict[str, Any]):
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
            self.set_playback_state("pause")
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
            self.set_playback_state("stop")
            self.is_playing = False
        elif not self.is_playing and (
            sample_time - self.grace_period_start > self.grace_period
        ):
            self.set_playback_state("play")
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
        with (
            self.transcription_lock
        ):  # this is only necessary for the local model... TODO: fix this somehow
            transcription = self.transcription_model.transcribe(audio_data)
        self._send_from_human_message(transcription)
        self.transcription_threads[identifier]["transcription"] = transcription
        self.transcription_threads[identifier]["event"].set()

    @abstractmethod
    def _send_from_human_message(self, data: str): ...

    def _on_to_human_message(self, message: HRIMessage):
        self.logger.info(f"Receieved message from human: {message.text}")
        self.logger.warning(
            f"Starting playback, current id: {self.current_transcription_id}"
        )
        if (
            self.current_speech_id is None
            and message.communication_id is not None
            and message.communication_id not in self.remembered_speech_ids
        ):
            self.current_speech_id = message.communication_id
            self.remembered_speech_ids.append(self.current_speech_id)
            if len(self.remembered_speech_ids) > 64:
                self.remembered_speech_ids.pop(0)
        if self.current_speech_id == message.communication_id:
            self.text_queues[self.current_transcription_id].put(message.text)
        self.playback_data.playing = True

    def add_detection_model(self, model: BaseVoiceDetectionModel):
        """
        Add a voice detection model to check before recording starts.

        Parameters
        ----------
        model : BaseVoiceDetectionModel
            The voice detection model to be added.
        """

        self.should_record_pipeline.append(model)

    def set_playback_state(self, state: Literal["play", "pause", "stop"]):
        """
        Set the playback state of the system.

        Parameters
        ----------
        state : {"play", "pause", "stop"}
            The desired playback state:
            - "play": Start or resume playback.
            - "pause": Pause the current playback.
            - "stop": Stop playback and reset playback-related data and queues.

        Notes
        -----
        - When state is "stop", this method:
          - Resets the `current_speech_id`.
          - Generates a new `current_transcription_id`.
          - Initializes new audio and text queues.
          - Clears previous playback data.
        - Logs actions and transitions for debugging and monitoring purposes.
        """
        if state == "play":
            self.playback_data.playing = True
        elif state == "pause":
            self.playback_data.playing = False
        elif state == "stop":
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

    def stop(self):
        """
        Clean exit the speech-to-speech agent, terminating playback and joining the transcription thread.
        """
        self.sound_connector.shutdown()

        self.logger.info("Stopping TextToSpeechAgent")
        self.terminate_agent.set()
        if self.audio_generating_thread is not None:
            self.audio_generating_thread.join()
