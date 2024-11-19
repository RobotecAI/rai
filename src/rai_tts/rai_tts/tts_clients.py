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
import os
import tempfile
import time
from abc import abstractmethod
from typing import Optional

import requests
from elevenlabs.client import ElevenLabs
from elevenlabs.types import Voice
from elevenlabs.types.voice_settings import VoiceSettings

logger = logging.getLogger(__name__)

TTS_TRIES = 5
TTS_RETRY_DELAY = 0.5


class TTSClient:
    @abstractmethod
    def synthesize_speech_to_file(self, text: str) -> str:
        pass

    @staticmethod
    def save_audio_to_file(audio_data: bytes, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_file_path = temp_audio_file.name

        return temp_file_path


class ElevenLabsClient(TTSClient):
    def __init__(self, voice: str, base_url: Optional[str] = None):
        self.base_url = base_url
        api_key = os.getenv(key="ELEVENLABS_API_KEY")
        self.client = ElevenLabs(base_url=None, api_key=api_key)

        self.voice_settings = VoiceSettings(
            stability=0.7,
            similarity_boost=0.5,
        )
        voices = self.client.voices.get_all().voices
        voice_id = next((v.voice_id for v in voices if v.name == voice), None)
        if voice_id is None:
            raise ValueError(f"Voice {voice} not found")
        self.voice = Voice(voice_id=voice_id, settings=self.voice_settings)

    def synthesize_speech_to_file(self, text: str) -> str:
        tries = 0
        while tries < TTS_TRIES:
            try:
                response = self.client.generate(
                    text=text,
                    voice=self.voice,
                    optimize_streaming_latency=4,
                )
                audio_data = b"".join(response)
                return self.save_audio_to_file(audio_data, suffix=".mp3")
            except Exception as e:
                logger.warn(f"Error occurred during synthesizing speech: {e}.")  # type: ignore
                tries += 1
                if tries == TTS_TRIES:
                    logger.error(
                        f"Failed to synthesize speech after {TTS_TRIES} tries. Creating empty audio file instead."
                    )
                time.sleep(TTS_RETRY_DELAY)

        audio_data = b""
        return self.save_audio_to_file(audio_data, suffix=".mp3")


class OpenTTSClient(TTSClient):
    def __init__(self, base_url: Optional[str] = None, voice: Optional[str] = None):
        self.base_url = base_url
        self.voice = voice

    def synthesize_speech_to_file(self, text: str) -> str:
        params = {
            "voice": self.voice,
            "text": text,
        }
        response = requests.get("http://localhost:5500/api/tts", params=params)
        response.raise_for_status()

        return self.save_audio_to_file(response.content, suffix=".wav")
