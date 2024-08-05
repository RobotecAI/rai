import os
import tempfile
from abc import abstractmethod
from typing import Optional

import requests
from elevenlabs.client import ElevenLabs


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
        self.voice = voice
        api_key = os.getenv(key="ELEVENLABS_API_KEY")
        self.client = ElevenLabs(base_url=None, api_key=api_key)

    def synthesize_speech_to_file(self, text: str) -> str:
        response = self.client.generate(
            text=text,
            voice=self.voice,
            optimize_streaming_latency=4,
        )
        audio_data = b"".join(response)
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
