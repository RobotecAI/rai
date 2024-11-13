import logging
from typing import Callable, Literal

import numpy as np
import sounddevice as sd
import torch
from numpy.typing import NDArray
from openwakeword.model import Model as OWWModel
from openwakeword.utils import download_models

logger = logging.getLogger(__name__)


DEFAULT_BLOCKSIZE = 1024
VAD_SAMPLING_RATE = 16000


def instantiate_stream(
    device_number: int, callback: Callable[[NDArray[np.int16], int, int, str], None]
) -> sd.InputStream:
    sd.default.latency = ("low", "low")  # type: ignore
    device_sample_rate: int = sd.query_devices(device=device_number, kind="input")[
        "default_samplerate"
    ]  # type: ignore
    window_size_samples = int(
        DEFAULT_BLOCKSIZE * device_sample_rate / VAD_SAMPLING_RATE
    )
    return sd.InputStream(
        samplerate=device_sample_rate,
        channels=1,
        device=device_number,
        dtype="int16",
        blocksize=window_size_samples,
        callback=callback,
    )


def instantiate_vad_model() -> torch.nn.Module:
    model, _ = torch.hub.load(  # type: ignore
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
    )
    return model


def instantiate_oww_model(
    wake_word_model: str, inference_framework: Literal["onnx", "tflite"] = "onnx"
) -> OWWModel:
    download_models(model_names=[wake_word_model])
    oww_model = OWWModel(
        wakeword_models=[
            wake_word_model,
        ],
        inference_framework=inference_framework,
    )
    return oww_model
