import numpy as np

from rai.communication import AudioInputDeviceConfig, StreamingAudioInputDevice
from rai_asr.models import SileroVAD

VAD_THRESHOLD = 0.1
OWW_THRESHOLD = 0.01
VAD_SAMPLING_RATE = 16000
DEV_ID = 12
DEVICE_SAMPLE_RATE = 44100
DEFAULT_BLOCKSIZE = 512


microphone_configuration = AudioInputDeviceConfig(
    block_size=DEFAULT_BLOCKSIZE,
    consumer_sampling_rate=VAD_SAMPLING_RATE,
    target_sampling_rate=DEVICE_SAMPLE_RATE,
    dtype="int16",
    device_number=DEV_ID,
)
vad = SileroVAD()

dev = StreamingAudioInputDevice()

dev.configure_device(str(DEV_ID), microphone_configuration)


def sample_callback(sample: np.ndarray, flag_dict):
    print("Sample received")
    print(vad.detected(sample, {}))


dev.start_action(str(DEV_ID), sample_callback)

while True:
    pass
