import logging

from rai.agents import VoiceRecognitionAgent
from rai.communication import AudioInputDeviceConfig
from rai_asr.models import LocalWhisper, OpenWakeWord, SileroVAD

logging.basicConfig(level=logging.DEBUG)

VAD_THRESHOLD = 0.1
OWW_THRESHOLD = 0.01
VAD_SAMPLING_RATE = 16000
DEV_ID = 12
DEVICE_SAMPLE_RATE = 44100
DEFAULT_BLOCKSIZE = 512


agent = VoiceRecognitionAgent()

vad = SileroVAD(
    sampling_rate=VAD_SAMPLING_RATE,
    threshold=VAD_THRESHOLD,
)

oww = OpenWakeWord(
    wake_word_model_path="hey jarvis",
    threshold=0.02,
)

whisper = LocalWhisper(
    model_name="tiny",
    sample_rate=VAD_SAMPLING_RATE,
    language="en",
)

microphone_configuration = AudioInputDeviceConfig(
    block_size=DEFAULT_BLOCKSIZE,
    consumer_sampling_rate=VAD_SAMPLING_RATE,
    target_sampling_rate=DEVICE_SAMPLE_RATE,
    dtype="int16",
    device_number=DEV_ID,
)


agent.setup(DEV_ID, microphone_configuration, whisper)
agent.add_detection_model(vad, "record")
agent.add_detection_model(oww, "record")
agent.add_detection_model(vad, "stop")

agent.run()
print("Press 'q' to quit: ")
x = 0
while x < 200000000000000:
    x += 1

agent.stop()
