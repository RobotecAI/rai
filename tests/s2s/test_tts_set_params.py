import pytest
from typing import Tuple

from rai_s2s.tts.models.base import TTSModel


class _DummyTTS(TTSModel):
    def get_speech(self, text: str):
        raise NotImplementedError

    def get_tts_params(self) -> Tuple[int, int]:
        return self.sample_rate, self.channels


def test_set_tts_params_accepts_positive():
    m = _DummyTTS()
    m.set_tts_params(16000, 1)
    assert m.get_tts_params() == (16000, 1)


@pytest.mark.parametrize("rate", [0, -1])
def test_set_tts_params_rejects_bad_rate(rate):
    m = _DummyTTS()
    with pytest.raises(ValueError, match="target_sample_rate"):
        m.set_tts_params(rate, 1)


@pytest.mark.parametrize("channels", [0, -2])
def test_set_tts_params_rejects_bad_channels(channels):
    m = _DummyTTS()
    with pytest.raises(ValueError, match="channels"):
        m.set_tts_params(16000, channels)
