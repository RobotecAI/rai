# Copyright (C) 2026 Robotec.AI
import pytest

from rai_s2s.asr.models.open_wake_word import OpenWakeWord


def test_open_wake_word_rejects_invalid_threshold(monkeypatch):
    monkeypatch.setattr(
        "rai_s2s.asr.models.open_wake_word.download_models", lambda: None
    )

    class _Boom:
        def __init__(self, *a, **k):
            raise AssertionError("OWWModel should not load on bad threshold")

    monkeypatch.setattr("rai_s2s.asr.models.open_wake_word.OWWModel", _Boom)

    for bad in (0, -0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="threshold"):
            OpenWakeWord(wake_word_model_path="unused.onnx", threshold=bad)


def test_open_wake_word_accepts_valid_threshold(monkeypatch):
    monkeypatch.setattr(
        "rai_s2s.asr.models.open_wake_word.download_models", lambda: None
    )

    class _Ok:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr("rai_s2s.asr.models.open_wake_word.OWWModel", _Ok)
    m = OpenWakeWord(wake_word_model_path="unused.onnx", threshold=0.5)
    assert m.threshold == 0.5
