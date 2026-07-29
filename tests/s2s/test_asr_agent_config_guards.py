# Copyright (C) 2025 Robotec.AI
import math

import pytest

from rai_s2s.asr.agents.initialization import VADConfig, WWConfig


def test_vad_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError):
        VADConfig(threshold=0.0)
    with pytest.raises(ValueError):
        VADConfig(threshold=-0.1)
    with pytest.raises(ValueError):
        VADConfig(threshold=1.1)


def test_vad_rejects_non_positive_grace() -> None:
    with pytest.raises(ValueError):
        VADConfig(silence_grace_period=0.0)
    with pytest.raises(ValueError):
        VADConfig(silence_grace_period=float("nan"))


def test_vad_accepts_defaults() -> None:
    cfg = VADConfig()
    assert cfg.threshold == 0.5


def test_ww_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        WWConfig(threshold=0.0)
    with pytest.raises(ValueError):
        WWConfig(threshold=math.nan)


def test_ww_accepts_defaults() -> None:
    cfg = WWConfig()
    assert 0.0 < cfg.threshold <= 1.0
