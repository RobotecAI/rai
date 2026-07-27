# Copyright (C) 2026
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/rai_s2s/rai_s2s/asr/models/base.py"


def _load_base():
    sys.modules.setdefault("numpy", MagicMock())
    sys.modules.setdefault("numpy.typing", MagicMock())
    for name in ["rai_s2s", "rai_s2s.asr", "rai_s2s.asr.models"]:
        sys.modules.setdefault(name, MagicMock())
    spec = importlib.util.spec_from_file_location("rai_s2s.asr.models.base", BASE)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_base_transcription_rejects_bad_sample_rate():
    mod = _load_base()

    class Dummy(mod.BaseTranscriptionModel):
        def transcribe(self, data):
            return ""

    with pytest.raises(ValueError, match="sample_rate"):
        Dummy("m", 0)
    with pytest.raises(ValueError, match="sample_rate"):
        Dummy("m", -16000)
    with pytest.raises(ValueError, match="model_name"):
        Dummy("  ", 16000)
    with pytest.raises(ValueError, match="language"):
        Dummy("m", 16000, language="")
