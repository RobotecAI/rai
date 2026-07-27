# Copyright (C) 2026
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/rai_s2s/rai_s2s/asr/models/base.py"


def _load_base():
    # Prefer real numpy; otherwise install minimal package stubs with _typing.
    try:
        import numpy  # noqa: F401
    except Exception:
        np = types.ModuleType("numpy")
        npt = types.ModuleType("numpy.typing")
        npt.NDArray = object
        sys.modules["numpy"] = np
        sys.modules["numpy.typing"] = npt
        # package mark
        np.__path__ = []  # type: ignore
    for name in ["rai_s2s", "rai_s2s.asr", "rai_s2s.asr.models"]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []  # type: ignore
            sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("rai_s2s_asr_models_base_under_test", BASE)
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
