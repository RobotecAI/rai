# Copyright (C) 2026
import importlib.util
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / "src/rai_s2s/rai_s2s/asr/models/silero_vad.py"


def _load_silero_module():
    # Stub heavy package imports used only at module level.
    sys.modules.setdefault("torch", MagicMock())
    sys.modules.setdefault("numpy", MagicMock())
    # base model path
    base_path = ROOT / "src/rai_s2s/rai_s2s/asr/models/base.py"
    # Ensure package stubs
    for name in [
        "rai_s2s",
        "rai_s2s.asr",
        "rai_s2s.asr.models",
    ]:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
    # Load real base
    spec_b = importlib.util.spec_from_file_location("rai_s2s.asr.models.base", base_path)
    base = importlib.util.module_from_spec(spec_b)
    assert spec_b and spec_b.loader
    # Minimal stubs for base imports
    sys.modules["numpy.typing"] = MagicMock()
    spec_b.loader.exec_module(base)
    sys.modules["rai_s2s.asr.models.base"] = base
    sys.modules["rai_s2s.asr.models"].BaseVoiceDetectionModel = base.BaseVoiceDetectionModel

    spec = importlib.util.spec_from_file_location(
        "rai_s2s.asr.models.silero_vad", MOD_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_silero_vad_rejects_bad_threshold_before_hub_load():
    mod = _load_silero_module()

    def boom(*a, **k):
        raise AssertionError("torch.hub.load should not run for invalid threshold")

    with patch.object(mod.torch.hub, "load", side_effect=boom):
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=0)
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=float("nan"))
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=1.5)
