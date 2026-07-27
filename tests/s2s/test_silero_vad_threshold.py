# Copyright (C) 2026
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / "src/rai_s2s/rai_s2s/asr/models/silero_vad.py"
BASE_PATH = ROOT / "src/rai_s2s/rai_s2s/asr/models/base.py"


def _ensure_numpy():
    try:
        import numpy  # noqa: F401
        return
    except Exception:
        pass
    np = types.ModuleType("numpy")
    np.__path__ = []  # type: ignore
    npt = types.ModuleType("numpy.typing")
    npt.NDArray = object
    sys.modules["numpy"] = np
    sys.modules["numpy.typing"] = npt


def _load_silero_module():
    _ensure_numpy()
    torch = MagicMock()
    sys.modules["torch"] = torch
    for name in ["rai_s2s", "rai_s2s.asr", "rai_s2s.asr.models"]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []  # type: ignore
            sys.modules[name] = m
    spec_b = importlib.util.spec_from_file_location("base_under_test_silero", BASE_PATH)
    base = importlib.util.module_from_spec(spec_b)
    assert spec_b and spec_b.loader
    spec_b.loader.exec_module(base)
    sys.modules["rai_s2s.asr.models"].BaseVoiceDetectionModel = base.BaseVoiceDetectionModel
    # patch import target used by silero
    import rai_s2s.asr.models as models_pkg  # type: ignore
    models_pkg.BaseVoiceDetectionModel = base.BaseVoiceDetectionModel

    # Also satisfy "from rai_s2s.asr.models import BaseVoiceDetectionModel"
    # by ensuring submodule package has attribute (already)
    spec = importlib.util.spec_from_file_location("silero_vad_under_test", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Pre-insert parent modules so relative-style imports via package path work
    sys.modules["rai_s2s.asr.models.silero_vad"] = mod
    # Make import try find BaseVoiceDetectionModel - monkeypatch importlib
    import builtins
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rai_s2s.asr.models" or name.endswith("asr.models"):
            m = sys.modules["rai_s2s.asr.models"]
            return m
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
    try:
        spec.loader.exec_module(mod)
    finally:
        builtins.__import__ = real_import
    mod.torch = torch
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
