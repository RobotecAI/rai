# Copyright (C) 2026
import math
from unittest.mock import patch

import pytest


def test_silero_vad_rejects_bad_threshold_before_hub_load():
    import rai_s2s.asr.models.silero_vad as mod

    def boom(*a, **k):
        raise AssertionError("torch.hub.load should not run for invalid threshold")

    with patch.object(mod.torch.hub, "load", side_effect=boom):
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=0)
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=-0.1)
        with pytest.raises(ValueError, match="threshold"):
            netto = float("nan")
            mod.SileroVAD(threshold=netto)
        with pytest.raises(ValueError, match="threshold"):
            mod.SileroVAD(threshold=1.5)
