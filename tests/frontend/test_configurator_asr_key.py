"""Regression: ASR vendor select must read the same config key the on_change writer uses."""
from pathlib import Path

def test_configurator_reads_transcription_model_key():
    text = Path("src/rai_core/rai/frontend/configurator.py").read_text()
    assert "\"transciption_model\"" not in text
    assert "config[\"asr\"][\"transcription_model\"]" in text
    assert "\"transcription_model\", TRANSCRIBE_MODELS[0]" in text
