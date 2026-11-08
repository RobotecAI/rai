# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for model weight management and loading."""

import os
import shutil
import time
import urllib.request
from logging import Logger
from pathlib import Path
from typing import Callable, Optional

# Substrings of errors raised when a weights file is truncated or corrupted.
_CORRUPTION_MARKERS = (
    "PytorchStreamReader",  # torch.load on a damaged .pt/.pth zip archive
    "Error while deserializing header",  # safetensors
    "incomplete metadata",  # safetensors
    "SafetensorError",
)


def load_model_with_error_handling(
    model_class,
    weights_path: Path,
    logger: Logger,
    weights_url: str,
    config_path: str | Path | None = None,
    redownload: Optional[Callable[[], None]] = None,
):
    """Load model with automatic error handling for corrupted weights.

    Args:
        model_class: A class that can be instantiated with weights_path and optionally config_path
        weights_path: Path to model weights file or directory
        logger: Logger instance for error messages
        weights_url: URL to download weights from if corrupted
        config_path: Optional path to config file
        redownload: Optional callable that re-fetches the weights to ``weights_path``.
            Defaults to downloading ``weights_url`` as a single file.

    Returns:
        The loaded model instance

    Raises:
        Exception: If model loading fails (after retry)
    """

    def _instantiate():
        if config_path is not None:
            return model_class(weights_path, config_path=config_path)
        return model_class(weights_path)

    try:
        return _instantiate()
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        if any(marker in f"{type(e).__name__}: {e}" for marker in _CORRUPTION_MARKERS):
            logger.error("The weights might be corrupted. Redownloading...")
            remove_weights(weights_path)
            if redownload is not None:
                redownload()
            else:
                download_weights(weights_path, logger, weights_url)
            return _instantiate()
        raise e


def _log_progress(
    logger: Logger,
    downloaded: int,
    total: int,
    elapsed: float,
    bar_width: int = 30,
) -> None:
    pct = min(100, int(downloaded * 100 / total))
    filled = int(bar_width * pct / 100)
    bar = (
        "=" * filled + ">" + " " * (bar_width - filled - 1)
        if filled < bar_width
        else "=" * bar_width
    )
    speed = downloaded / elapsed if elapsed > 0 else 0
    eta = (total - downloaded) / speed if speed > 0 else 0
    logger.info(
        f"[{bar}] {pct:3d}%  {downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MB"
        f"  @ {speed / 1024**2:.1f} MB/s  ETA: {eta:.0f}s"
    )


def download_weights(
    weights_path: Path,
    logger: Logger,
    weights_url: str,
    timeout: int = 600,
    progress_interval_pct: int = 5,
) -> None:
    """Download model weights from URL.

    Uses urllib.request (no subprocess dependency) with atomic write via a
    .part temp file so a failed/interrupted download never leaves a partial
    weights file at the target path.

    Args:
        weights_path: Path where weights should be saved
        logger: Logger instance for status messages
        weights_url: URL to download weights from
        timeout: Socket timeout in seconds (default 600)
        progress_interval_pct: Log a progress bar every N percent (default 5)

    Raises:
        Exception: If download fails
    """
    logger.info(f"Downloading weights from {weights_url} to {weights_path}")
    tmp_path = weights_path.with_suffix(".part")
    try:
        with urllib.request.urlopen(weights_url, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length", 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded = 0
            last_logged_pct = -progress_interval_pct
            start_time = time.monotonic()
            with open(tmp_path, "wb") as f:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        elapsed = time.monotonic() - start_time
                        pct = min(100, int(downloaded * 100 / total))
                        if pct - last_logged_pct >= progress_interval_pct:
                            _log_progress(logger, downloaded, total, elapsed)
                            last_logged_pct = pct
        tmp_path.rename(weights_path)
        file_size = os.path.getsize(weights_path)
        logger.info(f"Successfully downloaded weights ({file_size / 1024**2:.2f} MB)")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        if weights_path.exists():
            weights_path.unlink()
        raise Exception(f"Could not download weights: {e}") from e


def download_hf_snapshot(
    local_dir: Path,
    logger: Logger,
    repo_id: str,
    revision: Optional[str] = None,
) -> None:
    """Download a Hugging Face Hub model repository into ``local_dir``.

    Only safetensors weights plus the JSON/text config, tokenizer and preprocessor files
    are fetched; pickle-based ``*.bin`` weights are skipped. The snapshot is written to a
    ``.part`` directory and renamed into place, so a failed download never leaves a
    partial model at ``local_dir``.

    Args:
        local_dir: Directory where the model files should end up
        logger: Logger instance for status messages
        repo_id: Hugging Face Hub repository id, e.g. "IDEA-Research/grounding-dino-tiny"
        revision: Optional git revision (commit sha, branch or tag) to pin

    Raises:
        Exception: If download fails
    """
    from huggingface_hub import snapshot_download

    logger.info(
        f"Downloading {repo_id}@{revision or 'main'} from the Hugging Face Hub to {local_dir}"
    )
    tmp_dir = local_dir.with_name(local_dir.name + ".part")
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=tmp_dir,
            allow_patterns=["*.json", "*.txt", "*.safetensors"],
        )
        if local_dir.exists():
            remove_weights(local_dir)
        tmp_dir.rename(local_dir)
        size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
        logger.info(f"Successfully downloaded {repo_id} ({size / 1024**2:.2f} MB)")
    except Exception as e:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise Exception(f"Could not download weights: {e}") from e


def remove_weights(weights_path: Path):
    """Remove a weights file or a downloaded model directory.

    Args:
        weights_path: Path to weights file or directory to remove
    """
    try:
        if Path(weights_path).is_dir():
            shutil.rmtree(weights_path)
        else:
            os.remove(weights_path)
    except FileNotFoundError:
        # File already removed, idempotent operation
        pass
