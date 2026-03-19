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
import time
import urllib.request
from logging import Logger
from pathlib import Path


def load_model_with_error_handling(
    model_class,
    weights_path: Path,
    logger: Logger,
    weights_url: str,
    config_path: str | Path | None = None,
):
    """Load model with automatic error handling for corrupted weights.

    Args:
        model_class: A class that can be instantiated with weights_path and optionally config_path
        weights_path: Path to model weights file
        logger: Logger instance for error messages
        weights_url: URL to download weights from if corrupted
        config_path: Optional path to config file

    Returns:
        The loaded model instance

    Raises:
        RuntimeError: If model loading fails (after retry)
    """
    try:
        if config_path is not None:
            return model_class(weights_path, config_path=config_path)
        else:
            return model_class(weights_path)
    except RuntimeError as e:
        logger.error(f"Could not load model: {e}")
        if "PytorchStreamReader" in str(e):
            logger.error("The weights might be corrupted. Redownloading...")
            remove_weights(weights_path)
            download_weights(weights_path, logger, weights_url)
            if config_path is not None:
                return model_class(weights_path, config_path=config_path)
            else:
                return model_class(weights_path)
        else:
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


def remove_weights(weights_path: Path):
    """Remove weights file.

    Args:
        weights_path: Path to weights file to remove
    """
    try:
        os.remove(weights_path)
    except FileNotFoundError:
        # File already removed, idempotent operation
        pass
