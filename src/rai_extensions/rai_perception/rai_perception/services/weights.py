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
import subprocess
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


def download_weights(weights_path: Path, logger: Logger, weights_url: str):
    """Download model weights from URL.

    Args:
        weights_path: Path where weights should be saved
        logger: Logger instance for status messages
        weights_url: URL to download weights from

    Raises:
        Exception: If download fails
    """
    logger.info(f"Downloading weights from {weights_url} to {weights_path}")
    try:
        subprocess.run(
            [
                "wget",
                weights_url,
                "-O",
                str(weights_path),
                "--progress=dot:giga",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not os.path.exists(weights_path):
            raise Exception(f"Downloaded file not found at {weights_path}")
        file_size = os.path.getsize(weights_path)
        if file_size < 1024 * 1024:
            raise Exception(
                f"Downloaded file is too small ({file_size} bytes), expected > 1MB"
            )
        logger.info(
            f"Successfully downloaded weights ({file_size / (1024 * 1024):.2f} MB)"
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout if e.stdout else str(e)
        logger.error(f"wget failed: {error_msg}")
        if os.path.exists(weights_path):
            os.remove(weights_path)
        raise Exception(f"Could not download weights: {error_msg}")
    except Exception as e:
        logger.error(f"Could not download weights: {e}")
        if os.path.exists(weights_path):
            os.remove(weights_path)
        raise


def remove_weights(weights_path: Path):
    """Remove weights file.

    Args:
        weights_path: Path to weights file to remove
    """
    os.remove(weights_path)
