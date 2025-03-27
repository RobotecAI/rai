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

import json
from pathlib import Path
from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration.

    Returns
    -------
    Dict[str, Any]
        The default configuration dictionary.
    """
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def get_default_identity() -> Dict[str, Any]:
    """Get the default identity configuration.

    Returns
    -------
    Dict[str, Any]
        The default identity configuration dictionary.
    """
    identity_path = Path(__file__).parent / "identity.json"
    with open(identity_path, "r") as f:
        return json.load(f)


def get_default_constitution() -> Dict[str, Any]:
    """Get the default constitution configuration.

    Returns
    -------
    Dict[str, Any]
        The default constitution configuration dictionary.
    """
    constitution_path = Path(__file__).parent / "constitution.json"
    with open(constitution_path, "r") as f:
        return json.load(f)


def get_default_vector_db() -> Dict[str, Any]:
    """Get the default vector database configuration.

    Returns
    -------
    Dict[str, Any]
        The default vector database configuration dictionary.
    """
    vector_db_path = Path(__file__).parent / "vector_db.json"
    with open(vector_db_path, "r") as f:
        return json.load(f)
