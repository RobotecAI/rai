# Copyright (C) 2025 Julia Jia
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

import pytest
from rai_perception.algorithms.boxer import GDBoxer
from rai_perception.models.detection import get_model, list_available_models


def test_get_model_valid():
    """Test getting valid detection model."""
    AlgorithmClass, config_path = get_model("grounding_dino")

    assert AlgorithmClass is GDBoxer
    assert config_path.endswith("gdino_config.py")


def test_get_model_invalid_name():
    """Test getting invalid model name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown detection model"):
        get_model("invalid_model")


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()

    assert isinstance(models, list)
    assert "grounding_dino" in models
