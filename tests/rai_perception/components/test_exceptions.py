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

from rai_perception.components.exceptions import (
    PerceptionAlgorithmError,
    PerceptionError,
    PerceptionValidationError,
)


def test_perception_error_is_base_exception():
    """Test that PerceptionError is a base exception."""
    error = PerceptionError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"


def test_perception_algorithm_error_basic():
    """Test PerceptionAlgorithmError with minimal parameters."""
    error = PerceptionAlgorithmError(algorithm_stage="ransac")
    assert isinstance(error, PerceptionError)
    assert error.algorithm_stage == "ransac"
    assert error.strategy is None
    assert "ransac" in str(error)


def test_perception_algorithm_error_with_strategy():
    """Test PerceptionAlgorithmError with strategy."""
    error = PerceptionAlgorithmError(
        algorithm_stage="filtering", strategy="aggressive_outlier_removal"
    )
    assert error.algorithm_stage == "filtering"
    assert error.strategy == "aggressive_outlier_removal"
    assert "filtering" in str(error)
    assert "aggressive_outlier_removal" in str(error)


def test_perception_algorithm_error_with_suggestion():
    """Test PerceptionAlgorithmError with suggestion."""
    error = PerceptionAlgorithmError(
        algorithm_stage="estimation",
        strategy="top_plane",
        suggestion="Try strategy='centroid'",
    )
    assert error.suggestion == "Try strategy='centroid'"
    assert "Try strategy='centroid'" in str(error)


def test_perception_algorithm_error_with_input_info():
    """Test PerceptionAlgorithmError with input info."""
    error = PerceptionAlgorithmError(
        algorithm_stage="ransac",
        input_info={"point_count": 100, "noise_level": "high"},
    )
    assert error.input_info == {"point_count": 100, "noise_level": "high"}


def test_perception_validation_error_basic():
    """Test PerceptionValidationError with minimal parameters."""
    error = PerceptionValidationError(validation_rule="min_points", input_value=5)
    assert isinstance(error, PerceptionError)
    assert error.validation_rule == "min_points"
    assert error.input_value == 5
    assert "min_points" in str(error)


def test_perception_validation_error_with_range():
    """Test PerceptionValidationError with valid range."""
    error = PerceptionValidationError(
        validation_rule="threshold",
        input_value=1.5,
        valid_range="0.0 to 1.0",
    )
    assert error.valid_range == "0.0 to 1.0"
    assert "0.0 to 1.0" in str(error)


def test_perception_validation_error_with_suggestion():
    """Test PerceptionValidationError with suggestion."""
    error = PerceptionValidationError(
        validation_rule="point_count",
        input_value=3,
        suggestion="Increase min_points to at least 10",
    )
    assert error.suggestion == "Increase min_points to at least 10"
    assert "Increase min_points to at least 10" in str(error)
