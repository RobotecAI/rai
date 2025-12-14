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

from unittest.mock import MagicMock

import pytest
from geometry_msgs.msg import Point

from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticAnnotation, SemanticMapMemory

from .conftest import TEST_LOCATION_ID, make_annotation, make_pose


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=SQLiteBackend)
    return backend


@pytest.fixture
def memory(mock_backend):
    """Create a SemanticMapMemory instance with mock backend."""
    return SemanticMapMemory(mock_backend, location_id=TEST_LOCATION_ID)


@pytest.fixture
def real_backend(temp_db_path):
    """Create a real SQLiteBackend instance for integration testing."""
    backend = SQLiteBackend(temp_db_path)
    backend.init_schema()
    return backend


@pytest.fixture
def real_memory(real_backend):
    """Create a SemanticMapMemory instance with real backend."""
    return SemanticMapMemory(real_backend, location_id=TEST_LOCATION_ID)


def test_memory_store_annotation(memory):
    """Test storing an annotation returns an ID."""
    memory.backend.insert_annotation.return_value = "test-id"
    annotation_id = memory.store_annotation(
        object_class="cup",
        pose=make_pose(1.0, 2.0),
        confidence=0.9,
        timestamp=1234567890,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )
    assert annotation_id == "test-id"
    assert isinstance(annotation_id, str)
    memory.backend.insert_annotation.assert_called_once()
    call_args = memory.backend.insert_annotation.call_args[0][0]
    assert isinstance(call_args, SemanticAnnotation)
    assert call_args.object_class == "cup"
    assert call_args.location_id == TEST_LOCATION_ID


def test_memory_query_by_class(memory):
    """Test querying by object class returns list."""
    mock_annotation = make_annotation("test-id", "cup", 0.0, 0.0)
    memory.backend.spatial_query.return_value = [mock_annotation]
    results = memory.query_by_class("cup")
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].object_class == "cup"
    memory.backend.spatial_query.assert_called_once()
    call_kwargs = memory.backend.spatial_query.call_args[1]
    assert call_kwargs["filters"]["object_class"] == "cup"
    assert call_kwargs["filters"]["location_id"] == TEST_LOCATION_ID


def test_memory_query_by_location(memory):
    """Test querying by location returns list."""
    mock_annotation = make_annotation("test-id", "cup", 0.0, 0.0)
    memory.backend.spatial_query.return_value = [mock_annotation]
    center = Point(x=1.0, y=2.0, z=0.0)
    results = memory.query_by_location(center, radius=1.0)
    assert isinstance(results, list)
    assert len(results) == 1
    memory.backend.spatial_query.assert_called_once()
    call_args = memory.backend.spatial_query.call_args[0]
    assert call_args[0] == center
    assert call_args[1] == 1.0


def test_memory_query_by_region(memory):
    """Test querying by region returns list."""
    mock_annotation = make_annotation("test-id", "cup", 0.0, 0.0)
    memory.backend.spatial_query.return_value = [mock_annotation]
    bbox = (0.0, 0.0, 2.0, 2.0)
    results = memory.query_by_region(bbox)
    assert isinstance(results, list)
    memory.backend.spatial_query.assert_called_once()


def test_memory_get_map_metadata(memory):
    """Test getting map metadata returns MapMetadata."""
    memory.backend.spatial_query.return_value = []
    metadata = memory.get_map_metadata()
    assert metadata is not None
    assert metadata.location_id == TEST_LOCATION_ID
    assert metadata.map_frame_id == "map"


def test_memory_store_or_update_new_annotation(real_memory):
    """Test store_or_update_annotation creates new annotation when none nearby."""
    annotation_id = real_memory.store_or_update_annotation(
        object_class="cup",
        pose=make_pose(1.0, 2.0),
        confidence=0.9,
        timestamp=1234567890,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )
    assert isinstance(annotation_id, str)
    assert len(annotation_id) > 0

    center = Point(x=1.0, y=2.0, z=0.0)
    results = real_memory.query_by_location(center, radius=0.5)
    assert len(results) == 1
    assert results[0].object_class == "cup"
    assert results[0].confidence == 0.9


def test_memory_store_or_update_merges_nearby(real_memory):
    """Test store_or_update_annotation merges nearby duplicate detections."""
    annotation_id1 = real_memory.store_or_update_annotation(
        object_class="cup",
        pose=make_pose(1.0, 2.0),
        confidence=0.7,
        timestamp=1234567890,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )

    annotation_id2 = real_memory.store_or_update_annotation(
        object_class="cup",
        pose=make_pose(1.1, 2.1),
        confidence=0.9,
        timestamp=1234567900,
        detection_source="GroundedSAM",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
        merge_threshold=0.5,
    )

    assert annotation_id1 == annotation_id2

    center = Point(x=1.0, y=2.0, z=0.0)
    results = real_memory.query_by_location(center, radius=0.5)
    assert len(results) == 1
    assert results[0].confidence == 0.9
    assert results[0].detection_source == "GroundedSAM"


def test_memory_store_or_update_creates_separate_for_different_classes(real_memory):
    """Test store_or_update_annotation creates separate annotations for different classes."""
    annotation_id1 = real_memory.store_or_update_annotation(
        object_class="cup",
        pose=make_pose(1.0, 2.0),
        confidence=0.9,
        timestamp=1234567890,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )

    annotation_id2 = real_memory.store_or_update_annotation(
        object_class="bottle",
        pose=make_pose(1.1, 2.1),
        confidence=0.8,
        timestamp=1234567900,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
        merge_threshold=0.5,
    )

    assert annotation_id1 != annotation_id2

    center = Point(x=1.0, y=2.0, z=0.0)
    results = real_memory.query_by_location(center, radius=0.5)
    assert len(results) == 2


def test_memory_end_to_end_store_and_query(real_memory):
    """Test end-to-end: store multiple annotations and query by class."""
    real_memory.store_annotation(
        object_class="cup",
        pose=make_pose(1.0, 1.0),
        confidence=0.9,
        timestamp=1234567890,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )

    real_memory.store_annotation(
        object_class="bottle",
        pose=make_pose(2.0, 2.0),
        confidence=0.8,
        timestamp=1234567891,
        detection_source="GroundingDINO",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )

    real_memory.store_annotation(
        object_class="cup",
        pose=make_pose(3.0, 3.0),
        confidence=0.85,
        timestamp=1234567892,
        detection_source="GroundedSAM",
        source_frame="camera_frame",
        location_id=TEST_LOCATION_ID,
    )

    cups = real_memory.query_by_class("cup")
    assert len(cups) == 2
    assert all(c.object_class == "cup" for c in cups)

    bottles = real_memory.query_by_class("bottle")
    assert len(bottles) == 1
    assert bottles[0].object_class == "bottle"
