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
from geometry_msgs.msg import Point

from rai_semap.core.backend.sqlite_backend import SQLiteBackend

from .conftest import make_annotation


@pytest.fixture
def backend(temp_db_path):
    """Create a SQLiteBackend instance for testing."""
    backend = SQLiteBackend(temp_db_path)
    backend.init_schema()
    return backend


def test_backend_init_schema(backend):
    """Test that schema initialization completes without error."""
    backend.init_schema()


def test_backend_insert_annotation(backend):
    """Test inserting a single annotation."""
    annotation = make_annotation("test-id", "cup", 1.0, 2.0)
    annotation_id = backend.insert_annotation(annotation)
    assert annotation_id == "test-id"
    assert isinstance(annotation_id, str)


def test_backend_insert_and_query(backend):
    """Test inserting annotation and querying to verify persistence."""
    annotation = make_annotation("test-id-1", "cup", 1.0, 2.0)
    backend.insert_annotation(annotation)

    center = Point(x=1.0, y=2.0, z=0.0)
    results = backend.spatial_query(center, radius=0.5)
    assert len(results) == 1
    assert results[0].id == "test-id-1"
    assert results[0].object_class == "cup"
    assert results[0].confidence == 0.9
    assert results[0].pose.position.x == 1.0
    assert results[0].pose.position.y == 2.0
    assert results[0].location_id == "test_location"


def test_backend_query_by_class_filter(backend):
    """Test querying with object_class filter."""
    annotation1 = make_annotation("cup-1", "cup", 1.0, 1.0)
    annotation2 = make_annotation(
        "bottle-1", "bottle", 2.0, 2.0, confidence=0.8, timestamp=1234567891
    )

    backend.insert_annotation(annotation1)
    backend.insert_annotation(annotation2)

    center = Point(x=0.0, y=0.0, z=0.0)
    results = backend.spatial_query(
        center, radius=10.0, filters={"object_class": "cup"}
    )
    assert len(results) == 1
    assert results[0].object_class == "cup"
    assert results[0].id == "cup-1"


def test_backend_query_by_confidence_threshold(backend):
    """Test querying with confidence threshold filter."""
    annotation1 = make_annotation("high-conf", "cup", 1.0, 1.0, confidence=0.9)
    annotation2 = make_annotation(
        "low-conf", "cup", 2.0, 2.0, confidence=0.3, timestamp=1234567891
    )

    backend.insert_annotation(annotation1)
    backend.insert_annotation(annotation2)

    center = Point(x=0.0, y=0.0, z=0.0)
    results = backend.spatial_query(
        center, radius=10.0, filters={"confidence_threshold": 0.5}
    )
    assert len(results) == 1
    assert results[0].id == "high-conf"
    assert results[0].confidence >= 0.5


def test_backend_update_annotation(backend):
    """Test updating an existing annotation."""
    annotation = make_annotation("test-id", "cup", 1.0, 2.0, confidence=0.7)
    backend.insert_annotation(annotation)

    updated_annotation = make_annotation(
        "test-id",
        "cup",
        1.5,
        2.5,
        confidence=0.95,
        timestamp=1234567900,
        detection_source="GroundedSAM",
    )
    success = backend.update_annotation(updated_annotation)
    assert success is True

    center = Point(x=1.0, y=2.0, z=0.0)
    results = backend.spatial_query(center, radius=1.0)
    assert len(results) == 1
    assert results[0].confidence == 0.95
    assert results[0].pose.position.x == 1.5
    assert results[0].detection_source == "GroundedSAM"


def test_backend_delete_annotation(backend):
    """Test deleting an annotation."""
    annotation = make_annotation("test-id", "cup", 1.0, 2.0)
    backend.insert_annotation(annotation)

    success = backend.delete_annotation("test-id")
    assert success is True

    center = Point(x=1.0, y=2.0, z=0.0)
    results = backend.spatial_query(center, radius=1.0)
    assert len(results) == 0

    success = backend.delete_annotation("non-existent")
    assert success is False


def test_backend_spatial_query(backend):
    """Test spatial query returns list of annotations."""
    center = Point(x=0.0, y=0.0, z=0.0)
    radius = 1.0
    results = backend.spatial_query(center, radius)
    assert isinstance(results, list)
    assert len(results) == 0
