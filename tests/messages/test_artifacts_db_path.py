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

from pathlib import Path

from rai.messages.artifacts import get_stored_artifacts, store_artifacts


def test_store_artifacts_honors_db_path(tmp_path: Path):
    db = tmp_path / "custom_artifacts.pkl"
    store_artifacts("tc1", ["a"], db_path=str(db))
    assert db.is_file()
    assert get_stored_artifacts("tc1", db_path=str(db)) == ["a"]
    store_artifacts("tc1", ["b"], db_path=str(db))
    assert get_stored_artifacts("tc1", db_path=str(db)) == ["a", "b"]
    assert get_stored_artifacts("missing", db_path=str(db)) == []


def test_store_artifacts_creates_file_at_path(tmp_path: Path):
    db = tmp_path / "nested" / "db.pkl"
    db.parent.mkdir(parents=True)
    store_artifacts("x", [1], db_path=str(db))
    assert get_stored_artifacts("x", db_path=str(db)) == [1]


def test_store_artifacts_rejects_empty_tool_call_id(tmp_path: Path):
    import pytest

    db = tmp_path / "db.pkl"
    for bad in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="tool_call_id must be a non-empty string"):
            store_artifacts(bad, ["a"], db_path=str(db))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="tool_call_id must be a non-empty string"):
            get_stored_artifacts(bad, db_path=str(db))  # type: ignore[arg-type]
