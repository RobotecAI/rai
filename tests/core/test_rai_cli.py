import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rai.cli.rai_cli import create_rai_ws


@pytest.fixture
def rai_ws():
    directory = Path("/tmp/" + str(uuid.uuid4()))
    os.mkdir(directory)
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.name = "test_package"
    mock_args.destination_directory = directory
    mock_parser.parse_args.return_value = mock_args

    # Patch argparse.ArgumentParser to return our mock
    with patch("argparse.ArgumentParser", return_value=mock_parser):
        create_rai_ws()

    return directory


def test_create_rai_ws(rai_ws: Path):
    whoami_directory = Path(rai_ws) / "test_package_whoami"

    assert os.path.exists(whoami_directory), "Description folder is missing"

    description_files = os.listdir(whoami_directory / "description")
    assert "robot_constitution.txt" in description_files
