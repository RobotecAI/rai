import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from rai.cli.rai_cli import create_rai_ws


def test_create_rai_ws():
    with TemporaryDirectory() as directory:
        # Mock ArgumentParser and its methods
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.name = "test_package"
        mock_args.destination_directory = directory
        mock_parser.parse_args.return_value = mock_args

        # Patch argparse.ArgumentParser to return our mock
        with patch("argparse.ArgumentParser", return_value=mock_parser):
            create_rai_ws()

            whoami_directory = Path(directory) / "test_package_whoami"

            assert os.path.exists(whoami_directory), "Description folder is missing"

            description_files = os.listdir(whoami_directory / "description")
            assert "robot_constitution.txt" in description_files
