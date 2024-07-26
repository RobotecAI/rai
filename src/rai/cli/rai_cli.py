import argparse
import pathlib
import subprocess
from pathlib import Path


def create_robot_package():
    parser = argparse.ArgumentParser(description="Creation of a robot package.")
    parser.add_argument("--name", type=str, required=True, help="Package's name")
    parser.add_argument(
        "--destination-directory",
        type=str,
        required=False,
        default="src/",
        help="Package's destination",
    )
    args = parser.parse_args()
    package_name = args.name
    package_destination = args.destination_directory

    subprocess.run(
        [
            "ros2",
            "pkg",
            "create",
            "--build-type",
            "ament_python",
            f"{package_name}_whoami",
            "--destination-directory",
            f"{package_destination}/",
            "--license",
            "Apache-2.0",
        ]
    )

    package_path = Path(f"{package_destination}/{package_name}_whoami/description")
    package_path.mkdir(parents=True, exist_ok=True)

    (package_path / "documentation").mkdir(exist_ok=True)
    (package_path / "images").mkdir(exist_ok=True)
    (package_path / "robot_constitution.txt").touch()

    default_constitution_path = f"{pathlib.Path(__file__).parent.resolve()}/resources/default_robot_constitution.txt"
    with open(default_constitution_path, "r") as file:
        default_constitution = file.read()

    with open(f"{package_path}/robot_constitution.txt", "w") as file:
        file.write(default_constitution)


if __name__ == "__main__":
    create_robot_package()
