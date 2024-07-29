import argparse
import subprocess
from pathlib import Path


def create_rai_ws():
    parser = argparse.ArgumentParser(description="Creation of a robot package.")
    parser.add_argument("--name", type=str, required=True, help="Package's name")
    parser.add_argument(
        "--destination-directory",
        type=str,
        required=False,
        default="rai_ws/",
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

    default_constitution_path = "src/rai/cli/resources/default_robot_constitution.txt"
    with open(default_constitution_path, "r") as file:
        default_constitution = file.read()

    with open(f"{package_path}/robot_constitution.txt", "w") as file:
        file.write(default_constitution)

    # Modify setup.py file
    setup_py_path = Path(f"{package_destination}/{package_name}_whoami/setup.py")
    modify_setup_py(setup_py_path)


# TODO: Refactor this hacky solution
# NOTE (mkotynia) fast solution, worth refactor in the future and testing if it generic for all setup.py file confgurations
def modify_setup_py(setup_py_path):
    with open(setup_py_path, "r") as file:
        setup_content = file.read()

    setup_content = "import glob\nimport os\n\n" + setup_content
    # Find the position to insert the new function
    insert_pos = setup_content.find("setup(")

    # Split the script to insert the new function
    part1 = setup_content[:insert_pos]
    part2 = setup_content[insert_pos:]

    # Adding the new line in the data_files section
    data_files_insert = """
        (os.path.join("share", package_name, os.path.dirname(p)), [p])
        for p in glob.glob("description/**/*", recursive=True)
        if os.path.isfile(p)
    ]
    + ["""

    # Finding the position to insert the new line in the data_files section
    data_files_pos = part2.find("data_files=[\n") + len("data_files=[\n")

    # Splitting the part2 to insert the new line in the data_files section
    part2_head = part2[:data_files_pos]
    part2_tail = part2[data_files_pos:]

    # Reconstruct the modified script
    modified_script = part1 + part2_head + data_files_insert + "\n" + part2_tail

    with open(setup_py_path, "w") as file:
        file.write(modified_script)


if __name__ == "__main__":
    create_rai_ws()
