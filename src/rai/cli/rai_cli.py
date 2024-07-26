import argparse
import subprocess
from pathlib import Path

def create_robot_package():
    parser = argparse.ArgumentParser(description='Creation of a robot package.')
    parser.add_argument('--name', type=str, required=True, help='Package\'s name')
    args = parser.parse_args()
    package_name = args.name

    subprocess.run(['ros2', 'pkg', 'create', '--build-type', 'ament_python',  f'{package_name}_whoami', '--destination-directory', 'src/', '--license', 'Apache-2.0'])

    package_path = Path(f'src/{package_name}_whoami/resource')
    package_path.mkdir(parents=True, exist_ok=True)

    (package_path / 'documentation').mkdir(exist_ok=True)
    (package_path / 'images').mkdir(exist_ok=True)
    (package_path / 'robot_constitution.txt').touch()



if __name__ == '__main__':
    create_robot_package()