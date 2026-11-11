#!/usr/bin/env bash
# Create the RAI development environment.
#
# rclpy and the other ROS 2 Python packages are built for the system interpreter and live
# in its site-packages, so the venv uses that interpreter and can see them.
# Usage: source /opt/ros/<distro>/setup.bash && scripts/dev_env.sh [uv sync args, e.g. --all-groups]
set -euo pipefail

if [ -z "${ROS_DISTRO:-}" ]; then
    echo "Source /opt/ros/<distro>/setup.bash first." >&2
    exit 1
fi

cd "$(dirname "$0")/.."
if [ ! -f .venv/pyvenv.cfg ] || ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
    uv venv --clear --system-site-packages --python /usr/bin/python3
fi
uv sync "$@"

if [ -f ros_deps.repos ] && [ ! -d src/rai_interfaces ] && ! ros2 interface package rai_interfaces >/dev/null 2>&1; then
    vcs import . < ros_deps.repos
fi
colcon build --symlink-install
echo "Done. Run: source setup_shell.sh"
