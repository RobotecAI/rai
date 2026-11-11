#!/usr/bin/env bash
# Source ROS 2, the /rai colcon workspace and the venv, then run the command.
set -e
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
# shellcheck disable=SC1091
[ -f /rai/install/setup.bash ] && source /rai/install/setup.bash
# shellcheck disable=SC1091
[ -f /rai/.venv/bin/activate ] && source /rai/.venv/bin/activate
exec "$@"
