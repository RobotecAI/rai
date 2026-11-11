#!/usr/bin/env bash
# Source this file from the repository root: `source setup_shell.sh`.
# shellcheck disable=SC2317
# Create the environment first with `scripts/dev_env.sh`.

if [ -z "$ROS_DISTRO" ]; then
    echo "Source /opt/ros/<distro>/setup.bash first." >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f .venv/bin/activate ]; then
    echo "Missing .venv. Run scripts/dev_env.sh first." >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
[ -f install/setup.bash ] && . install/setup.bash
# shellcheck disable=SC1091
. .venv/bin/activate
