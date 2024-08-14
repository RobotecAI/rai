#!/usr/bin/env sh

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091
. install/setup.bash

export PYTHONPATH
PYTHONPATH="$(dirname "$(dirname "$(poetry run which python)")")/lib/python$(poetry run python --version | awk '{print $2}' | cut -d. -f1,2)/site-packages:$PYTHONPATH"
