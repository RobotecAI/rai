#!/usr/bin/env bash

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091

cd src/rai_finetune || {
    echo "Error: Failed to change to src/rai_finetune directory" >&2
    exit 1
}
. "$(poetry env info --path)"/bin/activate

# go back to the root directory
cd - || {
    echo "Error: Failed to return to previous directory" >&2
    exit 1
}

export PYTHONPATH
PYTHONPATH="$(dirname "$(dirname "$(poetry run which python)")")/lib/python$(poetry run python --version | awk '{print $2}' | cut -d. -f1,2)/site-packages:$PYTHONPATH"
PYTHONPATH="src/rai_finetune:$PYTHONPATH"
