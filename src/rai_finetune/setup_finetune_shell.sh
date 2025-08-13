#!/usr/bin/env sh

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091
cd src/rai_finetune || exit
. "$(poetry env info --path)"/bin/activate

# go back to the root directory
cd - || exit

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091

export PYTHONPATH
PYTHONPATH="$(dirname "$(dirname "$(poetry run which python)")")/lib/python$(poetry run python --version | awk '{print $2}' | cut -d. -f1,2)/site-packages:$PYTHONPATH"
PYTHONPATH="src/rai_finetune:$PYTHONPATH"
