#!/usr/bin/env bash

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091

cd src/rai_finetune || {
    echo "Error: Failed to change to src/rai_finetune directory" >&2
    exit 1
}

if [ -f ".venv/bin/activate" ]; then
    # Suppress ShellCheck warning about not following external file
    # shellcheck disable=SC1091
    . ".venv/bin/activate"
else
    echo "Missing src/rai_finetune/.venv. Run 'uv sync' in src/rai_finetune first." >&2
    exit 1
fi

# go back to the root directory
cd - || {
    echo "Error: Failed to return to previous directory" >&2
    exit 1
}

export PYTHONPATH
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
PYTHONPATH="${PYTHON_SITE_PACKAGES}:$PYTHONPATH"
PYTHONPATH="src/rai_finetune:$PYTHONPATH"
