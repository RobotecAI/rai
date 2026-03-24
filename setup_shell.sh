#!/usr/bin/env sh

if [ -f ".venv/bin/activate" ]; then
    # Suppress ShellCheck warning about not following external file
    # shellcheck disable=SC1091
    . ".venv/bin/activate"
else
    echo "Missing .venv. Run 'uv sync' first."
    if (return 0 2>/dev/null); then
        return 1
    fi
    exit 1
fi

# Suppress ShellCheck warning about not following external file
# shellcheck disable=SC1091

case "$SHELL" in
    *bash)
        . install/setup.bash
        echo "Sourced bash install"
        ;;
    *zsh)
        . install/setup.zsh
        echo "Sourced zsh install"
        ;;
    *fish)
        echo "fish is not supported"
        ;;
    *sh)
        . install/setup.sh
        echo "Sourced sh install."
        ;;
    *)
        echo "Unknown shell: $0"
        ;;
esac

export PYTHONPATH
PYTHON_SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
PYTHONPATH="${PYTHON_SITE_PACKAGES}:$PYTHONPATH"
PYTHONPATH="src/rai_core:$PYTHONPATH"
PYTHONPATH="src/rai_sim:$PYTHONPATH"
PYTHONPATH="src/rai_s2s:$PYTHONPATH"
PYTHONPATH="src/rai_bench:$PYTHONPATH"
