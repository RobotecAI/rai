#!/usr/bin/env bash
# Directory where the scripts are located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Array to store PIDs of background processes
declare -a PIDS

# Function to run a script with the given arguments
run_script() {
    local script="$1"
    shift
    python3 "$script" "$@" &
    # Store the PID of the last background process
    PIDS+=($!)
}

# Function to handle Ctrl+C (SIGINT)
handle_sigint() {
    echo -e "\nReceived SIGINT, forwarding to all running Python processes..."

    # Send SIGINT to all child processes
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Sending SIGINT to process $pid"
            kill -SIGINT "$pid"
        fi
    done

    echo "Waiting for all processes to exit..."
    wait

    echo "All processes have exited. Cleaning up and exiting."
    exit 0
}

# Main logic
main() {
    # Set up trap for SIGINT (Ctrl+C)
    trap handle_sigint SIGINT

    # Find all Python scripts in the scripts directory
    mapfile -t scripts < <(find "$SCRIPT_DIR" -name "*.py")

    # If no scripts found, exit
    if [ ${#scripts[@]} -eq 0 ]; then
        echo "No Python scripts found in $SCRIPT_DIR"
        exit 1
    fi

    echo "Found ${#scripts[@]} Python scripts in $SCRIPT_DIR"

    # Run all scripts in parallel with all arguments properly quoted
    for script in "${scripts[@]}"; do
        run_script "$script" "$@"
    done

    echo "All scripts are running in the background. Press Ctrl+C to stop them."

    # Wait for all background processes to finish
    wait

    echo "All scripts completed successfully."
}

# Call main with all arguments properly quoted
main "$@"
