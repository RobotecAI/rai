#!/bin/bash
# Script to build and run the ROS2 Humble segfault reproduction

set -e

echo "=========================================="
echo "ROS2 Humble Segfault Reproduction"
echo "Issue: https://github.com/RobotecAI/rai/issues/759"
echo "=========================================="
echo ""

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.humble-repro -t ros2-humble-segfault-repro .

echo ""
echo "Running reproduction script in Docker..."
echo "=========================================="
echo ""

# Run the container
# Note: We expect this to potentially segfault
docker run --rm ros2-humble-segfault-repro

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 139 ]; then
    echo "SEGFAULT REPRODUCED! (exit code 139)"
    echo "This confirms the issue exists in ROS2 Humble"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed without segfault (exit code 0)"
    echo "The issue may be intermittent or require more iterations"
else
    echo "Script exited with code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
