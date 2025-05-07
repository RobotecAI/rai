#!/usr/bin/env bash

set -e

DEMOS_DIR="demo_assets"
mkdir -p "$DEMOS_DIR"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <demo_name>"
    echo "Available demos: manipulation, rosbot, agriculture"
    exit 1
fi

DEMO_NAME="$1"

if [ -z "$ROS_DISTRO" ]; then
    echo "ROS_DISTRO environment variable is not set."
    exit 1
fi

MANIPULATION_HUMBLE="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIManipulationDemo_1.0.0_jammyhumble.zip"
MANIPULATION_JAZZY="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIManipulationDemo_1.0.0_noblejazzy.zip"
ROSBOT_HUMBLE="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIROSBotDemo_1.0.0_jammyhumble.zip"
ROSBOT_JAZZY="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIROSBotDemo_1.0.0_noblejazzy.zip"
AGRICULTURE_HUMBLE="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIAgricultureDemo_1.0.0_jammyhumble.zip"
AGRICULTURE_JAZZY="https://robotec-ml-roscon2024-demos.s3.eu-central-1.amazonaws.com/ROSCON_Release/RAIAgricultureDemo_1.0.0_noblejazzy.zip"

case "$DEMO_NAME" in
    manipulation)
        case "$ROS_DISTRO" in
            humble)
                URL="$MANIPULATION_HUMBLE"
                ;;
            jazzy)
                URL="$MANIPULATION_JAZZY"
                ;;
            *)
                echo "Unsupported ROS_DISTRO: $ROS_DISTRO"
                exit 1
                ;;
        esac
        ;;
    rosbot)
        case "$ROS_DISTRO" in
            humble)
                URL="$ROSBOT_HUMBLE"
                ;;
            jazzy)
                URL="$ROSBOT_JAZZY"
                ;;
            *)
                echo "Unsupported ROS_DISTRO: $ROS_DISTRO"
                exit 1
                ;;
        esac
        ;;
    agriculture)
        case "$ROS_DISTRO" in
            humble)
                URL="$AGRICULTURE_HUMBLE"
                ;;
            jazzy)
                URL="$AGRICULTURE_JAZZY"
                ;;
            *)
                echo "Unsupported ROS_DISTRO: $ROS_DISTRO"
                exit 1
                ;;
        esac
        ;;
    *)
        echo "Unknown demo name: $DEMO_NAME"
        echo "Available demos: manipulation, rosbot, agriculture"
        exit 1
        ;;
esac

FILENAME="${URL##*/}"

echo "Downloading $DEMO_NAME demo for ROS_DISTRO=$ROS_DISTRO..."
echo "From: $URL"
echo "To: $DEMOS_DIR/$FILENAME"

wget -q --show-progress -O "${DEMOS_DIR}/${FILENAME}" "$URL"

echo "Download complete: ${DEMOS_DIR}/${FILENAME}"

# Unzip to a subdirectory named after the demo
TARGET_DIR="${DEMOS_DIR}/${DEMO_NAME}"
if [ -d "$TARGET_DIR" ]; then
    echo "Removing existing directory: $TARGET_DIR"
    rm -rf "$TARGET_DIR"
fi

echo "Unzipping to $TARGET_DIR ..."
unzip -q "${DEMOS_DIR}/${FILENAME}" -d "$TARGET_DIR"

echo "Extraction complete."

rm -f "${DEMOS_DIR}/${FILENAME}"

echo "Done. Demo is available in $TARGET_DIR"
