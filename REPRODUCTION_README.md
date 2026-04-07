# ROS2 Humble Segfault Reproduction Scripts

This directory contains reproduction scripts for issue #759 - segmentation faults in ROS2 Humble during action tools tests.

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build the Docker image
docker build -f Dockerfile.humble-repro -t ros2-humble-segfault-repro .

# Run the reproduction
docker run --rm ros2-humble-segfault-repro

# Or use the helper script
./run_repro.sh
```

### Option 2: Manual (Requires ROS2 Humble)

```bash
# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Run the simplified version
python3 minimal_repro_simple.py

# Or run the comprehensive version
python3 minimal_repro.py
```

## Scripts

### `minimal_repro_simple.py` (Recommended for Sharing)

**Size:** ~70 lines  
**Purpose:** Minimal, easy-to-understand reproduction  
**Best for:** Sharing with ROS2 community, quick testing

**What it does:**
- Runs 10 test cycles
- Each cycle: init → create action servers → query actions → cleanup → shutdown
- Mimics the exact pattern that causes the segfault

**Expected behavior:** Segfault after 3-7 iterations (intermittent)

### `minimal_repro.py` (Comprehensive)

**Size:** ~200 lines  
**Purpose:** Detailed reproduction with extensive logging  
**Best for:** Debugging, understanding the exact sequence

**What it does:**
- Same as simple version but with detailed logging at each step
- Shows exactly when the segfault occurs
- Includes error handling and progress tracking

**Expected behavior:** Segfault after 3-7 iterations with detailed logs

## Dependencies

Both scripts require:
- Python 3
- ROS2 Humble (or Jazzy for comparison)
- `rclpy` (included with ROS2)
- `nav2_msgs` (install with: `apt install ros-humble-nav2-msgs`)

**No RAI dependencies required** - these are standalone reproduction scripts.

## What the Scripts Demonstrate

The scripts reproduce the exact problematic pattern:

1. **Multiple init/shutdown cycles** - Each test cycle calls `rclpy.init()` and `rclpy.shutdown()`
2. **Multi-threaded executors** - Action servers spin in separate threads
3. **Action query during cleanup** - `rclpy.action.get_action_names_and_types()` is called
4. **Race condition** - Threads access ROS2 resources during cleanup

This pattern triggers a race condition in ROS2 Humble's C++ layer, causing a segmentation fault.

## Exit Codes

- **0**: All iterations completed successfully (no segfault)
- **139**: Segmentation fault occurred (expected in ROS2 Humble)
- **Other**: Unexpected error

## Using with Docker

The `Dockerfile.humble-repro` creates a minimal environment with:
- ROS2 Humble Desktop Full
- Python 3
- nav2_msgs package
- The reproduction scripts

The `run_repro.sh` script:
1. Builds the Docker image
2. Runs the reproduction
3. Reports the exit code
4. Indicates if a segfault occurred (exit code 139)

## Troubleshooting

### "No segfault occurred"

The issue is intermittent due to the race condition. Try:
- Running multiple times
- Increasing the number of iterations in the script
- Running under load (multiple instances)

### "Docker not found"

Docker is not required. You can run the scripts directly with ROS2 Humble installed:

```bash
source /opt/ros/humble/setup.bash
python3 minimal_repro_simple.py
```

### "nav2_msgs not found"

Install nav2_msgs:

```bash
# For Humble
sudo apt install ros-humble-nav2-msgs

# For Jazzy
sudo apt install ros-jazzy-nav2-msgs
```

## Comparing ROS2 Versions

To compare Humble vs Jazzy behavior:

```bash
# Test with Humble
source /opt/ros/humble/setup.bash
python3 minimal_repro_simple.py

# Test with Jazzy
source /opt/ros/jazzy/setup.bash
python3 minimal_repro_simple.py
```

Expected: Segfault in Humble, no segfault in Jazzy (Jazzy has better thread safety)

## Sharing with ROS2 Community

If you want to report this to the ROS2 project:

1. Use `minimal_repro_simple.py` (easier to understand)
2. Include the Docker setup for reproducibility
3. Mention:
   - ROS2 Humble specific
   - Race condition in action server cleanup
   - Triggered by frequent init/shutdown cycles
   - Related to multi-threaded executors

## Related Files

- **`INVESTIGATION_REPORT.md`**: Detailed analysis of the issue
- **`PROPOSED_FIX.md`**: Fix documentation for the RAI codebase
- **`SUMMARY.md`**: Complete investigation overview
- **`QUICK_REFERENCE.md`**: Quick reference card

## Issue Links

- **GitHub Issue**: https://github.com/RobotecAI/rai/issues/759
- **Pull Request**: https://github.com/RobotecAI/rai/pull/783

## License

These reproduction scripts are provided as part of the RAI project investigation.
See the main repository for license information.

## Questions?

See the investigation documents for detailed analysis and explanation of the issue.
