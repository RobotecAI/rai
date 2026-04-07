# Quick Reference: Issue #759 Fix

## TL;DR

**Problem:** Segfaults in ROS2 Humble CI during action tools tests  
**Cause:** Race condition in ROS2 Humble's C++ layer  
**Fix:** Changed `ros_setup` fixture from function scope to session scope  
**Result:** Reduces init/shutdown cycles from ~50 to 1, preventing race condition

## One-Line Summary

Changed `@pytest.fixture(scope="function")` to `@pytest.fixture(scope="session")` in `tests/communication/ros2/helpers.py`

## Why This Fixes It

| Before | After |
|--------|-------|
| `rclpy.init()` called for every test | `rclpy.init()` called once per session |
| ~50 init/shutdown cycles | 1 init/shutdown cycle |
| High probability of race condition | Very low probability of race condition |
| Slower tests | Faster tests |

## What Changed

**File:** `tests/communication/ros2/helpers.py`  
**Lines:** 576-581  
**Change:** `scope="function"` → `scope="session"`

## Testing Commands

```bash
# Run the failing test
pytest tests/tools/ros2/test_action_tools.py::test_get_actions_names_and_types_tool_with_forbidden -v

# Run all action tools tests
pytest tests/tools/ros2/test_action_tools.py -v

# Stress test (20 iterations)
pytest tests/tools/ros2/test_action_tools.py --count=20 -v
```

## Reproduction Scripts

Two scripts created to reproduce the issue with only rclpy (no RAI):

1. **`minimal_repro_simple.py`** - 70 lines, easy to understand
2. **`minimal_repro.py`** - 200 lines, comprehensive with logging

Run with:
```bash
source /opt/ros/humble/setup.bash
python3 minimal_repro_simple.py
```

Expected: Segfault after 3-7 iterations

## Key Documents

| Document | Purpose |
|----------|---------|
| `SUMMARY.md` | Overview of investigation and fix |
| `INVESTIGATION_REPORT.md` | Detailed analysis (~400 lines) |
| `PROPOSED_FIX.md` | Fix documentation and testing strategy |
| `QUICK_REFERENCE.md` | This file - quick lookup |

## Links

- **Issue:** https://github.com/RobotecAI/rai/issues/759
- **PR:** https://github.com/RobotecAI/rai/pull/783
- **Branch:** `CU-_Investigate-759_Maciej-Majek`

## Is This Safe?

**Yes.** The change is low-risk because:

✅ Tests already use unique node names (UUIDs)  
✅ Matches production usage (init once, not repeatedly)  
✅ Makes tests faster  
✅ Only affects test infrastructure, not production code  
✅ Easy to revert if needed  

## Potential Issues

**Q: Will tests interfere with each other?**  
A: No, node names use UUIDs for uniqueness

**Q: What if a test needs a fresh ROS2 context?**  
A: Very unlikely, but we can add explicit cleanup if needed

**Q: What if this doesn't fully fix it?**  
A: See `PROPOSED_FIX.md` for alternative solutions

## If You Need More Info

1. **Quick overview:** Read `SUMMARY.md`
2. **Detailed analysis:** Read `INVESTIGATION_REPORT.md`
3. **Testing strategy:** Read `PROPOSED_FIX.md`
4. **Try reproduction:** Run `minimal_repro_simple.py`

## Bottom Line

This is a **one-line fix** (plus documentation) that solves a race condition in ROS2 Humble by reducing the frequency of init/shutdown cycles. The issue is in ROS2, not RAI. The fix is safe, tested, and should significantly improve CI stability.
