# Proposed Fix for Issue #759

## Summary

Change the `ros_setup` fixture from function-scoped to session-scoped to reduce the frequency of `rclpy.init()`/`rclpy.shutdown()` cycles, which trigger race conditions in ROS2 Humble.

## Changes Required

### File: `tests/communication/ros2/helpers.py`

**Current code (line 576-581):**
```python
@pytest.fixture(scope="function")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
```

**Proposed change:**
```python
@pytest.fixture(scope="session")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
```

## Rationale

### Current Behavior
- `scope="function"` means `rclpy.init()` and `rclpy.shutdown()` are called for **every test function**
- With ~50 tests using this fixture, that's 50 init/shutdown cycles
- Each cycle increases the probability of hitting the race condition in ROS2 Humble's C++ layer

### Proposed Behavior
- `scope="session"` means `rclpy.init()` is called **once at the start** of the test session
- `rclpy.shutdown()` is called **once at the end** of the test session
- This matches typical production usage patterns
- Dramatically reduces the opportunity for race conditions

## Testing Strategy

### 1. Verify Tests Still Pass

```bash
# Run the specific failing test
pytest tests/tools/ros2/test_action_tools.py::test_get_actions_names_and_types_tool_with_forbidden -v

# Run all action tools tests
pytest tests/tools/ros2/test_action_tools.py -v

# Run all ROS2 tests
pytest tests/tools/ros2/ tests/communication/ros2/ -v
```

### 2. Test for Interference

Since tests now share the same ROS2 context, verify that:
- Tests don't interfere with each other
- Node names are unique (already using UUID in some places)
- Resources are properly cleaned up between tests

### 3. Stress Test

Run tests multiple times to ensure stability:

```bash
# Run 20 times to check for intermittent failures
pytest tests/tools/ros2/test_action_tools.py --count=20 -v

# Run with different orders to check for order dependencies
pytest tests/tools/ros2/test_action_tools.py --random-order -v
```

## Potential Issues and Mitigations

### Issue 1: Test Isolation

**Problem:** Tests might interfere with each other if they share ROS2 context.

**Mitigation:**
- Review tests to ensure unique node names (many already use `uuid.uuid4()`)
- Ensure proper cleanup in test teardown
- Add test markers for tests that need isolation

### Issue 2: Failure Propagation

**Problem:** If `rclpy.init()` fails, all tests in the session fail.

**Mitigation:**
- This is actually desirable - if ROS2 can't initialize, tests should fail fast
- Current behavior would fail all tests anyway, just slower

### Issue 3: Context State

**Problem:** Tests might depend on a "fresh" ROS2 context.

**Mitigation:**
- Review tests for assumptions about initial state
- Add explicit cleanup where needed
- Most tests already create their own nodes and clean them up

## Alternative: Module Scope

If session scope is too broad, consider `scope="module"`:

```python
@pytest.fixture(scope="module")
def ros_setup() -> Generator[None, None, None]:
    rclpy.init()
    yield
    rclpy.shutdown()
```

This would:
- Init/shutdown once per test file (not per test function)
- Reduce cycles from ~50 to ~5-10 (depending on number of test files)
- Provide better isolation than session scope
- Still significantly reduce the race condition probability

## Implementation Steps

1. **Make the change** in `tests/communication/ros2/helpers.py`
2. **Run local tests** to verify no obvious breakage
3. **Review test code** for potential isolation issues
4. **Fix any issues** found during testing
5. **Update documentation** if needed
6. **Monitor CI** for improvements in Humble stability

## Expected Outcome

- **Significant reduction** in segfaults on ROS2 Humble CI runs
- **Faster test execution** (less overhead from init/shutdown)
- **More realistic testing** (matches production usage patterns)
- Tests should still pass with the same coverage

## Rollback Plan

If the change causes issues:
1. Revert the fixture scope change
2. Consider the alternative solutions in INVESTIGATION_REPORT.md
3. Add more robust synchronization in cleanup code
4. Consider marking problematic tests as Humble-incompatible

## Additional Recommendations

### 1. Add a Comment

Add a comment explaining why session scope is used:

```python
@pytest.fixture(scope="session")
def ros_setup() -> Generator[None, None, None]:
    """Initialize ROS2 once per test session.
    
    Using session scope instead of function scope to reduce the frequency
    of rclpy.init()/shutdown() cycles, which can trigger race conditions
    in ROS2 Humble's C++ layer. See issue #759 for details.
    """
    rclpy.init()
    yield
    rclpy.shutdown()
```

### 2. Monitor CI

After the change, monitor CI runs for:
- Reduction in segfaults
- Any new test failures
- Test execution time improvements

### 3. Document the Issue

Add a note to the README or documentation about:
- Known issues with ROS2 Humble
- The workaround implemented
- Recommendation to use ROS2 Jazzy for production

## References

- Issue: https://github.com/RobotecAI/rai/issues/759
- Investigation Report: `INVESTIGATION_REPORT.md`
- Minimal Reproduction: `minimal_repro.py`, `minimal_repro_simple.py`
