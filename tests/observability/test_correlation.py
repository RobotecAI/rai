from rai.observability.correlation import (
    get_observability_context,
    observability_context,
    reset_observability_context,
    set_observability_context,
)


def test_get_observability_context_empty_by_default():
    assert get_observability_context() == {}


def test_observability_context_scopes_and_restores():
    assert get_observability_context() == {}

    with observability_context(run_id="r", job_id="j", task_id="t", request_id="q"):
        assert get_observability_context() == {
            "run_id": "r",
            "job_id": "j",
            "task_id": "t",
            "request_id": "q",
        }

    # Restored back to empty.
    assert get_observability_context() == {}


def test_observability_context_nesting_overrides_then_restores():
    with observability_context(run_id="r1", job_id="j1"):
        assert get_observability_context() == {"run_id": "r1", "job_id": "j1"}

        with observability_context(job_id="j2", request_id="q2"):
            # Only provided fields override; others stay as-is.
            assert get_observability_context() == {
                "run_id": "r1",
                "job_id": "j2",
                "request_id": "q2",
            }

        # Inner context restored.
        assert get_observability_context() == {"run_id": "r1", "job_id": "j1"}


def test_set_and_reset_observability_context_tokens():
    assert get_observability_context() == {}

    tokens = set_observability_context(run_id="r", request_id="q")
    try:
        assert get_observability_context() == {"run_id": "r", "request_id": "q"}
    finally:
        reset_observability_context(tokens)

    assert get_observability_context() == {}
