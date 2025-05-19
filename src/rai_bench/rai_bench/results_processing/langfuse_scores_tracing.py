from typing import List
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langfuse.callback import CallbackHandler
from rai.initialization import get_tracing_callbacks


class ScoreTracingHandler:
    """
    Class to handle sending scores to tracing backends.
    """

    # TODO (mkotynia) handle grouping single benchmark scores to sessions
    # TODO (mkotynia) trace and send more metadata?
    @staticmethod
    def get_callbacks() -> List[BaseCallbackHandler]:
        return get_tracing_callbacks()

    @staticmethod
    def send_score(
        callback: BaseCallbackHandler,
        run_id: UUID,
        score: float,
        errors: List[List[str]] | None = None,
    ) -> None:
        comment = (
            "; ".join(", ".join(error_group) for error_group in errors)
            if errors
            else ""
        )
        if isinstance(callback, CallbackHandler):
            callback.langfuse.score(
                trace_id=str(run_id),
                name="tool calls result",
                value=score,
                comment=comment,
            )
            return None
        if isinstance(callback, LangChainTracer):
            callback.client.create_feedback(
                run_id=run_id,
                key="tool calls result",
                score=score,
                comment=comment,
            )
            return None
        raise NotImplementedError(
            f"Callback {callback} of type {callback.__class__.__name__} not supported"
        )
