from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence

from app.tools.file_tools import normalize_uploaded_files
from app.tools.report_tools import build_stage_outcomes
from app.tools.report_tools import synthesize_final_report
from app.tools.retrieval_tools import retrieve_contexts
from app.tools.rule_tools import evaluate_completeness_rules


class LangChainReviewChain:
    """LangChain-based conditional review chain for the agentic-RAG service.

    The chain uses LangChain runnables to orchestrate the existing
    completeness-review stages:
    1. normalize uploaded files
    2. retrieve relevant context from the vector indexes
    3. evaluate completeness rules
    4. build stage outcomes and prompts
    5. synthesize the final strict report

    Each stage is executed through explicit LangChain tool invocations so the
    workflow remains modular and debuggable.

    Args:
        None.

    Returns:
        LangChainReviewChain: Configured runnable review chain.
    """

    def __init__(self) -> None:
        """Initialize the runnable review chain.

        Args:
            None.

        Returns:
            None.
        """

        self._chain: RunnableSequence = (
            RunnableLambda(self._normalize_step)
            | RunnableLambda(self._retrieve_step)
            | RunnableLambda(self._evaluate_step)
            | RunnableLambda(self._stage_outcomes_step)
            | RunnableLambda(self._finalize_step)
        )

    def review(
        self,
        application_name: str,
        files: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the conditional review workflow for one application package.

        Args:
            application_name: Program name under review.
            files: Uploaded file payloads as plain dictionaries.

        Returns:
            dict[str, Any]: Structured review payload returned by the service.
        """

        return self._chain.invoke(
            {
                "application_name": application_name,
                "files": files,
            }
        )

    def _normalize_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Normalize uploaded files through the dedicated LangChain tool.

        Args:
            state: Current chain state.

        Returns:
            dict[str, Any]: Updated chain state with normalized file texts.
        """

        normalized_file_texts = normalize_uploaded_files.invoke({"files": state["files"]})
        return {
            **state,
            "normalized_file_texts": normalized_file_texts,
        }

    def _retrieve_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Retrieve vector context for the normalized application package.

        Args:
            state: Current chain state.

        Returns:
            dict[str, Any]: Updated chain state with retrieved contexts.
        """

        retrieved_contexts = retrieve_contexts.invoke(
            {
                "application_name": state["application_name"],
                "normalized_file_texts": state["normalized_file_texts"],
            }
        )
        return {
            **state,
            "retrieved_contexts": retrieved_contexts,
        }

    def _evaluate_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate completeness rules through the dedicated rule tool.

        Args:
            state: Current chain state.

        Returns:
            dict[str, Any]: Updated chain state with evaluation results.
        """

        evaluation = evaluate_completeness_rules.invoke(
            {"normalized_file_texts": state["normalized_file_texts"]}
        )
        return {
            **state,
            "evaluation": evaluation,
        }

    def _stage_outcomes_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Build the three review stage outcomes from current chain state.

        Args:
            state: Current chain state.

        Returns:
            dict[str, Any]: Updated chain state with stage outcomes.
        """

        stage_outcomes = build_stage_outcomes.invoke(
            {
                "application_name": state["application_name"],
                "normalized_file_texts": state["normalized_file_texts"],
                "evaluation": state["evaluation"],
                "retrieved_contexts": state["retrieved_contexts"],
            }
        )
        return {
            **state,
            "stage_outcomes": stage_outcomes,
        }

    def _finalize_step(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create the final strict report and return the response payload.

        Args:
            state: Current chain state.

        Returns:
            dict[str, Any]: Final structured review payload.
        """

        final_report_text = synthesize_final_report.invoke(
            {
                "application_name": state["application_name"],
                "stage_outcomes": state["stage_outcomes"],
            }
        )
        return {
            "application_name": state["application_name"],
            "normalized_file_texts": state["normalized_file_texts"],
            "retrieved_contexts": state["retrieved_contexts"],
            "stage_outcomes": state["stage_outcomes"],
            "final_report_text": final_report_text,
        }
