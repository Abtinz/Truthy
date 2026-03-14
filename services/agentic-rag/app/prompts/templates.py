from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import PromptTemplate


PROMPTS_DIRECTORY = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PromptBundle:
    """In-memory prompt bundle used by the agentic-RAG review flow.

    Args:
        system_prompt: Shared system-level instruction block.
        document_presence_template: Stage template for document-presence review.
        form_completion_template: Stage template for form-completion review.
        content_sufficiency_template: Stage template for content sufficiency.

    Returns:
        PromptBundle: Immutable prompt container for rendering stage prompts.
    """

    system_prompt: str
    document_presence_template: str
    form_completion_template: str
    content_sufficiency_template: str


def _load_prompt_file(file_name: str) -> str:
    """Load one prompt file from the prompts directory.

    Args:
        file_name: Prompt file name relative to the prompts directory.

    Returns:
        str: Prompt file contents with surrounding whitespace trimmed.
    """

    return (PROMPTS_DIRECTORY / file_name).read_text(encoding="utf-8").strip()


def load_prompt_bundle() -> PromptBundle:
    """Load the system prompt and stage templates from disk.

    Args:
        None.

    Returns:
        PromptBundle: Loaded prompt bundle for the current service runtime.
    """

    return PromptBundle(
        system_prompt=_load_prompt_file("system.txt"),
        document_presence_template=_load_prompt_file("document_presence.txt"),
        form_completion_template=_load_prompt_file("form_completion.txt"),
        content_sufficiency_template=_load_prompt_file("content_sufficiency.txt"),
    )


def render_document_presence_prompt(
    *,
    prompt_bundle: PromptBundle,
    application_name: str,
    uploaded_files: list[str],
) -> str:
    """Render the document-presence stage prompt.

    Args:
        prompt_bundle: Loaded system prompt and stage template bundle.
        application_name: Program name under review.
        uploaded_files: Uploaded file names included in the package.

    Returns:
        str: Rendered document-presence prompt text.
    """

    uploaded_file_block = "\n".join(f"- {file_name}" for file_name in uploaded_files) or "- none"
    return PromptTemplate.from_template(
        prompt_bundle.document_presence_template
    ).format(
        system_prompt=prompt_bundle.system_prompt,
        application_name=application_name,
        uploaded_files=uploaded_file_block,
    )


def render_form_completion_prompt(
    *,
    prompt_bundle: PromptBundle,
    application_name: str,
    form_evidence: list[str],
) -> str:
    """Render the form-completion stage prompt.

    Args:
        prompt_bundle: Loaded system prompt and stage template bundle.
        application_name: Program name under review.
        form_evidence: Form-related evidence lines prepared for the stage.

    Returns:
        str: Rendered form-completion prompt text.
    """

    form_evidence_block = "\n".join(f"- {line}" for line in form_evidence) or "- none"
    return PromptTemplate.from_template(
        prompt_bundle.form_completion_template
    ).format(
        system_prompt=prompt_bundle.system_prompt,
        application_name=application_name,
        form_evidence=form_evidence_block,
    )


def render_content_sufficiency_prompt(
    *,
    prompt_bundle: PromptBundle,
    application_name: str,
    retrieved_context: list[str],
    extracted_evidence: list[str],
) -> str:
    """Render the content-sufficiency stage prompt.

    Args:
        prompt_bundle: Loaded system prompt and stage template bundle.
        application_name: Program name under review.
        retrieved_context: Retrieved policy or checklist context lines.
        extracted_evidence: Evidence lines prepared for content review.

    Returns:
        str: Rendered content-sufficiency prompt text.
    """

    retrieved_context_block = "\n".join(f"- {line}" for line in retrieved_context) or "- none"
    extracted_evidence_block = "\n".join(f"- {line}" for line in extracted_evidence) or "- none"
    return PromptTemplate.from_template(
        prompt_bundle.content_sufficiency_template
    ).format(
        system_prompt=prompt_bundle.system_prompt,
        application_name=application_name,
        retrieved_context=retrieved_context_block,
        extracted_evidence=extracted_evidence_block,
    )
