from __future__ import annotations

from app.prompts.templates import load_prompt_bundle
from app.prompts.templates import render_content_sufficiency_prompt
from app.prompts.templates import render_document_presence_prompt
from app.prompts.templates import render_form_completion_prompt


def test_system_prompt_contains_main_completeness_rule() -> None:
    """Verify the system prompt contains the core completeness rule block.

    Args:
        None.

    Returns:
        None: Assertions validate the locked rule text.
    """

    prompt_bundle = load_prompt_bundle()

    print(prompt_bundle.system_prompt)

    assert "What is a complete application" in prompt_bundle.system_prompt
    assert "All the questions on the application form are answered." in prompt_bundle.system_prompt
    assert "Proof of payment has been submitted." in prompt_bundle.system_prompt
    assert "All required forms are signed." in prompt_bundle.system_prompt
    assert "All documents have been submitted." in prompt_bundle.system_prompt


def test_stage_templates_render_with_system_prompt() -> None:
    """Verify all stage templates render the system prompt and stage data.

    Args:
        None.

    Returns:
        None: Assertions validate the rendered prompt strings.
    """

    prompt_bundle = load_prompt_bundle()

    document_prompt = render_document_presence_prompt(
        prompt_bundle=prompt_bundle,
        application_name="visitor visa",
        uploaded_files=["imm5257.pdf", "fee_receipt.pdf"],
    )
    form_prompt = render_form_completion_prompt(
        prompt_bundle=prompt_bundle,
        application_name="visitor visa",
        form_evidence=["imm5257.pdf: Required questions remain unanswered on IMM 5257."],
    )
    content_prompt = render_content_sufficiency_prompt(
        prompt_bundle=prompt_bundle,
        application_name="visitor visa",
        retrieved_context=["Proof of payment has been submitted."],
        extracted_evidence=["fee_receipt.pdf: Proof of payment for the applicable fee is missing."],
    )

    print("DOCUMENT PROMPT:")
    print(document_prompt)
    print("FORM PROMPT:")
    print(form_prompt)
    print("CONTENT PROMPT:")
    print(content_prompt)

    assert "What is a complete application" in document_prompt
    assert "imm5257.pdf" in document_prompt
    assert "Required questions remain unanswered on IMM 5257." in form_prompt
    assert "Proof of payment for the applicable fee is missing." in content_prompt
