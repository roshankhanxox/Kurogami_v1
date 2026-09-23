"""Interpreter: sentence -> GoalSpec, schema-constrained, offline via FakeLLM."""

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.agents._prompt_loader import load_prompt
from kurogami.agents.interpreter import Interpreter
from kurogami.contracts import GoalSpec


def test_interpreter_returns_the_llms_parsed_goal_spec():
    raw_text = "Should I launch my invoicing tool for freelance designers in India?"
    expected = GoalSpec(
        raw_text=raw_text,
        product_description="Invoicing and late-payment chasing tool for freelancers.",
        target_market="Freelance designers in India",
        decision_type="market_entry",
        success_definition="A clear go/no-go with a pricing rationale.",
        ambiguities=["Launch timeline is not stated."],
    )
    prompt = load_prompt("interpret").format(raw_text=raw_text)
    llm = FakeLLM(responses={prompt: expected})

    result = Interpreter(llm).run(raw_text)

    assert result == expected


def test_interpreter_sends_the_raw_text_through_the_template():
    raw_text = "A very specific unique sentence nobody else would type."
    expected = GoalSpec(
        raw_text=raw_text,
        product_description="x",
        target_market="x",
        decision_type="pricing",
        success_definition="x",
    )
    prompt = load_prompt("interpret").format(raw_text=raw_text)
    llm = FakeLLM(responses={prompt: expected})

    result = Interpreter(llm).run(raw_text)

    assert result.raw_text == raw_text
