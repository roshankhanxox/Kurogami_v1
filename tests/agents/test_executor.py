"""Executor: sends generated_prompt + context to the LLM; SearchPort only for RESEARCH."""

from kurogami.adapters.search.fake import FakeSearch
from kurogami.agents.executor import Executor
from kurogami.contracts import LLMResponse, NodeKind, NodeSpec, PassCondition, SearchHit


class _CapturingLLM:
    """Records the last prompt it was sent; returns a fixed canned response."""

    def __init__(self, text: str = "{}") -> None:
        self.last_prompt: str | None = None
        self._text = text

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        self.last_prompt = prompt
        return LLMResponse(text=self._text, tokens_in=10, tokens_out=10, latency_ms=1, model_id="fake")


def _node(**overrides) -> NodeSpec:
    kwargs = {
        "node_id": "n_001",
        "parent_ids": [],
        "depth": 0,
        "kind": NodeKind.ANALYSIS,
        "title": "title",
        "node_goal": "goal",
        "generated_prompt": "do the analysis",
        "pass_condition": PassCondition(assertions=[], semantic_check="ok?"),
    }
    kwargs.update(overrides)
    return NodeSpec(**kwargs)


def test_executor_sends_the_generated_prompt():
    llm = _CapturingLLM()
    Executor(llm).run(_node())
    assert llm.last_prompt.startswith("do the analysis")


def test_executor_includes_ancestor_context():
    llm = _CapturingLLM()
    Executor(llm).run(_node(context={"n_000": "market sizing output"}))
    assert "market sizing output" in llm.last_prompt


def test_executor_includes_injected_constraints():
    llm = _CapturingLLM()
    Executor(llm).run(_node(injected_constraints=["Stay under INR 500/month."]))
    assert "Stay under INR 500/month." in llm.last_prompt


def test_executor_calls_search_only_for_research_nodes():
    search = FakeSearch(hits=[SearchHit(title="T", url="https://x.invalid", snippet="S")])
    llm = _CapturingLLM()

    Executor(llm, search=search).run(_node(kind=NodeKind.ANALYSIS))
    assert "Search results" not in llm.last_prompt

    Executor(llm, search=search).run(_node(kind=NodeKind.RESEARCH))
    assert "Search results" in llm.last_prompt
    assert "T" in llm.last_prompt


def test_executor_parses_json_output_into_structured():
    result = Executor(_CapturingLLM(text='{"wtp_ceiling_inr": 500}')).run(_node())
    assert result.structured == {"wtp_ceiling_inr": 500}


def test_executor_falls_back_to_empty_structured_on_non_json_output():
    result = Executor(_CapturingLLM(text="not json at all")).run(_node())
    assert result.structured == {}


def test_executor_extracts_json_from_a_fenced_code_block_in_prose():
    """Regression: a real LLM given a free-text prompt often writes prose with
    a trailing JSON block, not pure JSON -- json.loads() on the whole text
    fails in that case.
    """
    text = 'Here is my analysis.\n\n```json\n{"market_trends": ["a", "b"]}\n```\n'
    result = Executor(_CapturingLLM(text=text)).run(_node())
    assert result.structured == {"market_trends": ["a", "b"]}


def test_executor_extracts_a_bare_json_object_embedded_in_prose():
    text = 'Some prose before. {"key": "value"} and prose after.'
    result = Executor(_CapturingLLM(text=text)).run(_node())
    assert result.structured == {"key": "value"}


def test_executor_asks_for_the_keys_its_own_assertions_reference():
    """Regression: seen live -- an assertion referencing structured['market_trends']
    raised KeyError every time, because nothing ever told the model to
    produce that key. The executor now derives required keys from the
    node's own assertions, deterministically (no LLM guessing), and asks
    for exactly those.
    """
    node = _node(
        pass_condition=PassCondition(
            assertions=["len(structured['competitors']) >= 3", "structured['market_trends'] != []"],
            semantic_check="ok?",
        )
    )
    llm = _CapturingLLM()
    Executor(llm).run(node)

    assert "competitors" in llm.last_prompt
    assert "market_trends" in llm.last_prompt
    assert "```json" in llm.last_prompt


def test_executor_shows_the_exact_assertions_so_value_shapes_match():
    """Regression: seen live -- the model returned user_feedback as a single
    string where the assertion needed a list; knowing only key names wasn't
    enough to get the shape right.
    """
    assertion = "len(structured['user_feedback']) >= 3"
    node = _node(pass_condition=PassCondition(assertions=[assertion], semantic_check="ok?"))
    llm = _CapturingLLM()
    Executor(llm).run(node)
    assert assertion in llm.last_prompt


def test_executor_forbids_nesting_required_keys_under_a_wrapper():
    """Regression: seen live -- the model produced
    {"feasibility_analysis": {"feature_1": ..., "total_estimated_time": ...}}
    instead of putting the required keys at the top level, so a flat
    structured[...] assertion still raised KeyError.
    """
    node = _node(
        pass_condition=PassCondition(
            assertions=["structured['total_estimated_time'] is not None"],
            semantic_check="ok?",
        )
    )
    llm = _CapturingLLM()
    Executor(llm).run(node)

    assert "TOP level" in llm.last_prompt
    assert "do not nest" in llm.last_prompt


def test_executor_adds_no_required_keys_when_there_are_no_assertions():
    llm = _CapturingLLM()
    Executor(llm).run(_node())
    assert "exactly these keys" not in llm.last_prompt


def test_executor_always_lets_a_node_report_missing_prerequisites():
    """The only trigger for runtime gap-filling -- offered even with no assertions."""
    llm = _CapturingLLM()
    Executor(llm).run(_node())
    assert "fenced JSON" in llm.last_prompt
    assert "missing_prerequisites" in llm.last_prompt


def test_executor_prompt_version_is_stable_for_an_identical_prompt():
    llm = _CapturingLLM()
    node = _node()
    r1 = Executor(llm).run(node)
    r2 = Executor(llm).run(node)
    assert r1.prompt_version == r2.prompt_version


def test_executor_prompt_version_differs_for_different_prompts():
    llm = _CapturingLLM()
    r1 = Executor(llm).run(_node(generated_prompt="prompt one"))
    r2 = Executor(llm).run(_node(generated_prompt="prompt two"))
    assert r1.prompt_version != r2.prompt_version


def test_executor_finds_keys_read_through_get_and_membership():
    """Live incident: once `.get` was allowed, a subscript-only regex missed
    `structured.get('competitors')`, the model was never told the key, named it
    `tools`, and all three live runs failed on it."""
    assertions = [
        "isinstance(structured.get('competitors'), list)",
        "all(key in structured for key in ['budget_limit', 'timeline'])",
        "'risks' in structured",
        "len(structured['tiers']) >= 2",
    ]
    assert Executor._required_structured_keys(assertions) == [
        "competitors", "budget_limit", "timeline", "risks", "tiers",
    ]
