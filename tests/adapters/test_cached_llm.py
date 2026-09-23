"""CachedLLM: hash(prompt+model+temperature+schema) -> .cache/llm/<hash>.json.

Demo insurance (ARCHITECTURE.md section 9) -- fully offline, no network.
"""

from kurogami.adapters.llm.cached import CachedLLM
from kurogami.contracts import GoalSpec, LLMResponse


class _CountingLLM:
    def __init__(self, text: str = "response text") -> None:
        self.calls = 0
        self._text = text

    def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
        self.calls += 1
        return LLMResponse(
            text=self._text, tokens_in=1, tokens_out=1, latency_ms=5, model_id="inner"
        )


def test_second_identical_call_is_served_from_cache_not_the_inner_llm(tmp_path):
    inner = _CountingLLM()
    llm = CachedLLM(inner, model_id="m", cache_dir=tmp_path)

    first = llm.complete(prompt="p")
    second = llm.complete(prompt="p")

    assert inner.calls == 1
    assert first.text == second.text == "response text"
    assert second.latency_ms == 0  # served from disk


def test_different_prompts_get_separate_cache_entries(tmp_path):
    inner = _CountingLLM()
    llm = CachedLLM(inner, model_id="m", cache_dir=tmp_path)

    llm.complete(prompt="a")
    llm.complete(prompt="b")

    assert inner.calls == 2


def test_different_temperatures_get_separate_cache_entries(tmp_path):
    inner = _CountingLLM()
    llm = CachedLLM(inner, model_id="m", cache_dir=tmp_path)

    llm.complete(prompt="p", temperature=0.0)
    llm.complete(prompt="p", temperature=0.7)

    assert inner.calls == 2


def test_different_model_ids_get_separate_cache_entries(tmp_path):
    inner = _CountingLLM()
    CachedLLM(inner, model_id="model-a", cache_dir=tmp_path).complete(prompt="p")
    CachedLLM(inner, model_id="model-b", cache_dir=tmp_path).complete(prompt="p")

    assert inner.calls == 2


def test_cache_round_trips_a_schema_parsed_response(tmp_path):
    goal = GoalSpec(
        raw_text="x", product_description="x", target_market="x",
        decision_type="pricing", success_definition="x",
    )

    class _SchemaLLM:
        def complete(self, *, prompt, system=None, schema=None, temperature=0.0):
            return LLMResponse(
                text=goal.model_dump_json(), parsed=goal, tokens_in=1, tokens_out=1,
                latency_ms=1, model_id="inner",
            )

    llm = CachedLLM(_SchemaLLM(), model_id="m", cache_dir=tmp_path)
    llm.complete(prompt="p", schema=GoalSpec)
    second = llm.complete(prompt="p", schema=GoalSpec)

    assert second.parsed == goal


def test_cache_persists_across_separate_cachedllm_instances(tmp_path):
    inner_one = _CountingLLM()
    CachedLLM(inner_one, model_id="m", cache_dir=tmp_path).complete(prompt="p")

    inner_two = _CountingLLM()
    CachedLLM(inner_two, model_id="m", cache_dir=tmp_path).complete(prompt="p")

    assert inner_two.calls == 0  # served from the same on-disk cache


def test_creates_the_cache_directory(tmp_path):
    cache_dir = tmp_path / "nested" / "llm"
    CachedLLM(_CountingLLM(), model_id="m", cache_dir=cache_dir).complete(prompt="p")
    assert cache_dir.exists()
