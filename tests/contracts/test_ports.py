"""Ports import cleanly and the fakes satisfy them structurally, offline."""

from kurogami.adapters.llm.fake import FakeLLM
from kurogami.adapters.search.fake import FakeSearch
from kurogami.contracts import (
    ClockPort,
    ExecutorPort,
    InterruptPort,
    LLMPort,
    LLMResponse,
    PlannerPort,
    RuntimePort,
    SearchHit,
    SearchPort,
    TraceSink,
    VerifierPort,
)


def test_ports_import_cleanly():
    for port in (
        LLMPort,
        SearchPort,
        TraceSink,
        InterruptPort,
        RuntimePort,
        PlannerPort,
        ExecutorPort,
        VerifierPort,
        ClockPort,
    ):
        assert hasattr(port, "__mro__") or hasattr(port, "_is_protocol")


def test_fake_llm_satisfies_llm_port_shape():
    llm = FakeLLM(responses={"interpret": "hello"})
    response = llm.complete(prompt="interpret", prompt_name="interpret")
    assert isinstance(response, LLMResponse)
    assert response.text == "hello"


def test_fake_search_satisfies_search_port_shape():
    search = FakeSearch()
    hits = search.search("freelance designer invoicing India", max_results=3)
    assert isinstance(hits, list)
    assert all(isinstance(hit, SearchHit) for hit in hits)
