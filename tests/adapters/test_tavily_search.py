"""TavilySearch: request shape and response mapping, offline via an injected post()."""

import pytest

from kurogami.adapters.search.tavily import TAVILY_SEARCH_URL, TavilySearch


class _StubPost:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple] = []

    def __call__(self, url, body, headers, timeout_s):
        self.calls.append((url, body, headers, timeout_s))
        return self.payload


def test_sends_the_query_with_a_bearer_token():
    post = _StubPost({"results": []})
    TavilySearch("tvly-test", post=post).search("invoicing tools India", max_results=3)

    [(url, body, headers, _)] = post.calls
    assert url == TAVILY_SEARCH_URL
    assert body == {"query": "invoicing tools India", "max_results": 3, "search_depth": "basic"}
    assert headers["Authorization"] == "Bearer tvly-test"


def test_maps_results_to_search_hits():
    post = _StubPost(
        {
            "results": [
                {"title": "Zoho Invoice", "url": "https://zoho.com/in/invoice", "content": "Free plan", "score": 0.9},
                {"title": "Refrens", "url": "https://refrens.com", "content": "GST invoices"},
            ]
        }
    )
    hits = TavilySearch("tvly-test", post=post).search("q")

    assert [(h.title, h.url, h.snippet) for h in hits] == [
        ("Zoho Invoice", "https://zoho.com/in/invoice", "Free plan"),
        ("Refrens", "https://refrens.com", "GST invoices"),
    ]


def test_never_returns_more_than_asked():
    post = _StubPost({"results": [{"title": str(i), "url": "u", "content": "c"} for i in range(9)]})
    assert len(TavilySearch("k", post=post).search("q", max_results=2)) == 2


def test_requires_a_key():
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        TavilySearch("")
