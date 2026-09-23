"""Tavily adapter satisfying SearchPort. Real web results for RESEARCH nodes.

Seen live: with FakeSearch a research node's "results" were a placeholder, so it
answered from model memory and a check like "names the main competitors" had
nothing to be grounded in. Uses the standard library only (no new dependency).
"""

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from kurogami.contracts.ports import SearchHit

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# (url, body, headers, timeout_s) -> parsed JSON response
Post = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _urllib_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        parsed: dict[str, Any] = json.loads(response.read())
        return parsed


class TavilySearch:
    """SearchPort over api.tavily.com. `post` is injectable so tests never hit the network."""

    def __init__(
        self,
        api_key: str,
        *,
        search_depth: str = "basic",
        timeout_s: float = 30.0,
        post: Post = _urllib_post,
    ) -> None:
        if not api_key:
            raise ValueError("TavilySearch needs an API key (TAVILY_API_KEY)")
        self._api_key = api_key
        self._search_depth = search_depth
        self._timeout_s = timeout_s
        self._post = post

    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]:
        payload = self._post(
            TAVILY_SEARCH_URL,
            {"query": query, "max_results": max_results, "search_depth": self._search_depth},
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            self._timeout_s,
        )
        return [
            SearchHit(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
            for r in payload.get("results", [])
        ][:max_results]
