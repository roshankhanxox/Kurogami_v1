"""Deterministic, offline SearchPort implementation. No network, ever."""

from kurogami.contracts.ports import SearchHit


class FakeSearch:
    """Satisfies SearchPort by returning fixed hits, ignoring the query by default."""

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or [
            SearchHit(
                title="Fake search result",
                url="https://example.invalid/fake-result",
                snippet="Deterministic placeholder hit returned by FakeSearch.",
            )
        ]

    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]:
        return self._hits[:max_results]
