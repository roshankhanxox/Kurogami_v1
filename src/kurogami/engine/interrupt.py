"""ScriptedInterrupt: pause at exactly the given node ids with pre-scripted constraints.

Pure and offline -- no terminal I/O, no side effects -- so it's usable in
tests and reproducible in a demo (IMPLEMENTATION_PLAN.md Track C: "Build
ScriptedInterrupt before CliInterrupt... interactive typing on stage is a
liability. Have both, demo the scripted one.").

Note: IMPLEMENTATION_PLAN.md's file table lists CliInterrupt here too, but
CliInterrupt has to print to the terminal and read from stdin, and CLAUDE.md
section 3 is explicit: "No print() outside cli/. Use the trace sink or a
logger." CliInterrupt therefore lives in cli/interrupt.py instead, where
that's allowed; this file stays print-free.
"""

from kurogami.contracts import NodeSpec


class ScriptedInterrupt:
    """Satisfies InterruptPort. Pauses only at the given node ids."""

    def __init__(self, node_ids: list[str], constraints: dict[str, list[str]] | None = None) -> None:
        self._node_ids = set(node_ids)
        self._constraints = constraints or {}

    def should_pause(self, node: NodeSpec) -> bool:
        return node.node_id in self._node_ids

    def collect(self, node: NodeSpec) -> list[str]:
        return list(self._constraints.get(node.node_id, []))
