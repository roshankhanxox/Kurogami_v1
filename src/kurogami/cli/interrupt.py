"""CliInterrupt: prompts the terminal for human-injected constraints.

Lives in cli/, not engine/interrupt.py, because it does real terminal I/O
and CLAUDE.md section 3 restricts print()/input() to cli/. See
engine/interrupt.py for the pure, offline ScriptedInterrupt counterpart.
"""

from kurogami.contracts import NodeSpec


class CliInterrupt:
    """Satisfies InterruptPort. Pauses at the given node ids and prompts the terminal."""

    def __init__(self, node_ids: list[str]) -> None:
        self._node_ids = set(node_ids)

    def should_pause(self, node: NodeSpec) -> bool:
        return node.node_id in self._node_ids

    def collect(self, node: NodeSpec) -> list[str]:
        print(f"\n-- paused at {node.node_id} ({node.title}) --")
        print("Enter one constraint per line. Blank line to finish.")
        constraints: list[str] = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            constraints.append(line)
        return constraints
