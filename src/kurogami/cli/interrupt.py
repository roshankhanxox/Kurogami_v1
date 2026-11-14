"""CliInterrupt: prompts the terminal for human-injected constraints.

Lives in cli/, not engine/interrupt.py, because it does real terminal I/O
and CLAUDE.md section 3 restricts print()/input() to cli/. See
engine/interrupt.py for the pure, offline ScriptedInterrupt counterpart.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kurogami.contracts import NodeSpec


class CliInterrupt:
    """Satisfies InterruptPort. Pauses at the given node ids and prompts the terminal."""

    def __init__(self, node_ids: list[str], console: Console | None = None) -> None:
        self._node_ids = set(node_ids)
        self._console = console if console is not None else Console()

    def should_pause(self, node: NodeSpec) -> bool:
        return node.node_id in self._node_ids

    def collect(self, node: NodeSpec) -> list[str]:
        message = Text(f"Run paused at {node.node_id} · {node.node_goal}\n")
        message.append("Enter one constraint per line. Press Enter on a blank line to continue.")
        self._console.print(Panel(message, title="Your input", border_style="yellow"))
        constraints: list[str] = []
        while True:
            try:
                line = self._console.input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            constraints.append(line)
        if constraints:
            self._console.print("Constraint saved. Continuing with the affected tasks.")
        return constraints
