"""Internal helper: load prompts/<name>.md by name.

Lives inside agents/ (not as an import of the prompts/ package) because
CLAUDE.md section 2 R1 restricts agents/ to importing only contracts/.
prompts/ stays pure versioned data with no code of its own.
"""

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class UnknownPromptError(FileNotFoundError):
    """Raised when a prompt file is asked for that doesn't exist."""


def load_prompt(name: str) -> str:
    """Read prompts/<name>.md verbatim."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise UnknownPromptError(f"No prompt file named {name}.md in {_PROMPTS_DIR}")
    return path.read_text()


def prompt_version(name: str) -> str:
    """filename + content hash, per ARCHITECTURE.md P5: every trace record carries this."""
    content = load_prompt(name)
    digest = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{name}.md@{digest}"
