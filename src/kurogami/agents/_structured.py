"""Reading `structured` payloads and the assertions over them. Shared by the agents.

Deterministic, no LLM: the executor, rule checker, planner and localiser all have
to agree on which keys an assertion reads and which ancestor values it compares
against, so there is exactly one implementation of each.
"""

import ast
import json
import re
from typing import Any

_FENCED_JSON_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
ANCESTORS_NAME = "ancestors"


def extract_structured(text: str) -> dict[str, Any]:
    """The JSON object in a model's answer: fenced block, whole text, or outermost braces."""
    candidates: list[str] = []
    fenced = _FENCED_JSON_PATTERN.search(text)
    if fenced is not None:
        candidates.append(fenced.group(1))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def ancestor_payloads(context: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Each ancestor's structured payload, recovered from the context the engine assembled.

    Context values are either a direct parent's full answer (prose + JSON block) or a
    distant ancestor's structured payload serialised as JSON; both parse here. Keys
    starting with "_" are engine annotations (e.g. a backtrack reason), not ancestors.
    """
    return {
        node_id: extract_structured(text)
        for node_id, text in context.items()
        if not node_id.startswith("_")
    }


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _constant(node: ast.AST) -> object:
    return node.value if isinstance(node, ast.Constant) else None


def _parse(assertion: str) -> ast.Expression | None:
    try:
        return ast.parse(assertion, mode="eval")
    except SyntaxError:
        return None


def required_structured_keys(assertions: list[str]) -> list[str]:
    """Top-level keys the assertions read from `structured`, in order.

    Read from the parsed expression, not a regex: seen live, once `.get` was
    allowed, `structured.get('competitors')` slipped past a subscript-only regex,
    the model was never told the key, named it `tools`, and every run failed.
    Covers structured['k'], structured.get('k'), 'k' in structured, and a
    literal list of keys tested with `k in structured`.
    """
    keys: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value not in keys:
            keys.append(value)

    for assertion in assertions:
        tree = _parse(assertion)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and _is_name(node.value, "structured"):
                add(_constant(node.slice))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and _is_name(node.func.value, "structured")
                and node.args
            ):
                add(_constant(node.args[0]))
            elif isinstance(node, ast.Compare) and any(
                isinstance(op, ast.In | ast.NotIn) for op in node.ops
            ) and any(_is_name(c, "structured") for c in node.comparators):
                add(_constant(node.left))
                if isinstance(node.left, ast.Name):  # `k in structured for k in [...]`
                    for gen in ast.walk(tree):
                        if (
                            isinstance(gen, ast.comprehension)
                            and isinstance(gen.target, ast.Name)
                            and gen.target.id == node.left.id
                            and isinstance(gen.iter, ast.List | ast.Tuple)
                        ):
                            for element in gen.iter.elts:
                                add(_constant(element))
    return keys


def _ancestor_id(node: ast.AST) -> str | None:
    """`ancestors['a']` or `ancestors.get('a', ...)` -> 'a'."""
    if isinstance(node, ast.Subscript) and _is_name(node.value, ANCESTORS_NAME):
        value = _constant(node.slice)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_name(node.func.value, ANCESTORS_NAME)
        and node.args
    ):
        value = _constant(node.args[0])
    else:
        return None
    return value if isinstance(value, str) else None


def ancestor_references(assertion: str) -> list[tuple[str, str | None]]:
    """(ancestor_id, key) pairs an assertion reads; key is None if not a literal read.

    Recognises ancestors['a']['k'], ancestors['a'].get('k'), and the same with
    ancestors.get('a', {}) in place of the subscript.
    """
    tree = _parse(assertion)
    if tree is None:
        return []
    refs: list[tuple[str, str | None]] = []
    keyed: set[int] = set()
    for node in ast.walk(tree):
        inner: ast.AST | None = None
        key: object = None
        if isinstance(node, ast.Subscript):
            inner, key = node.value, _constant(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            inner, key = node.func.value, _constant(node.args[0])
        if inner is None:
            continue
        ancestor = _ancestor_id(inner)
        if ancestor is not None and isinstance(key, str):
            refs.append((ancestor, key))
            keyed.add(id(inner))
    for node in ast.walk(tree):
        ancestor = _ancestor_id(node)
        if ancestor is not None and id(node) not in keyed:
            refs.append((ancestor, None))
    return list(dict.fromkeys(refs))


def resolved_ancestor_values(assertions: list[str], context: dict[str, str]) -> list[str]:
    """Human-readable `ancestors['a']['k'] = value` lines for every literal ancestor read."""
    payloads = ancestor_payloads(context)
    lines: list[str] = []
    for assertion in assertions:
        for ancestor, key in ancestor_references(assertion):
            if key is None:
                continue
            payload = payloads.get(ancestor)
            if payload is None:
                shown = "(not available -- that ancestor's answer is not in your context)"
            elif key not in payload:
                shown = "(not available -- that ancestor did not report this key)"
            else:
                shown = json.dumps(payload[key])
            line = f"ancestors['{ancestor}']['{key}'] = {shown}"
            if line not in lines:
                lines.append(line)
    return lines


def _reads_key(node: ast.AST, key: str) -> bool:
    """structured['key'] or structured.get('key', ...)."""
    if isinstance(node, ast.Subscript) and _is_name(node.value, "structured"):
        return _constant(node.slice) == key
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_name(node.func.value, "structured")
        and bool(node.args)
        and _constant(node.args[0]) == key
    )


def example_value(key: str, assertions: list[str]) -> object:
    """A placeholder whose JSON type matches what the checks do with this key.

    Seen live: the example layout showed "..." for every key, the model copied it,
    wrote a sentence where `structured['late_payment_frequency'] >= 0` needed a
    number, and repeated it on all three retries.
    """
    for assertion in assertions:
        tree = _parse(assertion)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call | ast.Compare):
                continue
            call = node if isinstance(node, ast.Call) and node.args else None
            if call is not None and _is_name(call.func, "len") and _reads_key(call.args[0], key):
                return ["..."]
            if (
                call is not None
                and _is_name(call.func, "isinstance")
                and len(call.args) == 2
                and _reads_key(call.args[0], key)
            ):
                names = {n.id for n in ast.walk(call.args[1]) if isinstance(n, ast.Name)}
                if "list" in names:
                    return ["..."]
                if "dict" in names:
                    return {"...": "..."}
                if names & {"int", "float"}:
                    return 0
                if "bool" in names:
                    return True
            if isinstance(node, ast.Compare):
                sides = [node.left, *node.comparators]
                if not any(_reads_key(side, key) for side in sides):
                    continue
                if any(isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE) for op in node.ops):
                    return 0
                for op, right in zip(node.ops, node.comparators, strict=True):
                    if isinstance(op, ast.In) and isinstance(right, ast.List | ast.Tuple):
                        options = [_constant(e) for e in right.elts]
                        if options and options[0] is not None:
                            return options[0]
    return "..."


def describe_values(assertion: str, structured: dict[str, Any]) -> str:
    """What this node actually reported for the keys a check reads, with types."""
    parts = []
    for key in required_structured_keys([assertion]):
        if key not in structured:
            parts.append(f"'{key}' is missing")
            continue
        value = structured[key]
        shown = json.dumps(value)
        if len(shown) > 120:
            shown = shown[:117] + "..."
        parts.append(f"'{key}' is a {type(value).__name__}: {shown}")
    return "; ".join(parts)
