"""Evaluate PassCondition.assertions against NodeResult.structured, safely.

Assertions are strings written by an LLM (the planner) -- hostile input by
default (CLAUDE.md section 3: "treat every string from an LLM as hostile").
This never calls eval() on a raw, unvalidated string: every assertion is
first walked as an AST and rejected if it uses anything outside a small
whitelist, then evaluated with no builtins and only the whitelisted names
bound.
"""

import ast
from typing import Any, Literal

from kurogami.contracts import FailureReason, NodeResult, NodeSpec, Verdict

# Pure, side-effect-free builtins only. The type names are here so isinstance()
# can be used; nothing that imports, opens, reflects, or allocates unboundedly.
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "any": any,
    "all": all,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "set": set,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "isinstance": isinstance,
}
ALLOWED_FUNCTIONS = tuple(_SAFE_BUILTINS)
_ALLOWED_NAMES = frozenset({"structured", "context", *_SAFE_BUILTINS})
_ALLOWED_CALL_NAMES = frozenset(_SAFE_BUILTINS)

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Store,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.comprehension,
)


class UnsafeAssertionError(ValueError):
    """Raised when an assertion string uses syntax outside the whitelist."""


def _comprehension_bound_names(tree: ast.AST) -> set[str]:
    """Loop variables bound by a comprehension (e.g. the `x` in `... for x in ...`).

    These are arbitrary, model-chosen identifiers, locally scoped to the
    comprehension -- safe regardless of name, unlike a free-standing Name
    reference, which must be in _ALLOWED_NAMES.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return bound


# Read-only dict methods. Seen live: gpt-4.1-mini writes defensive assertions such as
# `isinstance(structured.get('tools'), list)` -- ~40 of them in one plan, all rejected,
# and the mass correction that followed degraded good prompts. Only these exact
# attribute names are reachable; nothing dunder, nothing that mutates. On a non-dict
# they raise AttributeError, which check() turns into a typed FAIL.
_ALLOWED_METHODS = frozenset({"get", "keys", "values", "items"})


def _validate(tree: ast.AST) -> None:
    locally_bound = _comprehension_bound_names(tree)
    method_calls = {
        id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr not in _ALLOWED_METHODS or id(node) not in method_calls:
                raise UnsafeAssertionError(
                    f"assertion uses a disallowed attribute: .{node.attr} "
                    f"(only calls to {', '.join(sorted(_ALLOWED_METHODS))})"
                )
            continue
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeAssertionError(f"assertion uses disallowed syntax: {type(node).__name__}")
        if (
            isinstance(node, ast.Name)
            and node.id not in _ALLOWED_NAMES
            and node.id not in locally_bound
        ):
            raise UnsafeAssertionError(f"assertion references disallowed name: {node.id!r}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute) and (
            not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALL_NAMES
        ):
            func = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
            raise UnsafeAssertionError(f"assertion calls a disallowed function: {func}")


def validate_assertion(expression: str) -> None:
    """Raise UnsafeAssertionError or SyntaxError if this assertion could never be evaluated."""
    _validate(ast.parse(expression, mode="eval"))


def _safe_eval(expression: str, structured: dict[str, Any], context: dict[str, str]) -> bool:
    tree = ast.parse(expression, mode="eval")
    _validate(tree)
    compiled = compile(tree, filename="<pass_condition.assertion>", mode="eval")
    # Comprehensions run in their own scope and only see globals, so the
    # whitelisted names live in globals (with __builtins__ still emptied).
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "structured": structured,
        "context": context,
        **_SAFE_BUILTINS,
    }
    return bool(eval(compiled, safe_globals))


class RuleChecker:
    """Cheap, deterministic PASS/FAIL. Runs before the semantic verifier (ARCHITECTURE.md 5)."""

    def check(self, node: NodeSpec, result: NodeResult) -> Verdict:
        for assertion in node.pass_condition.assertions:
            try:
                passed = _safe_eval(assertion, result.structured, node.context)
            except (UnsafeAssertionError, SyntaxError) as exc:
                # SyntaxError means the planner wrote a natural-language sentence
                # instead of a Python expression (seen live against a real LLM,
                # e.g. "The output includes a summary of current tools used...").
                # Same bucket as UnsafeAssertionError: the assertion string itself
                # is malformed, not a runtime failure against valid data.
                return self._fail(node, "schema", str(exc), assertion)
            except (
                TypeError, KeyError, IndexError, ValueError, ZeroDivisionError, AttributeError
            ) as exc:
                return self._fail(
                    node, "assertion", f"assertion raised {type(exc).__name__}: {exc}", assertion
                )
            if not passed:
                return self._fail(
                    node, "assertion", f"assertion evaluated to False: {assertion}", assertion
                )
        return Verdict(node_id=node.node_id, verdict="PASS", checked_by="rules")

    @staticmethod
    def _fail(
        node: NodeSpec, violated: Literal["assertion", "semantic", "schema"], summary: str, evidence: str
    ) -> Verdict:
        return Verdict(
            node_id=node.node_id,
            verdict="FAIL",
            checked_by="rules",
            reason=FailureReason(summary=summary, violated=violated, evidence=evidence),
        )
