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

_ALLOWED_NAMES = frozenset({"structured", "context", "len", "any", "all"})
_ALLOWED_CALL_NAMES = frozenset({"len", "any", "all"})

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


def _validate(tree: ast.AST) -> None:
    locally_bound = _comprehension_bound_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeAssertionError(f"assertion uses disallowed syntax: {type(node).__name__}")
        if (
            isinstance(node, ast.Name)
            and node.id not in _ALLOWED_NAMES
            and node.id not in locally_bound
        ):
            raise UnsafeAssertionError(f"assertion references disallowed name: {node.id!r}")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALL_NAMES
        ):
            raise UnsafeAssertionError("assertion calls a disallowed function")


def _safe_eval(expression: str, structured: dict[str, Any], context: dict[str, str]) -> bool:
    tree = ast.parse(expression, mode="eval")
    _validate(tree)
    compiled = compile(tree, filename="<pass_condition.assertion>", mode="eval")
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_locals: dict[str, Any] = {
        "structured": structured,
        "context": context,
        "len": len,
        "any": any,
        "all": all,
    }
    return bool(eval(compiled, safe_globals, safe_locals))


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
            except (TypeError, KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
                return self._fail(
                    node, "assertion", f"assertion raised {type(exc).__name__}: {exc}", assertion
                )
            if not passed:
                return self._fail(node, "assertion", "a deterministic assertion failed", assertion)
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
