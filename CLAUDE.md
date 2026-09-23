# CLAUDE.md — working agreement for this repository

> Read this before writing a single line. It applies equally to human contributors
> and to any coding agent operating in this repo.
> Companion documents: `docs/ARCHITECTURE.md` (the design),
> `docs/IMPLEMENTATION_PLAN.md` (the build order), `docs/CHECKPOINTS.md` (the gates).

---

## 1. What this project is

Kurogami is an **agent orchestration framework**. A Master Agent writes the prompts
for its own child agents, arranges them as a dependency tree, verifies each node
against a goal declared for that node, and on failure localises the responsible
ancestor, invalidates its subtree, and regenerates it.

The domain — market-entry strategy for early-stage products — is a **test vehicle,
not the product**. No market-entry knowledge belongs in Python. It lives in
`src/kurogami/prompts/`. If you are tempted to write a function called
`calculate_tam()`, stop: you are putting domain logic in the engine and breaking
the Sem-8 expansion story.

---

## 2. Non-negotiable rules

### R1 — Dependency direction

```
contracts/  ←  agents/  ←  engine/  ←  cli/
     ↑            ↑
     └── adapters/┘
```

- `contracts/` imports nothing from `kurogami.*`. Ever.
- `agents/` imports only `contracts/`.
- `engine/` imports `contracts/` and `agents/`.
- `adapters/` imports only `contracts/`.
- Concrete adapters are constructed **only** in `cli/main.py` and
  `bench/harness.py`. Nowhere else.

A needed import that points outward means a missing contract. Add the contract.

### R2 — Typed contracts at every seam

Anything crossing a module boundary is a pydantic model from `contracts/`. No
dicts, no tuples, no bare strings. If you need to pass something new, add a field
to a contract via PR — don't smuggle it in a `dict[str, Any]`.

### R3 — Agents are tree-blind

An agent takes a `NodeSpec` and returns a `NodeResult`. It must not import
`TreeStore`, must not know its parent, must not read siblings. Ancestry reaches it
only through `NodeSpec.context`, assembled by the engine. Breaking this means a
node cannot be re-run inside a regenerated subtree — which breaks the core claim.

### R4 — Every external dependency sits behind a port, and every port has a fake

New LLM provider, search API, file store, clock, RNG → define the `Protocol` in
`contracts/ports.py` first, then write the real adapter **and** a `Fake`
implementation. Non-negotiable consequence: **the entire test suite runs offline
with no API key.** A test that makes a network call is a broken test.

### R5 — Prompts are files

All prompts live in `src/kurogami/prompts/*.md`, loaded by name via the prompt
loader. Never inline a prompt as a Python string literal. Every trace record
carries the prompt filename and content hash.

### R6 — Nothing silently swallowed

Every node execution, verdict, backtrack and interrupt emits a `TraceRecord`.
No bare `except:`. No `except Exception: pass`. If you catch, you log the trace
record and re-raise or return a typed failure.

### R7 — Budget breaches are outcomes, not crashes

Depth cap, node cap, backtrack cap, token cap, retry cap, loop detector — all in
`engine/budget.py`. On breach: mark remaining nodes `SKIPPED`, emit a terminal
trace record with the reason, exit non-zero, **do not raise an unhandled
exception**. A demo must be able to survive a budget breach gracefully.

### R8 — The held-out split is sealed

`bench/goals/heldout/` must not be read, printed, opened, or used for tuning
until the evaluation phase (Stage 4). This is stated on Slides 6, 7 and 9. If a
coding agent is asked to "look at all the benchmark goals", it looks at
`bench/goals/dev/` only. Violating this invalidates every number we report.

---

## 3. Code conventions

- **Python 3.11+**, `pydantic` v2, `typer` for CLI, `rich` for rendering,
  `pytest` for tests. Add no other runtime dependency without team agreement —
  every dependency is a thing that can break during a demo.
- Type hints on every public function. `mypy --strict` on `contracts/` and
  `engine/` (the rest is best-effort).
- `ruff` for lint and format. Line length 100.
- Docstrings: one line saying *why*, not *what*. The type signature says what.
- No `print()` outside `cli/`. Use the trace sink or a logger.
- No `eval()` on model output. `agents/rules.py` evaluates assertions in an
  explicitly whitelisted namespace — treat every string from an LLM as hostile.
- Secrets from `.env` via `python-dotenv`. **Never commit a key.** `.env` is
  gitignored; `.env.example` lists variable names with empty values.

---

## 4. Git protocol

### Branching — group features, don't branch per file

Branch at the level of a **coherent unit of work that can be merged and
demonstrated on its own**. Roughly: one branch per module cluster, per owner.
Not one branch per file (merge noise); not one branch for everything
(unreviewable, and everyone blocks on everyone).

Current branches:

| Branch | Owner | Scope |
|---|---|---|
| `feat/engine-core` | Engine Owner | `engine/*` — store, context, scheduler, backtrack, budget, runner |
| `feat/planning-agents` | Planning Owner | `agents/interpreter.py`, `agents/planner.py`, `agents/executor.py`, `prompts/*` |
| `feat/verification-hitl` | Verification Owner | `agents/verifier.py`, `agents/rules.py`, `agents/localiser.py`, `engine/interrupt.py`, `cli/*`, `adapters/trace/*` |
| `feat/bench-harness` | Bench Owner | `adapters/llm/*`, `adapters/search/*`, `bench/*` |

Naming:

```
feat/<area>-<thing>      new capability        feat/engine-core
fix/<area>-<thing>       bug in merged code    fix/backtrack-orphan-nodes
exp/<what>               throwaway experiment  exp/planner-depth-prompting
docs/<what>              docs only             docs/architecture-update
```

Rules:

- **Branch off `main`, rebase onto `main` before opening a PR.** Do not merge
  `main` into your branch repeatedly — the history becomes unreadable.
- **One owner per branch.** If two people need the same file, the design is wrong
  — that is a seam that should be a contract.
- **Never commit directly to `main`** after the foundation merge.
- A branch that has accumulated several unrelated changes is too big. Split it.
  A branch should merge back while its diff is still reviewable in one sitting.
- Delete the branch after merge.
- `exp/` branches are expected to be deleted unmerged. That is what they are for.

### Commits

```
<area>: <imperative summary under 70 chars>

<why, if not obvious from the diff>
```

Examples:

```
engine: invalidate descendants on backtrack, preserve them in store
planner: cap children at 3 and require ancestor-referencing pass conditions
fix(verifier): return suspect_node_ids as ancestors, not the failing node
```

Commit at working checkpoints, not in one lump at the end. A commit that doesn't
pass `pytest` shouldn't exist on a shared branch.

### Pull requests

Even with four people, open PRs — the review is where architectural drift gets
caught. A PR must state:

1. Which gate in `docs/CHECKPOINTS.md` it advances.
2. Which tests prove it.
3. Whether it touches `contracts/` (if yes → Engine Owner reviews, no exceptions).

`contracts/` is **frozen** once the foundation merges. Changing it breaks every
other branch silently. If it truly must change, say so in the group chat *before*
opening the PR.

---

## 5. Testing

- `tests/` mirrors `src/kurogami/` structure.
- **Every test runs offline.** `conftest.py` provides `FakeLLM`, `FakeSearch`,
  and the 12-node fixture tree. A test that needs a key is a broken test.
- The highest-value tests in this repo, in order:
  1. `test_invalidate_kills_only_descendants` — the core claim
  2. `test_localiser_picks_shallowest_ancestor` — the hard part
  3. `test_context_excludes_siblings` — prevents contamination
  4. `test_budget_stops_cleanly_not_crash` — demo survival
  5. `test_full_run_with_fake_llm` — integration
- Prompt behaviour is **not** unit-tested (non-deterministic). Prompts are
  evaluated through `bench/`, on dev goals, with results recorded.

---

## 6. Instructions specific to coding agents

If you are an AI agent working in this repo:

1. **Read `docs/ARCHITECTURE.md` §2, §4 and §5 before proposing code.** The
   contracts in §4 are literal — implement them as written, don't improve them.
2. **Do not add dependencies.** If a task seems to need one, say so and stop.
3. **Do not refactor across module boundaries** without being asked. A
   "helpful" cleanup that moves a function from `engine/` to `agents/` breaks R1.
4. **Do not touch `contracts/`** unless the task explicitly says to.
5. **Do not read `bench/goals/heldout/`** (R8). If a task seems to require it,
   refuse and explain.
6. **Write the test in the same change as the code.** Not afterwards.
7. When you finish, state which gate in `docs/CHECKPOINTS.md` the work advances
   and which command demonstrates it.
8. **Work on a feature branch matching §4**, never on `main`. If the current
   branch is `main`, create one first.
9. If a requested change would violate R1–R8, **say so rather than doing it**.
   The rules exist because the examiner will probe exactly these seams.

---

## 7. Demo discipline

- **Code freeze once gate G8 passes.** After that, documentation only.
- Tag the demo build: `git tag -a v0.1-midsem`.
- Never `pip install` during a demo.
- The demo goal (`bench/goals/dev/g001.json`) is frozen. Do not edit it.
- If something breaks live, say what broke and show the recorded trace. An honest
  "here's what failed and here's why" scores better than a rigged run — and the
  risk register on Slide 12 already told them this was hard.
