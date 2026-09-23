# Kurogami — Implementation Plan

> **Structure:** this plan is ordered by **dependency**, not by calendar. Stages
> run in sequence; tracks inside a stage run in parallel. A stage is finished when
> its gate in `docs/CHECKPOINTS.md` passes — not when time runs out.
> **Team:** Engine Owner, Planning Owner, Verification Owner, Bench Owner.

---

## 0. The only thing that matters for the demo

The evaluator needs to see **one run** in which:

1. A single English sentence goes in.
2. The system prints **prompts it wrote itself** for child agents.
3. A tree of **depth ≥ 4** appears, built at runtime.
4. A verifier says **FAIL** on a node with a stated reason.
5. The system **backtracks to an ancestor**, kills its subtree, and rebuilds it.
6. A **human interrupt** at a node visibly changes what gets regenerated.
7. A JSONL trace exists on disk that the evaluator can open.

Everything in this plan is subordinate to that list. If a task does not move one
of those seven items forward, it belongs in **Stage 4** — do not start it.

### What is explicitly out of the pre-demo stages

Full 30-goal benchmark · held-out split · human rating protocol · baseline
comparison · LangGraph adapter · metric targets · model comparison · cost study.
All of these are semester work with real runway. Pulling any of them forward
competes directly with item 5 above, which is the one thing that cannot be faked.

---

## Stage 0 — Foundation (all four, together)

This is the highest-leverage block of the entire project. Everything after it runs
in parallel; nothing before it can.

**Branch:** `setup/foundation` → merged to `main` before anyone splits off.

| # | Task | Who | Done when |
|---|---|---|---|
| 1 | `git init`, `pyproject.toml`, package skeleton, `.env.example`, `.gitignore` | Engine Owner | `pip install -e ".[dev]"` succeeds |
| 2 | Write **every file in `contracts/`** — verbatim from ARCHITECTURE.md §4 | Engine Owner drives, all 4 in the room | `pytest tests/contracts` green |
| 3 | Write **every `Protocol` in `contracts/ports.py`** | same | models import cleanly |
| 4 | Write `FakeLLM` returning canned structured responses | Bench Owner | `FakeLLM().complete(...)` returns a parsed model |
| 5 | Agree the demo goal, write it down, **never change it** | all | committed to `bench/goals/dev/g001.json` |
| 6 | Create the four feature branches (§0.3) | Engine Owner | branches pushed |

### 0.1 The demo goal — freeze this immediately

> *"I built a tool that helps freelance designers in India send invoices and chase
> late payments. Should I launch it, and how?"*

Chosen because its natural decomposition **forces a contradiction**: market sizing
establishes that Indian freelance designers are price-sensitive; a naive
differentiation node pushes "premium compliance features"; pricing then produces
tiers that violate the willingness-to-pay ceiling. The failure is real, not
planted — which is a much better demo than `--inject-fault`.

**Gate — nobody leaves this block until:**
```bash
pytest tests/contracts -q        # green
python -c "from kurogami.contracts import NodeSpec, Verdict, TraceRecord"
```

### 0.2 Contracts freeze rule

Once Stage 0 merges, `contracts/` is **frozen**. A change requires a PR that
Engine Owner reviews, because every other branch is built on those types. One person
quietly editing `NodeSpec` breaks three other branches silently. This rule is the
difference between four parallel workstreams and four merge conflicts.

### 0.3 Branch layout

Four branches, one per coherent feature group — not one per file, not one giant
branch. Each is independently mergeable and independently demonstrable, which
maps directly onto Slide 11's "what each member personally demonstrates."

```
main
├── feat/engine-core          Engine Owner          store, context, scheduler, backtrack, runner, budget
├── feat/planning-agents      Planning Owner        interpreter, planner, prompt pack
├── feat/verification-hitl    Verification Owner    verifier, rules, localiser, interrupt, CLI, trace
└── feat/bench-harness        Bench Owner           LLM adapters, dev goals, harness skeleton, metrics
```

Merge order in Stage 2 is **engine → planning → verification → bench**, because
that is the dependency order. See CLAUDE.md §4 for the full git protocol.

---

## Stage 1 — Four parallel tracks

Each track below is written so its owner can work alone against `contracts/` with
`FakeLLM`, touching no one else's files.

### Track A — `feat/engine-core` · Engine Owner

**Goal: the tree mechanism works with zero LLM calls.**

| Order | File | What | Test that proves it |
|---|---|---|---|
| A1 | `engine/store.py` | `TreeStore`: `seed`, `attach`, `get`, `children`, `ancestors`, `descendants`, `mark_*`, `invalidate_subtree`, `snapshot` | `test_invalidate_kills_only_descendants` |
| A2 | `engine/context.py` | assemble ancestor outputs into `NodeSpec.context`; prefer `structured`; truncate by depth distance | `test_context_excludes_siblings` |
| A3 | `engine/scheduler.py` | `next()` → DFS, parents-before-children, skips non-`PENDING` | `test_scheduler_respects_dependencies` |
| A4 | `engine/budget.py` | all six limits from ARCHITECTURE §8 + loop detector | `test_budget_stops_cleanly_not_crash` |
| A5 | `engine/backtrack.py` | `locate()` → shallowest ancestor suspect; `apply()` → invalidate + requeue | `test_localiser_picks_shallowest_ancestor` |
| A6 | `engine/runner.py` | the §5 loop, all deps injected via ports | `test_full_run_with_fake_llm` |

**A1 and A5 are the most valuable code in the repository.** They are the claim.
Write them first, test them hardest.

**Hand-build a fixture tree in `conftest.py`** — 12 nodes, depth 5, mirroring the
manual trace already on Slide 14. Every engine test runs against it, offline, in
milliseconds.

**Gate A** — `pytest tests/engine -q` green, and:
```bash
python -m kurogami.engine.demo_tree   # prints fixture tree, forces a FAIL at
                                      # pricing, shows subtree invalidation
```
**Gate A is blocking.** If it does not pass, every other track stops and all four
people work on it. There is no demo without it — everything else is decoration
around this mechanism.

### Track B — `feat/planning-agents` · Planning Owner

**Goal: an LLM writes child-agent prompts, unaided, and they are well-formed.**

| Order | File | What |
|---|---|---|
| B1 | `prompts/interpret.md` | sentence → `GoalSpec`; must flag ambiguities rather than resolve them |
| B2 | `agents/interpreter.py` | `Interpreter(llm).run(text) -> GoalSpec`, schema-constrained |
| B3 | `prompts/plan.md` | `GoalSpec` → 2–3 **root** nodes. Must emit `node_goal`, `generated_prompt`, `pass_condition.assertions`, `pass_condition.semantic_check` per node |
| B4 | `agents/planner.py` | `.plan(goal)` and `.expand(node, result, goal)` |
| B5 | `prompts/expand.md` | given a passed node's output, decide children. **Must be allowed to return zero children** — that is how the tree terminates |
| B6 | `agents/executor.py` | generic: takes `NodeSpec.generated_prompt` + context, returns `NodeResult`; `SearchPort` only for `kind=RESEARCH` |

**The hard part is B3/B5, and it is a prompting problem, not a coding problem.**
Budget your effort accordingly. Two specific failure modes to design against:

- *Bushy shallow trees.* The planner emits 8 siblings at depth 1 and stops.
  Fix: instruct it to emit **at most 3 children**, and to prefer a child that
  *depends on* a sibling's output over a parallel sibling. Depth is the deliverable.
- *Vacuous pass conditions.* `"assertions": ["output is not empty"]`. Useless.
  Fix: require every `pass_condition` to reference **at least one named ancestor
  node's output**. That single constraint is what makes downstream contradiction
  detectable at all.

**Gate B**
```bash
kurogami plan --goal-file bench/goals/dev/g001.json --dry-run
```
prints a tree of depth ≥ 4, ≥ 10 nodes, every node carrying a non-trivial
generated prompt and a pass condition that names an ancestor.

### Track C — `feat/verification-hitl` · Verification Owner

**Goal: FAIL verdicts that name the right culprit, and a human can interject.**

| Order | File | What |
|---|---|---|
| C1 | `agents/rules.py` | evaluate `pass_condition.assertions` against `NodeResult.structured` in a restricted eval namespace. **No `eval()` on raw model output** — whitelist `len`, `any`, `all`, comparisons, and ancestor lookups |
| C2 | `prompts/verify.md` | given node goal + output + ancestor context, answer the semantic check; must return `suspect_node_ids` |
| C3 | `agents/verifier.py` | `Verifier(llm).check(node, result, context) -> Verdict` |
| C4 | `agents/localiser.py` | reason + store → target id (ARCHITECTURE §5b) |
| C5 | `adapters/trace/jsonl.py` | append-only JSONL, one record per line, flush per write |
| C6 | `cli/render.py` | `rich` tree; colour by `NodeStatus`; strike-through invalidated nodes |
| C7 | `engine/interrupt.py` | `ScriptedInterrupt` (list of node_ids, for tests/demo) **and** `CliInterrupt` (prompts at terminal) |
| C8 | `cli/main.py` | `run`, `plan`, `replay`, `bench` commands |

**C6 is worth more demo marks than its line count suggests.** The evaluator's
entire understanding of whether backtracking happened comes from what they see in
the terminal. Invest in it: show the tree live, re-render on each state change,
and when a subtree is invalidated make the kill **visually obvious** — struck
through, dimmed, with a red arrow pointing at the backtrack target.

**Build `ScriptedInterrupt` before `CliInterrupt`.** A scripted interrupt is
testable, reproducible, and safe to run in front of an evaluator; interactive
typing on stage is a liability. Have both, demo the scripted one.

**Gate C** — a hand-built failing `NodeResult` produces a `Verdict(FAIL)` with
populated `suspect_node_ids`, and `traces/*.jsonl` contains a record matching the
Slide-7 schema field-for-field.

### Track D — `feat/bench-harness` · Bench Owner

**Goal: the system can be driven reproducibly, and cheaply.**

| Order | File | What |
|---|---|---|
| D1 | `adapters/llm/anthropic.py`, `openai.py` | schema-constrained completion, token counting, retry on 429/5xx only |
| D2 | `adapters/llm/cached.py` | **build this early** — wraps any LLM, hashes `prompt+model+temp`, persists to `.cache/llm/`. This is demo insurance (ARCHITECTURE §9) |
| D3 | `adapters/llm/fake.py` | extend the Stage-0 stub: scripted per-prompt-name responses for the demo goal |
| D4 | `bench/goals/dev/g001..g003.json` | three dev goals against the frozen schema — g001 is the demo goal |
| D5 | `adapters/search/tavily.py` + `fake.py` | free tier; fake returns fixed hits |
| D6 | `bench/harness.py` | run N goals, collect traces, no metrics yet |

**D2 is the most important deliverable on this track.** Everything else here is
replaceable; the cache is what stops a network failure from ending the demo.

**Gate D** — one real API call through `AnthropicLLM` returns a parsed `GoalSpec`,
and running the same call twice serves the second from cache with zero network.

---

## Stage 2 — Integration (all four)

Merge in dependency order, testing after each merge:

```
feat/engine-core → main       → pytest tests/engine
feat/planning-agents → main   → pytest tests/agents ; kurogami plan --dry-run
feat/verification-hitl → main → pytest ; kurogami run --llm fake
feat/bench-harness → main     → full suite
```

Then the first true end-to-end run:

```bash
kurogami run --goal-file bench/goals/dev/g001.json --llm fake --verbose
```

**Expect this to fail the first several times.** That is normal, and it is why
integration is its own stage rather than an afterthought. The common failures, in
the order you will hit them: planner emits a schema the executor can't consume;
context assembly produces an empty dict; localiser returns a node id that isn't in
the store; invalidation cascades to the root and the run never terminates.

**Gate E** — the loop closes without exception, produces a tree, writes a trace.
Correctness of *content* does not matter yet — only that it runs to completion.

---

## Stage 3 — Make the demo real

| # | Task | Who | Why it matters |
|---|---|---|---|
| 1 | Run g001 end-to-end on real models, repeatedly, until it produces a depth-≥4 tree with a **genuine** verifier FAIL | all | this *is* the demo |
| 2 | If no genuine failure appears after several runs, enable `--inject-fault <node_id>` as the fallback and **say so at the viva** | Engine Owner | honesty beats a rigged demo |
| 3 | Tune `prompts/expand.md` until depth ≥ 4 is reliable across 3 consecutive runs | Planning Owner | Slide 3 promised depth ≥ 4 |
| 4 | Warm the cache: run the demo goal a few times so the demo is cache-served | Bench Owner | insurance |
| 5 | Record a terminal capture (`asciinema` or screen recording) of a successful run | Verification Owner | Slide 13 already asks whether recorded is acceptable |
| 6 | `kurogami replay traces/<run>.jsonl` renders the tree from disk | Verification Owner | offline fallback |

### Freeze and rehearse

- **Code freeze once Gate F passes.** After that, only README/doc edits. No
  exceptions — a "tiny fix" after freeze is how demos die.
- Tag it: `git tag -a v0.1-midsem -m "Mid-semester demo build"`.
- Full dry run of the demo script (CHECKPOINTS.md §4), timed, twice.
- Each person rehearses **their own** 90-second segment — Slide 11 promises that
  each member personally demonstrates something. The evaluator may well ask each
  of you separately.
- Write the failure-mode card: what to say if the run breaks live.

---

## Stage 4 — Semester work (post-viva)

Do not start any of this until the demo is behind you.

| Phase | Focus | Deliverable |
|---|---|---|
| S4.1 | Harden the engine; full 30-goal suite authored; **held-out split sealed** | benchmark frozen, split sealed on a separate branch |
| S4.2 | LangGraph runtime adapter; **planner model comparison** (Slide 8, open comparison #1) | comparison table, model chosen |
| S4.3 | Rule-vs-LLM **verification comparison** (#2); fault-localisation accuracy on the adversarial tier | localisation accuracy number |
| S4.4 | **Backtrack granularity comparison** (#3); flat baseline implemented; full benchmark execution | all five Slide-9 metrics computed |
| S4.5 | Blind human rating (2 external raters), variance analysis, buffer | final metric table |
| S4.6 | End-semester demo prep against the Slide-10 definition of done | live demo on an unseen goal |

Note how every experiment maps to a CLI flag that already exists by the end of
Stage 3. That is the payoff of the port design — the promised comparisons cost a
shell loop each, not a rewrite.

---

## Skills worth installing

Two different things get called "skills" here; you want both.

### Claude Code skills — for *you*, building the repo

Install via the Anthropic skills marketplace in Claude Code. In rough order of
payoff for this project:

| Skill | Why, specifically for Kurogami |
|---|---|
| **skill-creator** | You will want a project-local skill encoding the contracts + dependency rules so every coding-agent session starts compliant instead of re-deriving the architecture. Highest leverage item on this list. |
| **mcp-builder** | Sem 8 turns executors into MCP tool-callers. Also the cleanest way to expose the search adapter. Start reading it now, use it later. |
| **doc-coauthoring** | The end-semester report is a real deliverable. This is the workflow for it. |
| **pptx** | End-semester deck. You already know the pain of building one under pressure. |
| **dataviz** | Slide 9's five metrics need charts that don't look like default matplotlib. Read it before writing the first chart. |
| **artifact-design** | If you build a web trace viewer in Sem 8 rather than the CLI one. |
| **deep-research** | Literature positioning for the final report — related-work section on agent orchestration. |
| **pdf** | Reading papers programmatically, and producing the final report PDF. |

**Set up a project-local skill in Stage 0.** A `kurogami-conventions` skill holding
ARCHITECTURE §2 (the principles), §4 (the contracts) and CLAUDE.md §2 (the
dependency rules) means any agent you spawn writes compliant code without you
pasting context every time. Use `skill-creator` to build it; point it at
`docs/ARCHITECTURE.md`.

### Capabilities for *Kurogami's own* agents

Not downloads — things to build behind ports, in this order of value:

| Capability | Port | When |
|---|---|---|
| Web search | `SearchPort` (Tavily/Brave) | Stage 1 — `RESEARCH` nodes need it |
| Structured output / schema-constrained decoding | inside `LLMPort` | Stage 1 — everything depends on it |
| Prompt + response caching | `CachedLLM` | Stage 1 — demo insurance |
| Numeric sanity checking (market sizing arithmetic) | new `CalcPort` | Stage 4 — stops the sizing node hallucinating a TAM |
| Page fetch + extraction for pricing pages | extend `SearchPort` | Stage 4 |
| MCP tool-calling executors | `ToolPort` | Sem 8 |

Resist adding any of these before the demo except the first three. Each new port
is a new failure surface in a demo you cannot afford to have fail.

---

## Honest risk assessment for the pre-demo stages

| Risk | Likelihood | If it happens |
|---|---|---|
| Planner produces shallow trees (depth 2–3) | **High** | Constrain to ≤3 children and require each child's pass condition to name an ancestor. Fallback: seed 2 root nodes manually from `GoalSpec` and let the planner expand from there — still runtime-generated below depth 1, still defensible. |
| No genuine verifier FAIL in a live run | **High** | `--inject-fault`. Disclose it: *"the localisation and repair are real; the trigger is injected for reproducibility."* That is a normal thing to say about a mid-semester demo. |
| Localiser blames the wrong node | Medium | Fall back to parent-of-failing-node. Report localisation as an open problem — Slide 12 already lists it as a NEW risk, so you are consistent rather than caught out. |
| Integration takes longer than expected | Medium | Cut human interrupt to `ScriptedInterrupt` only; it is the least load-bearing of the seven demo items. |
| API rate limits / cost | Low | `CachedLLM` from Stage 1; cap `max_tokens_total`. |
| Someone edits `contracts/` mid-stream | Medium | The freeze rule (§0.2). Enforce it socially — this is a people problem, not a tooling one. |

**The single highest-probability way the demo goes badly** is sinking effort into
the benchmark harness and prompt polish and arriving at Stage 3 with no working
`invalidate_subtree`. Gate A exists to make that visible early rather than late.
