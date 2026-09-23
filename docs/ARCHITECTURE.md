# Kurogami — Architecture

> **Status:** authoritative design document. Code that contradicts this file is a
> bug in the code or an amendment to this file — never a silent divergence.
> **Audience:** the four team members, and any coding agent working in this repo.

---

## 0. The one-sentence system

Kurogami takes a single natural-language goal, has an LLM **write the prompts for
its own child agents**, arranges those children as a **dependency tree**, executes
the tree depth-first, **verifies every node against a goal declared for that node**,
and on failure **localises the responsible ancestor, invalidates its subtree, and
regenerates** — with a human able to pause at any node and inject a constraint that
the regenerated subtree must honour.

Everything in this document exists to make that sentence true, testable, and
demonstrable.

---

## 1. Architectural decision that differs from the deck

**The deck names LangGraph as the orchestration runtime. We are not building the
core on LangGraph, and this is deliberate.**

LangGraph's model is a graph whose *topology is declared before execution*. Our
central claim is a tree whose **topology is generated at runtime by the planner**
and **structurally mutated mid-run** when a subtree is invalidated. Expressing
"delete these seven nodes and re-expand from node 3 with a new constraint" inside
a pre-declared graph runtime means fighting the framework at exactly the point
where our novelty lives.

**What we do instead:** the tree engine is ours — a few hundred lines of plain
Python over an explicit node store. LangGraph is retained as an **adapter behind
the `RuntimePort` interface**, implementing the same contract. This means:

- The demo runs on the in-process runtime, which we fully control.
- The deck's LangGraph claim stays honest: it is a supported runtime, and the
  comparison between the two *becomes* a legitimate open comparison (identical
  trees, identical benchmark, different runtime) — which is exactly the kind of
  evidence Slide 8 promises.

**Viva answer if asked:** *"LangGraph assumes a static topology. Our planner
generates topology at runtime and our recovery mechanism mutates it mid-execution.
We kept LangGraph as a runtime adapter behind the same interface so we can measure
the trade-off, but the tree engine had to be ours for subtree invalidation to be
expressible at all."*

That is a stronger answer than "we used the framework."

---

## 2. Design principles (these are enforced, not aspirational)

### P1 — Dependency points inward, always

```
contracts/  ←  agents/  ←  engine/  ←  cli/
     ↑            ↑
     └── adapters/┘
```

`contracts/` imports **nothing** from the rest of the package. Every other module
may import `contracts/`. No module imports `cli/`. Adapters are imported only by
composition roots (`cli/main.py`, `bench/harness.py`), never by `agents/` or
`engine/`.

If you find yourself needing an import that points outward, you have found a
missing contract. Add the contract; do not add the import.

### P2 — Everything crossing a module boundary is a typed contract object

No dicts, no tuples, no "we'll just pass the raw string" at a seam. Pydantic
models in `contracts/` are the only currency. This is what makes every module
independently testable and independently replaceable — which is the
reusability claim we are making to the examiner.

### P3 — Agents do not know the tree exists

An agent receives a `NodeSpec` and returns a `NodeResult`. It cannot see its
parent, its siblings, or the tree. Everything an agent needs about its ancestry
arrives in `NodeSpec.context` — assembled by the engine. This is the single most
important boundary in the system: it is why a node can be re-executed in a
regenerated subtree with different ancestry and no code changes.

### P4 — Every external dependency sits behind a port

LLM calls, web search, trace persistence, clock, and randomness all cross a
`Protocol` defined in `contracts/ports.py`. Every port has a **fake
implementation** in `adapters/` that is deterministic and offline. Consequence:
the whole system runs, and the entire test suite passes, with no API key and no
network. This is also our demo insurance policy (§9).

### P5 — Prompts are versioned data files, not string literals

All prompts live in `src/kurogami/prompts/*.md` and are loaded by name. Changing
a prompt is a reviewable diff. A prompt's filename and content hash are recorded
in every trace record, so a result can always be traced back to the prompt that
produced it.

### P6 — Nothing is silently discarded

Every node execution, every verdict, every backtrack, every human interrupt
emits a trace record. The trace is the product, not a debug artifact — Slide 6
committed to it being the evaluation dataset.

---

## 3. Repository layout

```
kurogami/
├── CLAUDE.md                      # working agreement for humans + coding agents
├── README.md                      # 10-line quickstart only
├── pyproject.toml
├── .env.example
├── docs/
│   ├── ARCHITECTURE.md            # this file
│   ├── IMPLEMENTATION_PLAN.md
│   └── CHECKPOINTS.md
├── src/kurogami/
│   ├── contracts/                 # LAYER 0 — pure data, zero internal imports
│   │   ├── goal.py                # GoalSpec
│   │   ├── node.py                # NodeSpec, NodeResult, NodeStatus, NodeKind
│   │   ├── verdict.py             # Verdict, FailureReason, PassCondition
│   │   ├── tree.py                # TreeSnapshot, BacktrackEvent
│   │   ├── trace.py               # TraceRecord  (matches deck Slide 7 exactly)
│   │   ├── budget.py              # BudgetState, BudgetLimits
│   │   └── ports.py               # LLMPort, SearchPort, TraceSink, RuntimePort,
│   │                              # PlannerPort, ExecutorPort, VerifierPort,
│   │                              # InterruptPort, ClockPort
│   ├── adapters/                  # LAYER 1 — outward-facing, swappable
│   │   ├── llm/{anthropic,openai,fake,cached}.py
│   │   ├── search/{tavily,fake}.py
│   │   ├── trace/{jsonl,memory}.py
│   │   └── runtime/{inprocess,langgraph}.py
│   ├── prompts/                   # LAYER 1 — versioned prompt data
│   │   ├── interpret.md
│   │   ├── plan.md
│   │   ├── expand.md
│   │   ├── execute.md
│   │   ├── verify.md
│   │   └── localise.md
│   ├── agents/                    # LAYER 2 — stateless, tree-blind
│   │   ├── interpreter.py         # L1: text → GoalSpec
│   │   ├── planner.py             # L2: GoalSpec → child NodeSpecs (self-prompting)
│   │   ├── executor.py            # L3: NodeSpec → NodeResult
│   │   ├── verifier.py            # NodeResult → Verdict (LLM)
│   │   ├── rules.py               # NodeResult → Verdict (deterministic assertions)
│   │   └── localiser.py           # FailureReason → backtrack target node_id
│   ├── engine/                    # LAYER 3 — owns the tree, owns control flow
│   │   ├── store.py               # TreeStore: the node graph + invalidation
│   │   ├── context.py             # ancestry → NodeSpec.context assembly
│   │   ├── scheduler.py           # execution order (DFS, dependency-respecting)
│   │   ├── backtrack.py           # localisation + invalidation policy
│   │   ├── budget.py              # depth/expansion/token caps, loop detection
│   │   ├── interrupt.py           # human-in-the-loop pause points
│   │   └── runner.py              # THE LOOP — composes all of the above
│   ├── cli/                       # LAYER 4 — composition root
│   │   ├── main.py                # typer app
│   │   ├── render.py              # rich tree rendering, live node states
│   │   └── replay.py              # re-render a past run from its trace
│   └── bench/
│       ├── goals/dev/*.json       # development goals
│       ├── goals/heldout/         # SEALED — see CLAUDE.md §7
│       ├── harness.py
│       ├── baseline.py            # flat single-pass comparator
│       └── metrics.py             # the five Slide-9 metrics
└── tests/
    ├── conftest.py                # FakeLLM fixtures — no network in tests, ever
    ├── contracts/
    ├── engine/                    # the highest-value tests in the repo
    └── agents/
```

---

## 4. The contracts (implement these first, before anything else)

These are the spine. The opening work block is spent agreeing on exactly this file
set and nothing else, because all four workstreams branch from here.

### 4.1 `contracts/goal.py`

```python
class GoalSpec(BaseModel):
    """Level-1 output. The structured form of the user's sentence."""
    raw_text: str
    product_description: str
    target_market: str
    decision_type: Literal["market_entry", "positioning", "pricing", "launch"]
    success_definition: str            # what a good answer looks like
    known_constraints: list[str] = []  # user-supplied or interrupt-injected
    ambiguities: list[str] = []        # flagged, not resolved
```

### 4.2 `contracts/node.py`

```python
class NodeKind(StrEnum):
    RESEARCH = "research"      # needs SearchPort
    ANALYSIS = "analysis"      # pure reasoning over ancestor outputs
    SYNTHESIS = "synthesis"    # combines multiple parents
    DECISION  = "decision"     # produces a commitment downstream depends on

class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INVALIDATED = "invalidated"   # killed by an ancestor's backtrack
    SKIPPED = "skipped"           # budget exhausted

class NodeSpec(BaseModel):
    """Written BY THE PLANNER, not by us. This is the self-prompting claim."""
    node_id: str
    parent_ids: list[str]              # list, not scalar — DAG-shaped tree
    depth: int
    kind: NodeKind
    title: str
    node_goal: str                     # what THIS node must achieve
    generated_prompt: str              # ← the planner wrote this
    pass_condition: PassCondition      # machine-checkable, see verdict.py
    context: dict[str, str] = {}       # ancestor outputs, assembled by engine
    injected_constraints: list[str] = []   # from human interrupt

class NodeResult(BaseModel):
    node_id: str
    output: str
    structured: dict[str, Any] = {}    # parsed claims the verifier can assert on
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_id: str
    prompt_version: str                # filename + content hash
```

### 4.3 `contracts/verdict.py`

```python
class PassCondition(BaseModel):
    """Two-layer: cheap deterministic assertions + one semantic question."""
    assertions: list[str]     # e.g. "len(structured['competitors']) >= 5"
    semantic_check: str       # e.g. "Do the pricing tiers respect the WTP
                              #       ceiling established by the market sizing?"

class FailureReason(BaseModel):
    summary: str
    violated: Literal["assertion", "semantic", "schema"]
    evidence: str                       # quote the contradicting text
    suspect_node_ids: list[str] = []    # verifier's hypothesis — may be ancestors

class Verdict(BaseModel):
    node_id: str
    verdict: Literal["PASS", "FAIL"]
    reason: FailureReason | None = None
    checked_by: Literal["rules", "llm", "both"]
```

### 4.4 `contracts/trace.py`

**This must match Slide 7 field-for-field.** The examiner can read both.

```python
class TraceRecord(BaseModel):
    run_id: UUID
    seq: int                           # monotonic within run
    node_id: str
    parent_node_id: str | None
    depth: int
    generated_prompt: str
    node_goal: str
    output: str
    verifier_verdict: Literal["PASS", "FAIL"]
    failure_reason: str | None
    backtrack_target: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    human_interrupt: bool
```

### 4.5 `contracts/ports.py` — the reusability claim, made concrete

```python
class LLMPort(Protocol):
    def complete(self, *, prompt: str, system: str | None = None,
                 schema: type[BaseModel] | None = None,
                 temperature: float = 0.0) -> LLMResponse: ...

class SearchPort(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[SearchHit]: ...

class TraceSink(Protocol):
    def emit(self, record: TraceRecord) -> None: ...
    def close(self) -> None: ...

class InterruptPort(Protocol):
    def should_pause(self, node: NodeSpec) -> bool: ...
    def collect(self, node: NodeSpec) -> list[str]: ...   # returns constraints

class RuntimePort(Protocol):
    def run(self, plan: TreeSnapshot, *, hooks: RunnerHooks) -> TreeSnapshot: ...
```

Every one of these has a `Fake*` implementation. That is not a testing
convenience — it is the demonstration that the modules are genuinely decoupled.

---

## 5. Control flow — the runner loop

This is the whole system. Implement it exactly.

```
1.  interpret(raw_goal)                     → GoalSpec
2.  plan(GoalSpec)                          → root NodeSpecs (planner writes prompts)
3.  store.seed(root_nodes)
4.  LOOP while store.has_pending() and budget.ok():
4a.     node ← scheduler.next(store)         # DFS, parents-before-children
4b.     if interrupt.should_pause(node):
            constraints ← interrupt.collect(node)
            node.injected_constraints += constraints
            store.mark_dirty_descendants(node)
4c.     node.context ← context.assemble(store, node)   # ancestor outputs only
4d.     result ← executor.run(node)          # ← the ONLY place an LLM does work
4e.     verdict ← rules.check(node, result)  # cheap, deterministic, first
            if verdict.PASS: verdict ← verifier.check(node, result)  # semantic
4f.     emit trace record
4g.     if verdict.PASS:
            store.mark_passed(node, result)
            children ← planner.expand(node, result, goal)   # planner writes prompts
            store.attach(node, children)
        else:
            target ← localiser.locate(verdict.reason, store)  # ancestor or self
            store.invalidate_subtree(target)                  # ← THE CLAIM
            store.requeue(target, with_failure_context=verdict.reason)
            budget.record_backtrack()
5.  assemble final artifact from passed leaf + decision nodes
6.  trace.close()
```

### The three things that make this non-trivial (and therefore worth marks)

**(a) Context assembly (`4c`).** A node must see ancestor outputs and nothing else.
Give it siblings and you get contamination; give it the whole tree and you blow
the context window. `context.assemble` walks `parent_ids` transitively, takes each
ancestor's `structured` payload preferentially over raw `output`, and truncates by
depth distance.

**(b) Fault localisation (`4g`).** The failing node is usually **not** the wrong
node. Pricing fails because differentiation was wrong. `localiser.locate` takes the
verifier's `suspect_node_ids`, intersects with the actual ancestor set of the
failing node, and picks the **shallowest** suspect — because repairing the
shallowest wrong premise repairs everything beneath it. If there are no ancestor
suspects, the node itself is the target.

**(c) Invalidation (`store.invalidate_subtree`).** Mark the target `PENDING` and
every transitive descendant `INVALIDATED`. Do **not** delete them — keep them in
the store flagged, because the trace must show what was thrown away. Cost
accounting deduplicates these (Slide 7's stated preprocessing step).

---

## 6. Module contracts — what each owner must deliver

| Module | Owner | Public surface | Depends on | Testable without network? |
|---|---|---|---|---|
| `contracts/*` | **Engine Owner** (written first, then frozen) | the models above | nothing | n/a |
| `engine/store.py` | Engine Owner | `TreeStore` | contracts | ✅ pure |
| `engine/backtrack.py` | Engine Owner | `locate_and_invalidate()` | contracts, store | ✅ pure |
| `engine/runner.py` | Engine Owner | `Runner.run(goal) -> RunReport` | all ports | ✅ FakeLLM |
| `agents/interpreter.py` | **Planning Owner** | `Interpreter(llm).run(str) -> GoalSpec` | LLMPort | ✅ FakeLLM |
| `agents/planner.py` | Planning Owner | `Planner(llm).plan()`, `.expand()` | LLMPort | ✅ FakeLLM |
| `prompts/*` | Planning Owner | data files | — | ✅ |
| `agents/verifier.py` | **Verification Owner** | `Verifier(llm).check() -> Verdict` | LLMPort | ✅ FakeLLM |
| `agents/rules.py` | Verification Owner | `RuleChecker.check() -> Verdict` | contracts | ✅ pure |
| `engine/interrupt.py` | Verification Owner | `CliInterrupt`, `ScriptedInterrupt` | contracts | ✅ scripted |
| `cli/*`, `adapters/trace/*` | Verification Owner | `kurogami run/replay/bench` | everything | ✅ |
| `bench/*` | **Bench Owner** | `harness.run_suite()`, `metrics.compute()` | contracts, runner | ✅ |
| `adapters/llm/*` | Bench Owner | `AnthropicLLM`, `OpenAILLM`, `FakeLLM`, `CachedLLM` | LLMPort | ✅ |

**The seam discipline:** once the contracts are frozen, no two people edit the same
file. Every workstream talks to the others only through `contracts/`. If you need a
contract changed, it is a PR against `contracts/` reviewed by Engine Owner — not an edit
in passing.

---

## 7. Why this is "heavily modular" — the answer to give at the viva

Do not say "we used clean architecture." Say what is *swappable*, and show it:

| Swap | How | Proof it works |
|---|---|---|
| Change LLM provider | `--llm anthropic\|openai\|fake` | same benchmark, two providers |
| Run with zero network | `--llm fake --search fake` | whole test suite does this |
| Change runtime | `--runtime inprocess\|langgraph` | open comparison #1, Slide 8 |
| Change verification strategy | `--verify rules\|llm\|both` | open comparison #2, Slide 8 |
| Change backtrack granularity | `--backtrack subtree\|node` | open comparison #3, Slide 8 |
| Add a new domain vertical | new prompt pack in `prompts/`, zero code | the Sem-8 expansion claim |

That last row is the important one. **The system contains no market-entry logic in
code.** Market-entry knowledge lives entirely in `prompts/plan.md` and
`prompts/expand.md`. Swapping the prompt pack retargets the entire system at a new
vertical without touching the engine — which is precisely the "narrow now, broaden
in Sem 8" story the deck promises, made structural rather than aspirational.

Every one of the three "open comparisons" on Slide 8 is a CLI flag. That is not a
coincidence — the architecture was designed so the promised experiments are cheap
to run. Say that out loud.

---

## 8. Budget and safety rails (Slide 12's risk register, in code)

`engine/budget.py` enforces, and every limit is a constructor argument:

| Limit | Default | Guards against |
|---|---|---|
| `max_depth` | 6 | runaway expansion |
| `max_nodes` | 25 | planner producing a bush |
| `max_backtracks` | 3 | infinite repair loops |
| `max_tokens_total` | 150_000 | cost blowout |
| `max_node_retries` | 2 | single node thrashing |
| loop detector | signature = hash(node_goal + parent_ids) | re-expanding the same node identically |

On breach: stop cleanly, mark remaining nodes `SKIPPED`, emit a terminal trace
record with the reason, exit non-zero. **A budget breach is a legitimate,
reportable outcome — never a crash.** The demo must survive one.

---

## 9. Demo insurance (read this twice)

The single largest risk during the demo is a live API failure in front of the
evaluator.

1. **`CachedLLM` adapter.** Wraps any real LLM; keyed by
   `hash(prompt + model + temperature)`, persists to `.cache/llm/`. Run the demo
   goal a couple of times during rehearsal; the demo run is then served from cache
   — fast, free, deterministic, and *identical to a live run in every code path*.
   If asked, say so plainly: it is a cache, not a mock, and `--no-cache` proves it
   live.
2. **`FakeLLM` fallback.** Scripted responses for the exact demo goal. Last
   resort, and disclose it if used.
3. **Recorded trace + `kurogami replay`.** If the network dies entirely, replay a
   rehearsal run from its JSONL and walk the tree. Slide 13 already asks the
   evaluator whether a recorded walkthrough is acceptable — you have cover.
4. **Never run `pip install` or edit code during the demo.** Everything frozen
   before rehearsal begins.

---

## 10. What is explicitly NOT in the architecture

Deliberate, and defensible if asked:

- No database. JSONL traces on disk. A DB buys nothing at this record volume.
- No async/parallel node execution. Slide 3 scoped it out. Sequential execution
  makes the trace linear and legible, which matters more for evaluation.
- No web GUI. CLI + rich tree rendering. Scoped out on Slide 3.
- No persistent memory across runs. Each run is independent. Scoped out.
- No fine-tuning. Off-the-shelf models via API. Scoped out.
- No retry-with-same-prompt. Retrying a node unchanged cannot fix a wrong
  premise — that is the entire argument for backtracking, and building retry
  would undercut our own thesis.
