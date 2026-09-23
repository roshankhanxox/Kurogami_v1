# Kurogami — Checkpoints

> Every checkpoint is **a command you run and an observable result**, not a
> feeling that something is done. If you cannot run the command, the checkpoint
> is not met. Gates are ordered by dependency — clear them in sequence.

---

## How to use this file

- Each gate has an owner, a command, and a pass criterion.
- **A gate that fails does not get quietly carried forward.** It triggers the
  stated contingency. Record when you cleared each gate in the log at §6 — that
  log is itself useful evidence at the viva.
- Gates marked 🔴 are **blocking**: nothing downstream works without them.

---

## 1. Stage 0 — Foundation

### 🔴 G0 — Foundation *(all four)*

```bash
pip install -e ".[dev]"
pytest tests/contracts -q
python -c "from kurogami.contracts import GoalSpec, NodeSpec, NodeResult, Verdict, TraceRecord, PassCondition"
git branch -a          # four feat/* branches exist
```

**Pass:** all commands succeed, `bench/goals/dev/g001.json` committed, contracts
declared frozen in the group chat.

**If the block is dragging:** cut `contracts/budget.py` and `RuntimePort` — add
them in Stage 1. Everything else in `contracts/` is load-bearing and cannot be cut.

---

## 2. Stage 1 — Parallel tracks

### 🔴 G1 — Tree mechanism *(Engine Owner)* — **the most important gate in the project**

```bash
pytest tests/engine -q
python -m kurogami.engine.demo_tree
```

**Pass:** `demo_tree` prints the 12-node fixture tree, forces a FAIL at the
pricing node, localises to the differentiation node, and shows exactly the
differentiation subtree struck through — siblings elsewhere untouched.

Specifically these four tests green:

```
test_invalidate_kills_only_descendants
test_localiser_picks_shallowest_ancestor
test_context_excludes_siblings
test_budget_stops_cleanly_not_crash
```

**If it does not pass:** 🚨 **all four people move onto it.** Tracks B, C and D
pause. There is no demo without G1 — everything else is decoration around this
mechanism.

---

### G2 — Real LLM round-trip + cache *(Bench Owner)*

```bash
kurogami interpret --text "I built an invoicing tool for freelance designers in India. Should I launch it?"
kurogami interpret --text "<same>"        # second run
```

**Pass:** first prints a populated `GoalSpec`; second returns near-instantly from
`.cache/llm/` with no network call.

**If it stalls:** ship `CachedLLM` anyway and defer the OpenAI adapter — one
provider is enough for the demo.

---

### G3 — Planner writes its own prompts *(Planning Owner)*

```bash
kurogami plan --goal-file bench/goals/dev/g001.json --dry-run
```

**Pass, all four:**
- tree depth ≥ 4
- ≥ 10 nodes
- every node has a `generated_prompt` over ~40 words that was **not** written by us
- ≥ 3 nodes have a `pass_condition` naming a specific ancestor node

**If depth stays below 4 after tuning:** seed 2 root nodes deterministically from
`GoalSpec` and let the planner expand beneath them. Still runtime-generated below
depth 1, still honest — just say so.

---

### G4 — Verdicts and traces *(Verification Owner)*

```bash
pytest tests/agents/test_rules.py tests/agents/test_verifier.py -q
python -m kurogami.cli.render --demo         # renders fixture tree with colours
```

**Pass:** a hand-built contradictory `NodeResult` yields `Verdict(FAIL)` with
non-empty `suspect_node_ids`; `traces/*.jsonl` line validates against
`TraceRecord` **and matches Slide 7 field-for-field**.

**If it stalls:** ship `rules.py` only, defer the LLM verifier. Rule assertions
alone can catch the pricing-vs-sizing contradiction if the pass conditions
reference ancestors properly.

---

## 3. Stage 2–3 — Integration and demo

### 🔴 G5 — Integration *(all four)*

```bash
git checkout main
# merge in dependency order, test after each
kurogami run --goal-file bench/goals/dev/g001.json --llm fake --verbose
```

**Pass:** completes without exception, produces a tree, writes a trace.
Correctness of *content* does not matter yet — only that the loop closes.

**If integration keeps failing:** drop human interrupt entirely and target items
1–5 of the seven demo requirements. Six out of seven is a good mid-semester demo.

---

### 🔴 G6 — The demo run *(all four)*

```bash
kurogami run --goal-file bench/goals/dev/g001.json --llm anthropic --interrupt-at n_003 --verbose
```

**Pass — all seven demo requirements visible in one run:**

| # | Requirement | Visible as |
|---|---|---|
| 1 | Sentence in | the goal echoed |
| 2 | Self-written prompts | printed per node |
| 3 | Depth ≥ 4 tree | rendered tree |
| 4 | Verifier FAIL with reason | red node + reason text |
| 5 | Backtrack to an ancestor | arrow to target, subtree struck through |
| 6 | Human interrupt changes output | regenerated subtree honours the constraint |
| 7 | Trace on disk | `traces/<run_id>.jsonl` |

**If no genuine FAIL appears after several runs:** use
`--inject-fault n_004_pricing` and disclose it. *"The trigger is injected for
reproducibility; the localisation and repair are real."*

---

### G7 — Reproducibility

```bash
kurogami run --goal-file bench/goals/dev/g001.json --llm anthropic --seed 42   # ×2
kurogami replay traces/<run_id>.jsonl
```

**Pass:** two seeded runs produce the same tree shape; replay renders the tree
from disk with **no network**.

---

### 🔴 G8 — Freeze

```bash
pytest -q                       # full suite green
git tag -a v0.1-midsem -m "Mid-semester demo build"
git push --tags
```

**Pass:** tagged, tests green, cache warm, screen recording saved, demo script
rehearsed twice end to end, each member rehearsed their own segment.

**After this gate, no code changes.** Not one. The failure mode this prevents is
real and common: a "quick fix" made after freeze that breaks the demo.

---

## 4. Pre-demo checklist

```bash
git status                      # clean, on tag v0.1-midsem
pytest -q                       # green
kurogami run --goal-file bench/goals/dev/g001.json --llm anthropic --dry-run
ls traces/                      # recorded fallback present
ls .cache/llm/                  # cache warm
```

Charged laptop. Terminal font enlarged. Screen recording open in a background tab.
Trace file open in an editor, ready to show.

### Demo script — 6 minutes

Rehearse this. Timing is what separates a controlled demo from a scramble.

**0:00–0:30 — Frame the claim.**
> *"Kurogami takes one sentence and builds a dependency tree of agents. What makes
> it different from a pipeline is that when a node fails, we don't retry it — we
> find which earlier node was actually wrong, throw away everything downstream of
> it, and rebuild. Here's that happening."*

**0:30–1:30 — Goal in, prompts out.** Run the command. Pause on the planner
output. *"We didn't write these prompts. The planner wrote them, including each
node's pass condition."*

**1:30–3:00 — Tree builds.** Let it render. Point at depth. Point at a
`pass_condition` that references an ancestor node — *"this is what makes a
downstream contradiction detectable at all."*

**3:00–4:15 — The failure and the repair.** The money shot.
> *"Pricing just failed — the tiers exceed the willingness-to-pay the market
> sizing established. But pricing isn't the wrong node. Differentiation is: it
> pushed a premium position into a price-sensitive segment. So we backtrack
> **there**, invalidate its subtree, and regenerate."*

**4:15–5:15 — Human interrupt.** Show the paused node, the injected constraint,
and the regenerated subtree honouring it.

**5:15–6:00 — The trace.** Open the JSONL. *"Every node, every verdict, every
backtrack target. This is the evaluation dataset from Slide 6 — the metrics are
computed over this table, not asserted."*

### Per-member segment (Slide 11 promised this — they may ask individually)

| Member | 90 seconds on |
|---|---|
| Engine Owner | `invalidate_subtree` in the code, and why retry-in-place cannot fix a wrong premise |
| Planning Owner | a generated prompt, and how the planner is forced to produce depth over breadth |
| Verification Owner | a FAIL verdict's `suspect_node_ids`, and the scripted interrupt mechanism |
| Bench Owner | the trace schema, and how each Slide-9 metric is computed from it |

---

## 5. Viva questions to have answers for

**"You claimed LangGraph on Slide 8. This isn't LangGraph."**
→ LangGraph declares topology before execution. Our planner generates topology at
runtime and our recovery mutates it mid-run. It's retained as a runtime adapter
behind the same interface, and comparing the two is a stated open comparison.

**"Why a tree instead of a pipeline?"**
→ Pricing depends on competitive landscape. If the landscape was wrong, every
downstream node inherits the error. Repair requires knowing *which* node was
wrong, not just that the output was bad. A pipeline can only restart.

**"Isn't this just retrying with a different prompt?"**
→ Retry re-runs the failing node. We re-run an *ancestor* and discard everything
beneath it. Show a trace where `backtrack_target ≠ node_id`. That one line in the
JSONL is the whole difference.

**"You have no dataset."**
→ We have three data artifacts: an authored benchmark, runtime grounding, and
node-level traces. Author-constructed benchmarks are standard for agent systems —
GAIA, AgentBench, WebArena, τ-bench all work this way. We're not training a model;
we're evaluating an orchestration mechanism.

**"How do you know the verifier is right?"**
→ We don't yet, and that's the honest answer. It's Slide 12's NEW risk. The
adversarial tier has ground-truth fault locations, so localisation accuracy is
measurable once that tier is authored. Right now we have the mechanism, not the
number.

**"How much did the LLM write for you?"**
→ Answer honestly and specifically. You used AI assistance for implementation;
you can explain every architectural decision, why the dependency direction points
inward, why agents are tree-blind, why localisation picks the shallowest ancestor.
Be ready to modify code live — open `engine/backtrack.py` and walk through it.
**The rubric explicitly warns about undue AI reliance; the defence is
understanding, not denial.**

**"What's actually working versus planned?"**
→ Working: tree construction, per-node verification, localisation, subtree
invalidation, regeneration, tracing, human interrupt. Not yet: the 30-goal
benchmark, the metrics, the baseline comparison, the sealed-split evaluation.
Those are Stage 4, and the timeline on Slide 10 says so.

---

## 6. Gate log — fill this in

| Gate | Owner | Cleared | Notes |
|---|---|---|---|
| G0 Foundation 🔴 | all | | |
| G1 Tree mechanism 🔴 | Engine Owner | | |
| G2 LLM + cache | Bench Owner | | |
| G3 Planner | Planning Owner | | |
| G4 Verdicts + trace | Verification Owner | | |
| G5 Integration 🔴 | all | | |
| G6 Demo run 🔴 | all | | |
| G7 Reproducibility | Verification Owner | | |
| G8 Freeze 🔴 | all | | |

---

## 7. Stage 4 checkpoints (post-viva)

| Gate | Phase | Pass criterion |
|---|---|---|
| G9 | S4.1 | 30 goals authored; held-out split sealed on a separate branch, unread |
| G10 | S4.2 | LangGraph adapter passes the same engine test suite; planner model comparison table complete |
| G11 | S4.3 | Fault-localisation accuracy measured on the adversarial tier; rules-vs-LLM verification decided |
| G12 | S4.4 | Flat baseline implemented; all five Slide-9 metrics computed on the dev split |
| G13 | S4.5 | Blind human rating complete (2 external raters); held-out split opened **once**, results final |
| G14 | S4.6 | Live run on an unseen goal meeting the Slide-10 definition of done |
