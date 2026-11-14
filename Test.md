# Test Kurogami

Run these from `~/Projects/Kurogami_v1` with `(.venv)` active. If `kurogami` is missing, run `python -m pip install -e .` before the demo.

## Test with a live model

Use an internet connection and set `OPENAI_API_KEY` in the project's `.env` file.
Run this first. It makes fresh model calls and accepts any market-entry question:

```sh
kurogami run --text "Should I launch an invoicing tool for freelance designers in India?" --llm openai --no-cache --search none --verbose
```

Look for the graph, node verdicts, and a final recommendation. Read the recommendation and check whether it answers your question. `--search none` disables web research, so check market claims separately. If the run stops early, read the terminal reason and the trace path.

After a run, type `kurogami open-trace` to open its task dependencies and model calls in your browser. To open a specific run, pass the execution trace path printed by the CLI. Add `--no-browser` to generate the HTML and print its location.

To test a planted failure with a real verifier, run the development pilot once. It uses fresh model calls for all three goals and their flat baselines:

```sh
python -m kurogami.bench.harness --out "traces/live-pilot-$(date +%s)"
```

Open `g003/result.json` inside the new `traces/live-pilot-…` directory. Recovery needs both a correct ancestor backtrack and a final recommendation that keeps the pilot free. The previous run failed this check.

To test human input, copy a node ID from the live graph and add `--interrupt-at NODE_ID` to the live command. If the freshly generated graph uses different IDs, the pause may not trigger. Use any constraint when prompted, press Enter on a blank line, and read the final recommendation to check whether the model followed it.

## Offline fallback for the demo

Extract the offline demo ZIP into the project folder so `endsem-demo/goal.json` exists. These commands run without API keys. They show the runner using saved responses.

1. **Complete a goal**
   ```sh
   kurogami run --goal-file endsem-demo/goal.json --llm openai --offline --cache-dir endsem-demo/invoicing/cache
   ```
   Expect 10 passing nodes, depth 4, and a printed recommendation using recorded provider responses.

2. **Recover from a verifier failure**
   ```sh
   kurogami run --text "Should I launch my invoicing tool?" --llm openai --offline --model controlled-demo --cache-dir endsem-demo/recovery/cache --verbose
   ```
   A scripted verifier failure at `a1` backtracks to `a0`. The regenerated `a1~r1` passes and `b0` runs once.

3. **Inject a constraint**
   ```sh
   printf '%s\n\n' 'Use email only.' | kurogami run --text "Should I launch my invoicing tool?" --llm openai --offline --model controlled-demo --cache-dir endsem-demo/interrupt/cache --interrupt-at a0
   ```
   This feeds the constraint and a blank line automatically. Expect `paused at a0` and `Recommendation: use email only.` To type it yourself, run the command starting at `kurogami run`. At the `>` prompt, enter exactly `Use email only.` with a capital U and a period, then press Enter again on the empty line. Different wording, capitalization, or punctuation causes an offline cache miss because this scripted demo has responses saved for that exact input.
