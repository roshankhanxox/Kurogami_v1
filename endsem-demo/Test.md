# Test Kurogami

From the repository root, install with `python -m pip install -e .` and extract the offline demo ZIP so its `endsem-demo/` folder is beside this file. All three commands run without API keys.

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
   kurogami run --text "Should I launch my invoicing tool?" --llm openai --offline --model controlled-demo --cache-dir endsem-demo/interrupt/cache --interrupt-at a0
   ```
   At `a0`, type `Use email only.` and then a blank line. The scripted model response reflects the constraint in the recommendation.
