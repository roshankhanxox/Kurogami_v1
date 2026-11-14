# Kurogami Textual demo

## Start here

```bash
cd ~/Projects/Kurogami_v1
source .venv/bin/activate
kurogami
```

The home screen has a question box and **Recent runs** underneath. Type the following settings in the question box, pressing **Enter** after each line:

```text
/llm openai
/model gpt-4.1-mini
/search none
/review on
```

The live run needs internet access and the OpenAI key in the project's `.env`. With search off, label market estimates as assumptions. The live run can take several minutes, so keep a saved run ready.

## Seven-minute presentation

| Time | What to show |
| --- | --- |
| 0:00–0:30 | Show the scope slide. Say that Kurogami turns a market-entry question into a graph of model-written tasks with verification and recovery. |
| 0:30–2:00 | Submit the live question. When the plan appears, select a task and show its **Prompt** and **Checks** tabs. If the plan is still loading after this segment, switch to the saved run. |
| 2:00–3:30 | Add a constraint with **c**, or mark a pause with **p**, then press **r**. Show the results changing or switch to the backup if the run is slow or stops. |
| 3:30–5:00 | Press **a** for the output and **e** for the browser. Pick a task with more than one prerequisite and show the model call stack. |
| 5:00–6:00 | Show where the planted-fault run stopped. Explain that the verifier can identify a failure without locating its true source. State that the earlier development pilot completed two of three goals. |
| 6:00–7:00 | Take a panel question or reopen an archived example. |

## Run the live demo

1. Type **Should I pilot an appointment reminder for independent tutors? Recommend a go or no-go decision, an email-only pilot, and three success measures. Use one two-week phase. Treat unknown market figures as assumptions.** Press **Enter**. On this build, a live run of this question completed with 12 tasks at depth 5. Its recommendation used email only and listed three measures.
2. When the plan appears, select a task with the arrow keys. Use **Prompt** and **Checks**. Explain that the tool wrote the plan and decided which tasks depend on earlier work.
3. To show human input, select a task that has yet to run and press **c**. Enter a constraint appropriate to the task, such as **Include a manual opt-out in the pilot design.** Choose **Apply**. Alternatively, press **p** to mark a task where the run should pause. A new constraint changes the run, so check the result before claiming it was followed.
4. Press **r** to start. Show the task statuses as they update. If the run stops, show the reason and switch to the saved run.
5. When the run finishes, press **a** for the final task, then **f** for full-screen output. Use **Page Down** or **End** to read through the answer, and **Esc** to go back. Check whether the decision follows your constraint. Press **e** to open the graph and expand **Model call stack**.

## Open a backup without model calls

From the home screen, press **Tab** to focus **Recent runs**, choose a run, and press **Enter**. The completed invoicing run and the planted-fault stock-tracking run are saved in the local project. The planted case exhausted its recovery budget. Use it to explain what went wrong.

You can also open the completed backup directly:

```bash
kurogami tui --open traces/demo-invoicing.snapshot.json
```

Press **a** for the recommendation and **e** for the browser view. If the browser does not open, exit the screen and use:

```bash
kurogami export traces/demo-invoicing.snapshot.json --open
```

The live tutor rehearsal is also saved in **Recent runs**. Open it directly with:

```bash
kurogami tui --open traces/textual-live-simple/348d847f-8aa8-481e-a1f5-12ad4c996548.snapshot.json
```

The HTML can show a task list and the call stack offline. Its graphical layout uses browser libraries loaded over the internet.
