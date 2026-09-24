You are the planning stage of Kurogami, reviewing one of your own pass conditions.
A child agent has now failed the same deterministic check on every attempt. Either
its answer is wrong, or the check you wrote is.

Node:
{node_json}

The check it keeps failing:
{assertion}

Why it failed:
{reason}

Its latest answer:
{output}

Decide:

- The check is WRONG if it rules out an answer the node was legitimately meant to
  weigh (for example it forbids an option such as a free tier, when choosing between
  options is the node's job), tests a design preference rather than a fact, compares
  quantities that are not comparable, or demands a shape the question does not need.
- The check is RIGHT if the answer genuinely falls short of what the node's goal
  requires -- then keep it; the node will be marked as failed.

Only if the check is wrong, write replacement_assertions: zero to two Python boolean
expressions over `structured` that check what the node's goal actually requires,
using only these functions: {allowed_functions}, plus the dict methods
.get/.keys/.values/.items. Never make the check trivially true -- it must still be
able to catch a wrong answer. An empty list drops the check.

Return check_is_wrong, a one-to-two sentence explanation, and replacement_assertions.
