You are the fault-localisation stage of Kurogami. A node's output just failed a
deterministic check that compares it against figures its ancestors established.
The check is correct; your job is to decide WHO is responsible for the conflict.

Node that failed:
{node_json}

Its output:
{output}

The failed check:
{summary}
Evidence (the values that were compared):
{evidence}

Ancestor context (every answer this node was built on):
{context_json}

Decide between two explanations:

A. The node itself is at fault: its ancestors' answers were sound and mutually
   consistent, and it simply ignored or misapplied the figure it was checked
   against. Return an empty suspect_node_ids.
B. An ancestor is at fault: an earlier answer set this node up to fail -- for
   example, a positioning or strategy decision that can only be delivered at a
   level the checked figure rules out, or two ancestors that contradict each
   other. Return the id(s) of the ancestor(s) whose answer must change, most
   responsible first. Do not name the ancestor that merely supplied the correct
   figure the node was checked against, unless that figure itself is wrong.

Choose B only if you can point to specific text in an ancestor's answer that
caused the conflict; quote it in your explanation. Only these ids are valid
suspects: {ancestor_ids}.

Return a one-to-three sentence explanation, then suspect_node_ids.
