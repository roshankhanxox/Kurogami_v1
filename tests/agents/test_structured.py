"""Shared structured-payload helpers: one reading of assertions for every agent."""

from kurogami.agents._structured import (
    ancestor_payloads,
    ancestor_references,
    extract_structured,
    resolved_ancestor_values,
)


def test_extracts_the_fenced_json_block_from_prose():
    assert extract_structured('prose\n```json\n{"a": 1}\n```') == {"a": 1}


def test_ancestor_payloads_parse_full_answers_and_serialised_payloads_alike():
    context = {
        "parent": 'Full prose answer.\n```json\n{"ceiling": 500}\n```',
        "grandparent": '{"size": 1000}',
        "_backtrack_reason": "not an ancestor",
    }
    assert ancestor_payloads(context) == {"parent": {"ceiling": 500}, "grandparent": {"size": 1000}}


def test_ancestor_references_cover_subscript_and_get_forms():
    assert ancestor_references("structured['p'] <= ancestors['wtp']['ceiling']") == [
        ("wtp", "ceiling")
    ]
    assert ancestor_references("structured['p'] <= ancestors.get('wtp', {}).get('ceiling', 0)") == [
        ("wtp", "ceiling")
    ]
    assert ancestor_references("len(ancestors['wtp']) > 0") == [("wtp", None)]
    assert ancestor_references("structured['p'] > 0") == []


def test_resolved_values_show_the_figure_or_say_why_it_is_missing():
    context = {"wtp": '{"ceiling": 500}'}
    lines = resolved_ancestor_values(
        ["structured['p'] <= ancestors['wtp']['ceiling']", "structured['q'] <= ancestors['wtp']['floor']",
         "structured['r'] <= ancestors['gone']['x']"],
        context,
    )
    assert lines[0] == "ancestors['wtp']['ceiling'] = 500"
    assert "did not report" in lines[1]
    assert "not in your context" in lines[2]


def test_example_placeholders_match_what_the_checks_do_with_each_key():
    """Live incident: a "..." placeholder for a key compared with >= 0 made the model
    write a sentence there, on every retry."""
    from kurogami.agents._structured import example_value

    assertions = [
        "structured['late_payment_frequency'] >= 0",
        "len(structured['common_challenges']) >= 2",
        "isinstance(structured.get('plan'), dict)",
        "structured['tier'] in ['low', 'high']",
        "isinstance(structured['price'], (int, float))",
    ]
    assert example_value("late_payment_frequency", assertions) == 0
    assert example_value("common_challenges", assertions) == ["..."]
    assert example_value("plan", assertions) == {"...": "..."}
    assert example_value("tier", assertions) == "low"
    assert example_value("price", assertions) == 0
    assert example_value("notes", assertions) == "..."
