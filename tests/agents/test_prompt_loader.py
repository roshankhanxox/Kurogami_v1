"""The internal prompt loader: reads prompts/<name>.md, hashes it for prompt_version."""

import pytest

from kurogami.agents._prompt_loader import UnknownPromptError, load_prompt, prompt_version


def test_load_prompt_reads_file_content():
    content = load_prompt("interpret")
    assert "GoalSpec" in content


def test_load_prompt_raises_on_unknown_name():
    with pytest.raises(UnknownPromptError):
        load_prompt("does_not_exist")


def test_prompt_version_is_filename_plus_stable_hash():
    v1 = prompt_version("plan")
    v2 = prompt_version("plan")
    assert v1 == v2
    assert v1.startswith("plan.md@")


def test_prompt_version_differs_for_different_prompts():
    assert prompt_version("plan") != prompt_version("expand")
