"""Tests for Gemma prompt routing and structure.

Pin down the cross-language prompt builder so accidental regressions
(missing language, wrong few-shot count, missing empty-list anchor) get
caught in CI rather than at 50k chunks of compute.
"""
from __future__ import annotations

import pytest
from gheim_training.data.labelling.gemma import prompts
from gheim_training.data.label_space import CATEGORIES


@pytest.mark.parametrize("lang", list(prompts.SUPPORTED_PROMPT_LANGS))
def test_every_supported_lang_has_rules_and_examples(lang: prompts.PromptLang) -> None:
    """Every prompt language must have a rules block + at least 3 examples."""
    assert lang in prompts._RULES_BY_LANG
    assert lang in prompts._EXAMPLES_BY_LANG
    rules = prompts._RULES_BY_LANG[lang]
    assert len(rules) > 500, f"{lang} rules block looks too short ({len(rules)} chars)"
    examples = prompts._EXAMPLES_BY_LANG[lang]
    assert len(examples) >= 3, f"{lang} has only {len(examples)} few-shot examples"


@pytest.mark.parametrize("lang", list(prompts.SUPPORTED_PROMPT_LANGS))
def test_every_lang_includes_explicit_empty_anchor(lang: prompts.PromptLang) -> None:
    """At least one few-shot example MUST output ``{"spans":[]}`` so the
    model has a positive demo of the empty-list answer. Without this,
    Gemma either over-tags or never emits empty when prompted with
    structured output (PoC showed it defaulted to empty for the WRONG
    reason — pattern-matching English negative examples)."""
    examples = prompts._EXAMPLES_BY_LANG[lang]
    assert any(
        ex["assistant"].replace(" ", "") == '{"spans":[]}'
        for ex in examples
    ), f"{lang} has no empty-list few-shot example"


@pytest.mark.parametrize("lang", list(prompts.SUPPORTED_PROMPT_LANGS))
def test_every_lang_mentions_all_8_category_keywords(lang: prompts.PromptLang) -> None:
    """The rules block for every language must reference all 8 gheim
    categories by name (so Gemma sees the exact label tokens it must
    emit). Catches typos / missing categories per language."""
    rules = prompts._RULES_BY_LANG[lang]
    missing = [c for c in CATEGORIES if c not in rules]
    assert not missing, f"{lang} rules block missing category labels: {missing}"


def test_resolve_prompt_lang_routes_known_langs() -> None:
    assert prompts._resolve_prompt_lang("de_ch") == "de_ch"
    assert prompts._resolve_prompt_lang("fr_ch") == "fr_ch"
    assert prompts._resolve_prompt_lang("it_ch") == "it_ch"
    assert prompts._resolve_prompt_lang("rm") == "rm"
    assert prompts._resolve_prompt_lang("en") == "en"


def test_resolve_prompt_lang_gsw_falls_back_to_de() -> None:
    """Written GSW is too close to standard DE for any LID; route to DE prompt."""
    assert prompts._resolve_prompt_lang("gsw") == "de_ch"


def test_build_messages_uses_target_language_rules() -> None:
    """Building a message list with language="fr_ch" should produce a
    system prompt in French, with French few-shot examples."""
    msgs = prompts.build_messages("Le tribunal a statué.", language="fr_ch")
    assert msgs[0]["role"] == "system"
    # French rule blocks contain "extracteur" — won't be in DE/IT/EN/RM blocks
    assert "extracteur" in msgs[0]["content"]
    # First user message after system should be the first FR few-shot example
    first_demo = msgs[1]["content"]
    assert "Pierre Dubois" in first_demo


def test_build_messages_uses_target_language_examples_for_rm() -> None:
    msgs = prompts.build_messages("Bun di!", language="rm")
    # Rules in RM
    assert "rumantsch" in msgs[0]["content"].lower()
    # First demo also in RM
    assert "Andri Caduff" in msgs[1]["content"]


def test_build_messages_ends_with_user_chunk() -> None:
    """Last message must be the actual chunk, formatted as `Input: <<< … >>>`."""
    msgs = prompts.build_messages("Anna Müller wohnt in Bern.", language="de_ch")
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"].startswith("Input:\n<<<\n")
    assert "Anna Müller wohnt in Bern." in last["content"]


def test_demonstrations_are_full_user_assistant_turns() -> None:
    """Every example contributes (user, assistant) pair — alternating turns."""
    msgs = prompts.build_messages("X", language="de_ch")
    n_demos = len(prompts._EXAMPLES_BY_LANG["de_ch"])
    # 1 system + 2*n_demos demo turns + 1 final user
    assert len(msgs) == 1 + 2 * n_demos + 1
    # Demo turns alternate user/assistant
    for i, msg in enumerate(msgs[1:-1]):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected_role


def test_few_shot_assistant_outputs_are_valid_json() -> None:
    """Every demo's assistant message must parse as a {spans: [...]} object —
    if any are malformed, Gemma's structured-output decoder will see
    inconsistent demonstrations and may misbehave."""
    import json
    for lang in prompts.SUPPORTED_PROMPT_LANGS:
        for i, ex in enumerate(prompts._EXAMPLES_BY_LANG[lang]):
            obj = json.loads(ex["assistant"])
            assert "spans" in obj, f"{lang}[{i}] missing 'spans' field"
            assert isinstance(obj["spans"], list), f"{lang}[{i}] spans not a list"
            for s in obj["spans"]:
                assert "value" in s and "label" in s
                assert s["label"] in CATEGORIES, (
                    f"{lang}[{i}] uses non-canonical label {s['label']!r}"
                )


def test_demo_assistant_values_appear_verbatim_in_user_text() -> None:
    """Every span value in a few-shot example must be a verbatim substring
    of the corresponding user text. If we ship demonstrations where the
    model can't actually find the value, we're teaching it to hallucinate."""
    import json
    for lang in prompts.SUPPORTED_PROMPT_LANGS:
        for i, ex in enumerate(prompts._EXAMPLES_BY_LANG[lang]):
            obj = json.loads(ex["assistant"])
            for s in obj["spans"]:
                assert s["value"] in ex["user"], (
                    f"{lang}[{i}] demo value {s['value']!r} not found in "
                    f"user text — bad demo, would teach hallucination"
                )
