"""The seam between the engine and a model family: the registry, and what it refuses.

`convert_tree` is one engine for every architecture a family claims, so the promise these tests pin is
narrow and load-bearing: a tree's `model_type` selects exactly one family, a type no family claims is
refused by naming the ones that are supported, and a second family can be added without touching the
engine. Nothing here needs a model, a GPU or torch -- a family is data plus callables, and the fake family
below is deliberately one the engine is never handed a tree for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apostate.conversion import (  # noqa: E402
    FAMILIES,
    ConversionRefused,
    Family,
    family_for_config,
    qwen35,
    register,
)


def _not_reached(name: str):
    """A callable a `Family` must carry, whose only job here is to prove the engine never called it."""

    def call(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"the engine reached {name} on a family registered only to test the registry")

    return call


def _fake_family(arch: str, model_type: str) -> Family:
    """The smallest registerable family: no rules, no geometry, and callables that refuse to be used."""
    return Family(
        arch=arch,
        model_types=(model_type,),
        decoder_rules=(),
        global_rules=(),
        draft_rules=(),
        f32_suffixes=(),
        plain_norm="",
        unit_offset_aliases=(),
        vision_prefix="vision.",
        draft_prefix="draft.",
        space_marker="\u2581",
        special_tokens=("bos", "eos"),
        is_f32=_not_reached("is_f32"),
        unit_offset=_not_reached("unit_offset"),
        map_name=_not_reached("map_name"),
        plan_transform=_not_reached("plan_transform"),
        apply_transform=_not_reached("apply_transform"),
        linear_attention=_not_reached("linear_attention"),
        recurrent_layers=_not_reached("recurrent_layers"),
        draft_layer_count=_not_reached("draft_layer_count"),
    )


@pytest.fixture
def fake_family():
    """A second registered family, removed again so the real registry is left exactly as it was."""
    family = register(_fake_family("fake36", "fake_3_6"))
    yield family
    assert FAMILIES.pop(family.arch, None) is family


def test_both_qwen_model_types_resolve_to_the_qwen_family():
    assert family_for_config({"model_type": "qwen3_5"}) is qwen35.QWEN35
    # A checkpoint that wraps a vision tower keeps the language stack's own type under `text_config`.
    assert family_for_config({"text_config": {"model_type": "qwen3_5_text"}}) is qwen35.QWEN35
    assert FAMILIES["qwen35"] is qwen35.QWEN35


def test_an_unknown_model_type_is_refused_by_naming_what_is_supported():
    with pytest.raises(ConversionRefused, match="not supported") as refusal:
        family_for_config({"model_type": "llama"})

    message = str(refusal.value)
    assert "llama" in message
    assert "qwen3_5" in message and "qwen3_5_text" in message


def test_a_second_family_is_selected_by_its_own_model_type(fake_family):
    assert family_for_config({"model_type": "fake_3_6"}) is fake_family
    # The top-level `model_type` is the truth for a vision-wrapped tree; the nested one is the fallback.
    assert family_for_config({"model_type": "fake_3_6", "text_config": {"model_type": "qwen3_5_text"}}) is fake_family
    # A tree the second family does not claim still resolves to the family that does.
    assert family_for_config({"model_type": "qwen3_5"}) is qwen35.QWEN35


def test_a_duplicate_architecture_is_refused_and_leaves_the_registry_alone():
    with pytest.raises(ConversionRefused, match="already registered"):
        register(_fake_family("qwen35", "qwen3_5_clone"))

    assert FAMILIES["qwen35"] is qwen35.QWEN35
    with pytest.raises(ConversionRefused, match="not supported"):
        family_for_config({"model_type": "qwen3_5_clone"})
