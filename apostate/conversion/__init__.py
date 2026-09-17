"""Which model family owns a tree, and what a family has to answer.

`convert_tree` is one engine for every architecture it knows: this package is the seam between them. A
family owns everything that varies by architecture -- the GGUF arch name and the `model_type` values
that select it, the tensor-name mapping, the dtype and value transforms, the attention geometry, the
draft (MTP) layer count, the source prefixes the engine filters on, and the vocabulary's special-token
ordering -- and the engine asks the registry which family owns a tree instead of naming one itself.

Why the split: as more model types are added, `convert_tree` would otherwise grow a branch and a table
per family until nobody could read it. Here a new family is a new module beside `qwen35.py` plus one
`register()` call, and the engine does not change.

Refusing is this package's job too. `ConversionRefused` is the single refusal type the engine and every
family raise, so an unsupported tree, an unmapped tensor and a geometry whose permutation is undefined
all reach the CLI the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


class ConversionRefused(RuntimeError):
    """A refusal is information: name what is wrong and what would fix it."""


# --- config fields a family reads for itself ---------------------------------------------------------
#
# These readers live here rather than in the engine because a family reads its own geometry out of
# `config.json` and must refuse a missing or malformed field exactly as the engine does -- and a family
# importing the engine would be an import cycle. The engine imports them back.


def _need(text: Mapping[str, Any], key: str) -> Any:
    if key not in text:
        raise ConversionRefused(f"config.json (text_config) is missing {key!r}; cannot write metadata")
    return text[key]


def _need_int(text: Mapping[str, Any], key: str) -> int:
    value = _need(text, key)
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConversionRefused(f"config.json field {key!r} is not an integer: {value!r}") from error


def _optional_int(value: Any, default: int, label: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConversionRefused(f"config field {label!r} is not an integer: {value!r}") from error


@dataclass(frozen=True)
class Family:
    """Everything the engine varies by architecture, and nothing else.

    The engine is written against this object rather than against a family, so adding a family is a
    module beside `qwen35.py` plus one `register()` call -- no new branch, and no table, in the engine.

    The data the engine reads directly is small: the arch name it writes into the GGUF and its `<arch>.*`
    keys, the source prefixes it filters on, and the vocabulary facts (`space_marker`, `special_tokens`)
    it writes into the tokenizer. The rest -- the three rule tuples, the dtype rule's suffixes, the plain
    norm -- is the family's own business: the engine never walks a rule table itself, `map_name`,
    `is_f32` and `unit_offset` do.

    The callables are the family's own code, moved with the rules they implement. The engine calls them
    and never asks what is inside them:

    * `is_f32(target, out_shape)` -- upstream's dtype rule for a *target* name: True when llama.cpp keeps
      this tensor at F32 whatever the file type.
    * `unit_offset(source)` -- True for the RMSNorms this family stores as `w - 1`.
    * `map_name(source, block_count, draft)` -- the target name for a source tensor, or None.
    * `plan_transform(source, shape, geometry)` -- this tensor's transform, in whatever shape the family
      wants. The engine reads `needs_float32` off the returned value (a transform that needs float32
      arithmetic cannot be written bf16) and otherwise carries it untouched.
    * `apply_transform(tensor, transform)` -- applies such a value to the loaded tensor.
    * `linear_attention(config_text)` -- the family's attention geometry, refusing one it cannot define
      a permutation on.
    * `recurrent_layers(block_count, interval, draft_layers)` -- which layers are not full attention.
    * `draft_layer_count(config_text, sources)` -- how many speculative draft layers the tree holds.
    """

    #: The GGUF architecture name; also the `<arch>.*` key prefix.
    arch: str
    #: The `model_type` values in a tree's config.json that select this family.
    model_types: tuple[str, ...]
    # Name mapping as (pattern, gguf name): `{b}` is the block index, `{d}` the draft layer index.
    decoder_rules: tuple[tuple[str, str], ...]
    global_rules: tuple[tuple[str, str], ...]
    draft_rules: tuple[tuple[str, str], ...]
    #: Suffixes llama.cpp keeps at F32 whatever the file type, beyond every 1-D tensor.
    f32_suffixes: tuple[str, ...]
    #: The one norm stored as a plain weight rather than as `w - 1`; `unit_offset` reads it.
    plain_norm: str
    #: Names whose remapped GGUF spelling ends in `norm.weight` but whose source spelling does not.
    unit_offset_aliases: tuple[str, ...]
    #: Source names under this prefix are the vision tower (skipped) and the draft block (on request).
    vision_prefix: str
    draft_prefix: str
    #: The marker the tree's vocabulary uses for a space, unescaped in the token text.
    space_marker: str
    #: `SpecialVocab`'s token types, in the order upstream resolves them.
    special_tokens: tuple[str, ...]
    # The family's own code; the class docstring above says what each one has to answer.
    is_f32: Callable[[str, Sequence[int]], bool]
    unit_offset: Callable[[str], bool]
    map_name: Callable[[str, int, bool], str | None]
    plan_transform: Callable[[str, Sequence[int], Any], Any]
    apply_transform: Callable[[Any, Any], Any]
    linear_attention: Callable[[Mapping[str, Any]], Any]
    recurrent_layers: Callable[[int, int, int], list[bool]]
    draft_layer_count: Callable[[Mapping[str, Any], Iterable[str]], int]


#: Every registered family, keyed by architecture name.
FAMILIES: dict[str, Family] = {}


def register(family: Family) -> Family:
    """Add a family to the registry and return it, so a module can name its own family at import.

    A duplicate architecture is refused rather than overwritten: two families claiming one arch name
    would silently make every tree of that architecture resolve to whichever was imported last.
    """
    existing = FAMILIES.get(family.arch)
    if existing is not None:
        raise ConversionRefused(
            f"architecture {family.arch!r} is already registered for model types "
            f"{', '.join(existing.model_types)}; refusing to replace it"
        )
    FAMILIES[family.arch] = family
    return family


def family_for_config(config: Mapping[str, Any]) -> Family:
    """The family that owns this tree, by the `model_type` its config declares.

    A checkpoint that wraps a vision tower keeps the language stack's own `model_type` under
    `text_config`, so the nested value is the fallback and the top-level one wins. Refuses a type no
    family claims, naming the ones that ARE supported rather than leaving the operator to guess.
    """
    nested = config.get("text_config")
    text = nested if isinstance(nested, dict) else config
    model_type = str(config.get("model_type") or text.get("model_type") or "")
    for family in FAMILIES.values():
        if model_type in family.model_types:
            return family
    supported = sorted({name for family in FAMILIES.values() for name in family.model_types})
    raise ConversionRefused(
        f"model_type {model_type!r} is not supported by this converter "
        f"(supported: {', '.join(supported)}); use llama.cpp's own converter"
    )


# Importing the family modules is what registers them, and it happens here so that `FAMILIES` is complete
# for anyone who imports the registry -- the engine in particular never has to know a family's module.
from . import qwen35  # noqa: E402,F401
