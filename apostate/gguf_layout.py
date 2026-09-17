"""The GGUF layout facts `convert-tree` and `prepare-quant` must not answer differently.

Both commands repair the same defect from opposite ends -- one turns a bake's HF tree into a GGUF, the
other repairs a GGUF that already exists -- and both therefore need the same four answers:

* the k-quant block, and the smallest width above a declaration that a k-quant can represent;
* which tensors are MLP weights, and which axis of them carries the intermediate dimension;
* where a file's draft (MTP) block starts, which `block_count` spells two different ways;
* what to say when a width is left unaligned.

They used to answer all four privately. That is one edit away from answering them differently: a family
that puts the intermediate dimension somewhere else, or a file that counts its draft layer differently,
would be repaired correctly by one command and mis-handled silently by the other. This module is the
single answer, and `tests/test_gguf_layout.py` pins both commands to it.

Every function that can fail takes a `refuse` callable -- each command's own exception type -- so this
module stays free of the front doors' imports and neither command owns it. Nothing here imports gguf,
torch or numpy: it is arithmetic over tensor names, shapes and metadata fields, so it works while only
gguf-py is installed.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, NoReturn, Sequence

#: The k-quant block. A tensor whose `ne[0]` is not a multiple of this cannot be stored as a k-quant,
#: and llama.cpp answers with a warning and an F16 fallback rather than an error -- the +6.97 GiB this
#: project measured on a 27B, on 192 tensors.
QK_K = 256

#: The MLP projections, in llama.cpp's naming rather than any family's. Every architecture with a gated
#: MLP writes these three, which is why the repair is architecture-agnostic while the mapping *into*
#: them is not.
MLP_SUFFIXES = ("ffn_down.weight", "ffn_gate.weight", "ffn_up.weight")

#: Which *array* axis carries the intermediate dimension, per projection. gguf-py hands arrays back in
#: reversed `ne` order -- the same reversal the writer applies on the way out -- so `ffn_down`'s
#: intermediate (`ne[0]`, the fastest axis) is the array's last one, while `ffn_gate` and `ffn_up`
#: carry theirs on `ne[1]`, i.e. the array's first axis. `mlp_axis` falls back to the shape when a
#: projection is not listed, so a new family is never silently unrepaired.
MLP_AXIS = {"ffn_gate": 0, "ffn_up": 0, "ffn_down": 1}

#: `blk.<n>.` -- the only name shape the draft rules depend on.
BLOCK = re.compile(r"^blk\.(\d+)\.")


class LayoutRefused(RuntimeError):
    """A layout question whose answer would be a guess. Callers re-raise as their own refusal type."""


def is_mlp_weight(name: str) -> bool:
    """True for the three gated-MLP projections."""
    return name.endswith(MLP_SUFFIXES)


def projection_of(name: str) -> str:
    """`blk.3.ffn_gate.weight` -> `ffn_gate`; the name `MLP_AXIS` is keyed by."""
    return name.rpartition(".")[0].rpartition(".")[2]


def mlp_axis(name: str, shape: Sequence[int], width: int, refuse: Callable[[str], NoReturn]) -> int:
    """The array axis of `name`'s intermediate dimension, `width` wide.

    The pinned table first, the shape second: the shape fallback means a projection this table does not
    list is still repaired rather than ignored, and refusing when the shape is not decisive means a
    fused or expert layout is named instead of padded along the wrong axis.
    """
    axis = MLP_AXIS.get(projection_of(name))
    if axis is not None and 0 <= axis < len(shape):
        return axis
    candidates = [index for index, size in enumerate(shape) if int(size) == width]
    if len(candidates) != 1:
        refuse(
            f"{name} does not expose exactly one axis at the declared width {width} "
            f"(shape {tuple(int(size) for size in shape)}); fused or expert layouts need an explicit "
            "implementation"
        )
    return candidates[0]


def aligned_width(width: int) -> int:
    """The smallest width at or above `width` a k-quant can represent. Pure arithmetic."""
    return width + (-width % QK_K)


def require_aligned(width: int, label: str, refuse: Callable[[str], NoReturn]) -> int:
    """Refuse a requested width a k-quant cannot represent, naming the flag that asked for it."""
    if width % QK_K:
        refuse(f"{label} {width} is not a multiple of {QK_K}, so no k-quant can represent it")
    return width


def declared_width(
    fields: Mapping[str, Any], integer: Callable[[Any, str], int], refuse: Callable[[str], NoReturn]
) -> tuple[int, tuple[str, ...]]:
    """The declared feed-forward width and the keys that declared it.

    Read from the metadata rather than from a tensor, and refused when two keys disagree -- a file that
    declares two widths has no single answer, and padding the wrong one produces a header that
    contradicts its own data. A per-layer *array* is a legal GGUF spelling of the same field, so its
    entries are flattened and must agree like any other pair of declarations.
    """
    keys = tuple(name for name in fields if name.endswith(".feed_forward_length"))
    if not keys:
        refuse("no <arch>.feed_forward_length in the GGUF; cannot tell which width to align to")
    values: set[int] = set()
    for key in keys:
        value = fields[key]
        if isinstance(value, (list, tuple)):
            values.update(integer(item, key) for item in value)
        else:
            values.add(integer(value, key))
    if len(values) != 1:
        refuse(f"feed-forward metadata disagrees: {sorted(values)}")
    return values.pop(), keys


def block_count(
    fields: Mapping[str, Any], integer: Callable[[Any, str], int], refuse: Callable[[str], NoReturn]
) -> int | None:
    """`<arch>.block_count`, or None when the file does not declare one."""
    values = [fields[name] for name in fields if name.endswith(".block_count")]
    if not values:
        return None
    parsed = {integer(value, "block_count") for value in values}
    if len(parsed) != 1:
        refuse(f"block-count metadata disagrees: {sorted(parsed)}")
    return parsed.pop()


def draft_threshold(
    fields: Mapping[str, Any], integer: Callable[[Any, str], int], refuse: Callable[[str], NoReturn]
) -> int | None:
    """The first `blk.<n>` index belonging to the draft (MTP) block, or None when there is none.

    `block_count` spells the draft block two ways: a *trunk* declares only the decoder layers, so the
    draft sits at exactly `block_count`, while an MTP-bearing file declares the draft layer as a block
    too (65 for a 64-layer trunk with one MTP layer, the draft at 64). Subtracting
    `nextn_predict_layers` gives the one index right for both -- testing `index >= block_count` alone
    misses the draft on every file that counts it, which is every file this project ships.
    """
    count = block_count(fields, integer, refuse)
    if count is None:
        return None
    nextn = 0
    for name, value in fields.items():
        if name.endswith(".nextn_predict_layers"):
            nextn = integer(value, "nextn_predict_layers")
            break
    return count - nextn if nextn > 0 else count


def is_draft(name: str, threshold: int | None) -> bool:
    """True when `blk.<n>.*` belongs to the draft block, given `draft_threshold`'s index."""
    match = BLOCK.match(name)
    return threshold is not None and match is not None and int(match.group(1)) >= threshold


def unaligned_note(width: int) -> str:
    """The one sentence both commands print when a width is left unaligned, so the advice cannot drift.

    It names the consequence (F16 storage, which llama.cpp reports as a warning and not an error) and
    the number that fixes it, because an operator who sees the size and not the cause re-quantizes the
    same file.
    """
    return (
        f"width {width} is not a multiple of {QK_K}: llama.cpp stores those tensors as F16 rather than "
        f"quantizing them. Align it to {aligned_width(width)} before quantizing"
    )
