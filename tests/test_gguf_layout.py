"""The one home for the layout facts `convert-tree` and `prepare-quant` both act on.

Two commands repair the same defect from opposite ends, so both must answer the same questions the same
way. These tests pin the answers, and the last one pins the *agreement*: it fails if either command
stops importing the shared rule and grows its own copy again.

Nothing here needs a model: the rules are arithmetic over names, shapes and metadata fields.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "apostate" / "gguf_layout.py"
SPEC = importlib.util.spec_from_file_location("gguf_layout_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)

QK_K = LAYOUT.QK_K


def refuse(message: str) -> None:
    """The `refuse` callable both commands pass in; here it just raises, so a test can read the text."""
    raise ValueError(message)


def as_integer(value, label):
    return int(value)


# --- the block, and the width above a declaration ---------------------------------------------------


def test_the_block_is_llama_cpps():
    assert QK_K == 256


@pytest.mark.parametrize(
    ("width", "aligned"),
    [(17409, 17664), (17408, 17408), (17664, 17664), (1, 256), (257, 512), (5120, 5120)],
)
def test_aligned_width_rounds_up_to_the_next_block(width, aligned):
    """The additive bake leaves `base + 1`; the repair is always the *next* multiple, never a guess."""
    assert LAYOUT.aligned_width(width) == aligned
    assert LAYOUT.aligned_width(width) % QK_K == 0


def test_an_aligned_width_is_its_own_answer():
    """A tree that is already aligned must not be padded to the one above it."""
    assert LAYOUT.aligned_width(17408) == 17408


def test_require_aligned_names_the_flag():
    assert LAYOUT.require_aligned(17664, "--pad-mlp-to", refuse) == 17664
    with pytest.raises(ValueError, match=r"--pad-mlp-to 17409 is not a multiple of 256"):
        LAYOUT.require_aligned(17409, "--pad-mlp-to", refuse)


# --- which tensors are MLP, and along which axis ----------------------------------------------------


@pytest.mark.parametrize("name", ["blk.0.ffn_gate.weight", "blk.63.ffn_up.weight", "blk.64.ffn_down.weight"])
def test_mlp_weights_are_recognised_by_llama_cpp_names(name):
    assert LAYOUT.is_mlp_weight(name)


@pytest.mark.parametrize("name", ["blk.0.attn_q.weight", "token_embd.weight", "blk.0.ffn_gate_inp.weight"])
def test_non_mlp_weights_are_not(name):
    """`ffn_gate_inp` is the router of an MoE stack, not a gated projection: it must not be padded."""
    assert not LAYOUT.is_mlp_weight(name)


def test_the_pinned_axis_matches_the_shape_it_pads():
    """The table is the pinned knowledge; the shape is what it must agree with.

    `ffn_down` is `[hidden, intermediate]` in GGUF `ne` order, so its intermediate sits on the array's
    last axis once gguf-py reverses `ne`; `ffn_gate`/`ffn_up` are `[intermediate, hidden]` and sit on
    the first. Getting this backwards pads the hidden dimension and silently transposes the projection.
    """
    hidden, intermediate = 5120, 17409
    assert LAYOUT.mlp_axis("blk.0.ffn_down.weight", (hidden, intermediate), intermediate, refuse) == 1
    assert LAYOUT.mlp_axis("blk.0.ffn_gate.weight", (intermediate, hidden), intermediate, refuse) == 0
    assert LAYOUT.mlp_axis("blk.0.ffn_up.weight", (intermediate, hidden), intermediate, refuse) == 0


def test_a_projection_the_table_does_not_list_falls_back_to_the_shape():
    """A new family is still repaired rather than silently skipped."""
    assert LAYOUT.mlp_axis("blk.0.ffn_extra.weight", (7, 4096), 4096, refuse) == 1


def test_an_ambiguous_shape_is_refused_by_name():
    with pytest.raises(ValueError, match="blk.0.ffn_gate_exps.weight"):
        LAYOUT.mlp_axis("blk.0.ffn_gate_exps.weight", (17409, 17409), 17409, refuse)


# --- metadata: the declared width, and where the draft block starts ---------------------------------
# `prepare_quant._fields` unwraps each GGUF field with `.contents()` before these rules see it, so the
# rules take plain values: a test that wraps them in a field object tests a shape nothing uses.


def test_the_declared_width_is_read_from_the_architecture_key():
    fields = {"qwen35.feed_forward_length": 17409, "qwen35.block_count": 64}
    assert LAYOUT.declared_width(fields, as_integer, refuse) == (17409, ("qwen35.feed_forward_length",))


def test_a_per_layer_width_array_must_agree_with_itself():
    """GGUF can spell the field as an array; an array that disagrees has no single answer to pad to."""
    agreeing = {"qwen35.feed_forward_length": [17409, 17409, 17409]}
    assert LAYOUT.declared_width(agreeing, as_integer, refuse)[0] == 17409

    disagreeing = {"qwen35.feed_forward_length": [17409, 17408]}
    with pytest.raises(ValueError, match="disagrees"):
        LAYOUT.declared_width(disagreeing, as_integer, refuse)


def test_a_missing_width_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="feed_forward_length"):
        LAYOUT.declared_width({}, as_integer, refuse)


def test_the_draft_threshold_handles_both_spellings_of_block_count():
    """A trunk declares only decoder layers; an MTP-bearing file counts the draft layer as a block.

    `index >= block_count` alone misses the draft on every file that counts it -- which is every file
    this project ships.
    """
    trunk = {"qwen35.block_count": 64}
    assert LAYOUT.draft_threshold(trunk, as_integer, refuse) == 64

    with_draft = {"qwen35.block_count": 65, "qwen35.nextn_predict_layers": 1}
    assert LAYOUT.draft_threshold(with_draft, as_integer, refuse) == 64


def test_the_draft_block_is_the_layer_at_the_threshold():
    threshold = 64
    assert not LAYOUT.is_draft("blk.63.ffn_down.weight", threshold)
    assert LAYOUT.is_draft("blk.64.ffn_down.weight", threshold)
    assert LAYOUT.is_draft("blk.64.nextn.eh_proj.weight", threshold)
    assert not LAYOUT.is_draft("token_embd.weight", threshold)
    assert not LAYOUT.is_draft("blk.64.ffn_down.weight", None)


def test_an_unaligned_width_gets_one_sentence_in_both_commands():
    note = LAYOUT.unaligned_note(17409)
    assert "17409" in note and "17664" in note and "F16" in note


# --- the agreement guard ---------------------------------------------------------------------------


def _imported_names(path: Path) -> set[str]:
    """Every name a module imports from `gguf_layout`, however it aliases them."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "gguf_layout":
            for alias in node.names:
                names.add(alias.name)
    return names


@pytest.mark.parametrize("module", ["convert_tree.py", "prepare_quant.py"])
def test_both_commands_take_their_layout_facts_from_the_shared_module(module):
    """The guard: the moment a command answers a layout question privately again, this fails.

    It is deliberately about the *import*, not about behaviour, because that is the failure mode --
    a second copy of `QK_K`, of the next-multiple arithmetic, or of the MLP axis table reads correct
    and diverges silently later.
    """
    imported = _imported_names(REPO_ROOT / "apostate" / module)

    assert "QK_K" in imported, f"{module} no longer takes the block size from gguf_layout"
    assert "aligned_width" in imported, f"{module} no longer takes the alignment rule from gguf_layout"
    assert "is_mlp_weight" in imported, f"{module} no longer takes the MLP-name rule from gguf_layout"

    source = (REPO_ROOT / "apostate" / module).read_text(encoding="utf-8")
    assert "QK_K = 256" not in source, f"{module} declares its own block size again"
    assert "-source_width % QK_K" not in source, f"{module} recomputes the alignment rule again"
    assert '"ffn_down.weight"' not in source, f"{module} names the MLP projections privately again"
