"""The Qwen3.5/3.8 family: everything a bf16 tree of this family needs to become a bf16 GGUF.

The engine (`apostate.convert_tree`) reads the tree, decides every job and writes the bytes; what is
written differently *because the tree is this family* is here. Registering the family at the bottom of
this module is the whole of what the engine has to know about it.

Three per-architecture transforms separate a bf16 tree from a bf16 GGUF. Each was read off upstream's
Qwen3.5 model class (`Qwen3NextModel.modify_tensors` and `_LinearAttentionVReorderBase`) and then
confirmed byte-for-byte against a GGUF upstream produced from the same tree, which is also how the
tokenizer details below were settled:

* **RMSNorm.** This family stores `w - 1`, so every `*norm.weight` **except** `linear_attn.norm.weight`
  is written `w + 1`, in float32.
* **`ssm_a`.** The tree holds `A_log`; the GGUF holds `-exp(A_log)` in float32. Computed with torch
  rather than numpy, because the two disagree in the last ulp on 18 of the 48 values of a real layer.
* **V-head order.** HF groups the linear-attention V heads by K head, ggml broadcasts them tiled, so
  `attn_qkv` (V rows only), `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`
  (V channels only) and `ssm_out` (columns) are permuted. A permutation, so it runs before the bf16
  reinterpretation and costs nothing.

Refuses rather than guessing: a tensor whose shape contradicts the geometry its permutation is defined
against, a linear-attention geometry whose V heads do not divide into the K heads, and a draft layer
count the tree and the config disagree on. (The refusals that are not family facts -- an unmapped name,
a missing shard, an architecture nobody claims -- are the engine's.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import ConversionRefused, Family, _need_int, _optional_int, register

# The vision tower is skipped (a text GGUF has no use for it) and the draft (MTP) block is written only
# when the caller asks for it, so these two prefixes are what the engine filters source names by.
VISION_PREFIX = "model.visual."
DRAFT_PREFIX = "mtp."

# Tensor name mapping, measured from a real tree against the GGUF upstream produced from it. Each entry
# is (pattern, gguf name); `{b}` is the block index and every pattern is anchored.
_DECODER_RULES: tuple[tuple[str, str], ...] = (
    (r"^model\.language_model\.layers\.(\d+)\.input_layernorm\.weight$", "blk.{b}.attn_norm.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.post_attention_layernorm\.weight$", "blk.{b}.post_attention_norm.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "blk.{b}.ffn_gate.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.weight$", "blk.{b}.ffn_up.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$", "blk.{b}.ffn_down.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "blk.{b}.attn_q.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "blk.{b}.attn_k.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "blk.{b}.attn_v.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "blk.{b}.attn_output.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.q_norm\.weight$", "blk.{b}.attn_q_norm.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.self_attn\.k_norm\.weight$", "blk.{b}.attn_k_norm.weight"),
    # The linear-attention (SSM) layers; `in_proj_z` is the output gate and `out_proj` the writer, both
    # confirmed by shape against the reference GGUF rather than by name.
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight$", "blk.{b}.attn_qkv.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_z\.weight$", "blk.{b}.attn_gate.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.out_proj\.weight$", "blk.{b}.ssm_out.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_a\.weight$", "blk.{b}.ssm_alpha.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.in_proj_b\.weight$", "blk.{b}.ssm_beta.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.conv1d\.weight$", "blk.{b}.ssm_conv1d.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.norm\.weight$", "blk.{b}.ssm_norm.weight"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.A_log$", "blk.{b}.ssm_a"),
    (r"^model\.language_model\.layers\.(\d+)\.linear_attn\.dt_bias$", "blk.{b}.ssm_dt.bias"),
)

_GLOBAL_RULES: tuple[tuple[str, str], ...] = (
    (r"^model\.language_model\.embed_tokens\.weight$", "token_embd.weight"),
    (r"^model\.language_model\.norm\.weight$", "output_norm.weight"),
    (r"^lm_head\.weight$", "output.weight"),
)

# The draft (MTP) block, written only with `--with-mtp`. Its layer index is the trunk's block count --
# llama.cpp's dummy block -- and its four MTP-only weights take the `nextn` names every MTP-bearing GGUF
# in this project carries (`blk.<trunk>.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`), which is
# what upstream's `_QwenMtpMixin` remaps `mtp.*` onto.
_DRAFT_RULES: tuple[tuple[str, str], ...] = (
    (r"^mtp\.fc\.weight$", "blk.{d}.nextn.eh_proj.weight"),
    (r"^mtp\.norm\.weight$", "blk.{d}.nextn.shared_head_norm.weight"),
    (r"^mtp\.pre_fc_norm_embedding\.weight$", "blk.{d}.nextn.enorm.weight"),
    (r"^mtp\.pre_fc_norm_hidden\.weight$", "blk.{d}.nextn.hnorm.weight"),
    (r"^mtp\.layers\.(\d+)\.input_layernorm\.weight$", "blk.{d}.attn_norm.weight"),
    (r"^mtp\.layers\.(\d+)\.post_attention_layernorm\.weight$", "blk.{d}.post_attention_norm.weight"),
    (r"^mtp\.layers\.(\d+)\.mlp\.gate_proj\.weight$", "blk.{d}.ffn_gate.weight"),
    (r"^mtp\.layers\.(\d+)\.mlp\.up_proj\.weight$", "blk.{d}.ffn_up.weight"),
    (r"^mtp\.layers\.(\d+)\.mlp\.down_proj\.weight$", "blk.{d}.ffn_down.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.q_proj\.weight$", "blk.{d}.attn_q.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.k_proj\.weight$", "blk.{d}.attn_k.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.v_proj\.weight$", "blk.{d}.attn_v.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.o_proj\.weight$", "blk.{d}.attn_output.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.q_norm\.weight$", "blk.{d}.attn_q_norm.weight"),
    (r"^mtp\.layers\.(\d+)\.self_attn\.k_norm\.weight$", "blk.{d}.attn_k_norm.weight"),
)

# Tensors llama.cpp keeps at F32 whatever the file type: every 1-D tensor, every norm, and the 2-D SSM
# state upstream names explicitly.
_F32_SUFFIXES = ("_norm.weight", "norm.weight", ".ssm_a", ".ssm_dt.bias", ".ssm_conv1d.weight")
# The one norm this family stores as a plain weight, plus the two draft names whose HF spelling does not
# end in `norm.weight` but whose remapped one (`enorm`/`hnorm`) does.
_PLAIN_NORM = "linear_attn.norm.weight"
_UNIT_OFFSET_ALIASES = ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight")
# `SpecialVocab`'s types in the order upstream resolves them: a name in `tokenizer_config.json` beats an
# id in `config.json`, and the first source to set a type keeps it.
_SPECIAL_TOKENS = ("bos", "eos", "unk", "sep", "pad", "cls", "mask")
_SPACE_MARKER = "\u2581"


def _is_f32(name: str, out_shape: Sequence[int]) -> bool:
    """Upstream's dtype rule: `n_dims <= 1 or new_name.endswith("_norm.weight")`, plus the SSM state."""
    return len(out_shape) <= 1 or any(name.endswith(suffix) for suffix in _F32_SUFFIXES)


def _unit_offset(source: str) -> bool:
    """True for the RMSNorms this family stores as `w - 1`."""
    if source.endswith(_PLAIN_NORM):
        return False
    return source.endswith("norm.weight") or source in _UNIT_OFFSET_ALIASES


@dataclass(frozen=True)
class LinearAttention:
    """The SSM geometry the V-head reorder is defined against."""

    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int

    @property
    def per_key(self) -> int:
        return self.value_heads // self.key_heads

    def order(self, dim: int, head_dim: int, split: int | None = None) -> "VHeadOrder | None":
        """The permutation for one tensor, or None when upstream reorders nothing (equal head counts)."""
        if self.key_heads == self.value_heads:
            return None
        return VHeadOrder(dim, head_dim, self.key_heads, self.per_key, split)


@dataclass(frozen=True)
class VHeadOrder:
    """Grouped V heads -> tiled, on one axis (upstream's `_reorder_v_heads`).

    HF stores the V heads grouped by K head -- `[G0_v0..v{r-1}, G1_v0..v{r-1}, ...]` -- while ggml
    broadcasts them tiled -- `[K0, K1, ..., K0, K1, ...]`. `split` is the length of the leading run that
    is left alone (`attn_qkv`'s q|k rows, `ssm_conv1d`'s q|k channels); `dim` 1 is `ssm_out`'s input
    columns, the one tensor reordered along its second axis.
    """

    dim: int
    head_dim: int
    key_heads: int
    per_key: int
    split: int | None = None


@dataclass(frozen=True)
class Transform:
    """What has to happen to a tensor's values before they are written."""

    unit_offset: bool = False
    negate_exp: bool = False
    reorder: VHeadOrder | None = None

    @property
    def needs_float32(self) -> bool:
        return self.unit_offset or self.negate_exp


def linear_attention(text: Mapping[str, Any]) -> LinearAttention:
    """The SSM head geometry, refusing a ratio the V-head reorder cannot be defined on."""
    key_heads = _need_int(text, "linear_num_key_heads")
    value_heads = _need_int(text, "linear_num_value_heads")
    if key_heads <= 0 or value_heads <= 0:
        raise ConversionRefused(
            f"linear_num_key_heads/linear_num_value_heads must be positive, got {key_heads}/{value_heads}"
        )
    if value_heads % key_heads:
        raise ConversionRefused(
            f"linear_num_value_heads ({value_heads}) is not a multiple of linear_num_key_heads "
            f"({key_heads}); the linear-attention V-head order is undefined"
        )
    return LinearAttention(
        key_heads=key_heads,
        value_heads=value_heads,
        key_dim=_need_int(text, "linear_key_head_dim"),
        value_dim=_need_int(text, "linear_value_head_dim"),
    )


def _expect_axis(source: str, shape: Sequence[int], axis: int, size: int) -> None:
    """Refuse a tensor whose shape contradicts the geometry its permutation is defined against."""
    if len(shape) <= axis or shape[axis] != size:
        raise ConversionRefused(
            f"{source} has shape {tuple(shape)}, which does not carry {size} on axis {axis} as the "
            "linear-attention geometry requires; refusing to permute a tensor this converter cannot place"
        )


def plan_transform(source: str, shape: Sequence[int], linear: LinearAttention) -> Transform:
    """Every value transform upstream applies to this tensor, decided by name and shape.

    Upstream decides this by name in `modify_tensors`, on the name *after* `mtp.*` has been remapped onto
    the dummy layer -- which is why the draft head's two pre-norm weights count as `norm.weight` here
    even though their HF spelling does not end that way. The shapes are checked rather than trusted: a
    permutation applied to the wrong axis produces a file that loads and is wrong.
    """
    if source.endswith(".linear_attn.A_log"):
        _expect_axis(source, shape, 0, linear.value_heads)
        return Transform(negate_exp=True, reorder=linear.order(0, 1))
    if source.endswith(".linear_attn.dt_bias"):
        _expect_axis(source, shape, 0, linear.value_heads)
        return Transform(reorder=linear.order(0, 1))
    if source.endswith(".linear_attn.in_proj_qkv.weight"):
        qk = 2 * linear.key_heads * linear.key_dim
        _expect_axis(source, shape, 0, qk + linear.value_heads * linear.value_dim)
        return Transform(reorder=linear.order(0, linear.value_dim, split=qk))
    if source.endswith(".linear_attn.in_proj_z.weight"):
        _expect_axis(source, shape, 0, linear.value_heads * linear.value_dim)
        return Transform(reorder=linear.order(0, linear.value_dim))
    if source.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight")):
        _expect_axis(source, shape, 0, linear.value_heads)
        return Transform(reorder=linear.order(0, 1))
    if source.endswith(".linear_attn.conv1d.weight"):
        qk = 2 * linear.key_heads * linear.key_dim
        _expect_axis(source, shape, 0, qk + linear.value_heads * linear.value_dim)
        return Transform(reorder=linear.order(0, linear.value_dim, split=qk))
    if source.endswith(".linear_attn.out_proj.weight"):
        _expect_axis(source, shape, 1, linear.value_heads * linear.value_dim)
        return Transform(reorder=linear.order(1, linear.value_dim))
    return Transform(unit_offset=_unit_offset(source))


def _map_name(source: str, block_count: int, draft: bool) -> str | None:
    """This tensor's GGUF name, or None when no rule matches (the engine refuses rather than dropping)."""
    rules = _DRAFT_RULES if draft else _DECODER_RULES + _GLOBAL_RULES
    for pattern, template in rules:
        match = re.match(pattern, source)
        if match is None:
            continue
        if "{b}" in template:
            return template.format(b=int(match.group(1)))
        if "{d}" in template:
            # the MTP layer index, or 0 for the four head weights that sit on the first dummy layer
            index = int(match.group(1)) if match.groups() else 0
            return template.format(d=block_count + index)
        return template
    return None


def _reorder(tensor: Any, order: VHeadOrder) -> Any:
    """One axis, grouped V heads -> tiled (upstream's `_reorder_v_heads`)."""
    import torch

    body = tensor if order.split is None else tensor[order.split :]
    if order.dim == 0:
        view = body.reshape((order.key_heads, order.per_key, order.head_dim) + tuple(body.shape[1:]))
        tiled = view.transpose(0, 1).reshape(body.shape)
    else:
        view = body.reshape((body.shape[0], order.key_heads, order.per_key, order.head_dim))
        tiled = view.transpose(1, 2).reshape(body.shape)
    tiled = tiled.contiguous()  # the writer reinterprets this tensor as uint16 for a bf16 target
    if order.split is None:
        return tiled
    return torch.cat((tensor[: order.split], tiled), dim=0)


def _transformed(tensor: Any, transform: Transform) -> Any:
    """Apply the value transforms, in the order upstream applies them."""
    import torch

    if transform.reorder is not None:
        tensor = _reorder(tensor, transform.reorder)
    if transform.negate_exp:
        # torch, not numpy: the two disagree in the last ulp on 18 of the 48 values of a real layer.
        return -torch.exp(tensor.float())
    if transform.unit_offset:
        return tensor.float() + 1.0
    return tensor


def recurrent_layers(block_count: int, interval: int, draft_layers: int = 0) -> list[bool]:
    """Which layers are linear-attention; the draft (MTP) layers are full attention."""
    if interval <= 0:
        raise ConversionRefused(f"full_attention_interval must be positive, got {interval}")
    return [((index + 1) % interval) != 0 for index in range(block_count)] + [False] * draft_layers


def draft_layer_count(text: Mapping[str, Any], sources: Iterable[str]) -> int:
    """How many MTP layers the tree holds; the tree and the config must agree.

    The tree is the artifact and the config is a claim, but together they decide both the dummy block
    index and the block count, so a disagreement would place tensors outside the model's own block count.
    """
    indices = {
        int(match.group(1))
        for source in sources
        if (match := re.match(r"^mtp\.layers\.(\d+)\.", source)) is not None
    }
    observed = max(indices, default=-1) + 1
    declared = _optional_int(text.get("mtp_num_hidden_layers"), 0, "mtp_num_hidden_layers")
    if declared and observed and declared != observed:
        raise ConversionRefused(
            f"config.json declares mtp_num_hidden_layers {declared} but the tree holds {observed} "
            "draft layers; the block count and the tensor indices would disagree"
        )
    return observed or declared


#: `model_type` in the tree's config.json -> the GGUF architecture name, for both spellings a 3.5/3.8
#: tree uses: the standalone one and the one a vision-wrapped checkpoint keeps under `text_config`.
QWEN35 = register(
    Family(
        arch="qwen35",
        model_types=("qwen3_5", "qwen3_5_text"),
        decoder_rules=_DECODER_RULES,
        global_rules=_GLOBAL_RULES,
        draft_rules=_DRAFT_RULES,
        f32_suffixes=_F32_SUFFIXES,
        plain_norm=_PLAIN_NORM,
        unit_offset_aliases=_UNIT_OFFSET_ALIASES,
        vision_prefix=VISION_PREFIX,
        draft_prefix=DRAFT_PREFIX,
        space_marker=_SPACE_MARKER,
        special_tokens=_SPECIAL_TOKENS,
        is_f32=_is_f32,
        unit_offset=_unit_offset,
        map_name=_map_name,
        plan_transform=plan_transform,
        apply_transform=_transformed,
        linear_attention=linear_attention,
        recurrent_layers=recurrent_layers,
        draft_layer_count=draft_layer_count,
    )
)
