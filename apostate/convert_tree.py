"""Convert a safetensors tree straight to BF16 GGUF, with the diode's width fixed.

`apostate prepare-quant` starts from a GGUF that already exists. A bake does not produce one: it produces
an HF tree whose `intermediate_size` is `base + 1` (17408 -> 17409 on Qwen3.8-27B), which no k-quant
block can represent, so stock tooling has nothing to quantize until that width is repaired. This module
is the step before `prepare-quant`, and it repairs the width **in the same pass** -- one read of the
shards, one write of the GGUF, no second conversion.

Why not the stock converter. Measured on this project's 27B tree: `convert_hf_to_gguf.py` moves it at
~4.5 MB/s against hardware that writes at 1221 MB/s -- about 3.4 hours, and the cost is per tensor
(~40 s each against ~1.2 s of work, with 99 s of a 150 s profile inside `_thread.lock.acquire`). The
saving here is structural rather than a tuning trick: **a bf16 source written as BF16 GGUF needs no
dtype conversion at all**. Upstream routes bf16 -> float32 -> BF16 through `quantize_blocks`; the same
16 bits reinterpreted as `uint16` measured 834 MB/s end to end. The whole conversion, padding included,
took 82 s on the real tree.

What varies by architecture does not live here. This module is the engine -- read the tree, decide every
job, write the bytes -- plus the CLI, and it asks `apostate.conversion` for the family that owns the
tree's `model_type`. The transforms that family applies to values, and why each was measured, are
documented with the family that applies them (`apostate/conversion/qwen35.py`), because they are that
family's and not the engine's.

Refuses rather than guessing: an architecture no family claims, a tensor with no mapping, a shard the
index names but the tree does not have, a geometry the family cannot define a permutation on, a shape
that contradicts the config, and an output path that already exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# `ConversionRefused` is the one refusal type a caller catches, the registry resolves the family that owns
# a tree, and the int readers are shared because a family reads its own geometry fields with them. The
# float readers below are the engine's own: families do not write metadata.
from .conversion import (
    ConversionRefused,
    Family,
    _need,
    _need_int,
    _optional_int,
    family_for_config,
)
from .gguf_layout import (
    MLP_AXIS,
    QK_K,
    aligned_width,
    is_mlp_weight,
    projection_of,
    unaligned_note,
)

#: 32 is MOSTLY_BF16 in llama.cpp's LlamaFileType enum.
FILE_TYPE_BF16 = 32


def _deps():
    try:
        import numpy as np
        import torch
        from gguf import GGMLQuantizationType, GGUFWriter
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - the gguf extra installs all four
        raise ConversionRefused(
            f"needs numpy, torch, gguf and safetensors: pip install 'apostate[gguf]' plus torch and "
            f"safetensors ({error})"
        ) from error
    return np, torch, GGMLQuantizationType, GGUFWriter, safe_open


@dataclass(frozen=True)
class Job:
    """One source tensor and where it lands, decided before a byte is written.

    `transform` is the family's own value: the engine carries it from `Family.plan_transform` to
    `Family.apply_transform` untouched, and reads only `needs_float32` off it -- a transform that has to
    do float32 arithmetic cannot be written into a bf16 file.
    """

    source: str
    target: str
    shard: str
    shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    f32: bool
    transform: Any
    pad_axis: int | None = None
    squeeze: bool = False


@dataclass
class Receipt:
    """What the conversion did, in one shape the CLI output and `--receipt` both read."""

    tree: str
    out: str
    architecture: str
    block_count: int
    source_width: int
    target_width: int
    padded: str
    tensors: int
    padded_tensors: int
    draft_tensors: int
    vision_tensors_skipped: int
    bytes_written: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def document(self) -> dict[str, Any]:
        return {
            "schema": "apostate.tree-conversion.v1",
            "tree": self.tree,
            "out": self.out,
            "architecture": self.architecture,
            "block_count": self.block_count,
            "source_width": self.source_width,
            "target_width": self.target_width,
            "padding": self.padded,
            "tensors": self.tensors,
            "padded_tensors": self.padded_tensors,
            "draft_tensors": self.draft_tensors,
            "vision_tensors_skipped": self.vision_tensors_skipped,
            "bytes_written": self.bytes_written,
            "seconds": round(self.seconds, 2),
            "warnings": list(self.warnings),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConversionRefused(f"{path} is missing") from error
    except json.JSONDecodeError as error:
        raise ConversionRefused(f"{path} is not valid JSON: {error}") from error


def text_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The language stack's config; these trees keep the real dimensions under `text_config`."""
    nested = config.get("text_config")
    return dict(nested) if isinstance(nested, dict) else dict(config)


def _need_float(text: Mapping[str, Any], key: str) -> float:
    value = _need(text, key)
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConversionRefused(f"config.json field {key!r} is not a number: {value!r}") from error


def _optional_float(value: Any, default: float, label: str) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConversionRefused(f"config field {label!r} is not a number: {value!r}") from error


def resolve_architecture(config: Mapping[str, Any]) -> str:
    """The GGUF architecture name for this tree: whichever family the registry resolves.

    The lookup itself -- which `model_type` values are supported, and what to say when one is not --
    belongs to the registry, so this is the name out of the family it returns.
    """
    return family_for_config(config).arch


def resolve_widths(text: Mapping[str, Any], pad_mlp_to: int | None, no_pad: bool) -> tuple[int, int, str]:
    """`(source_width, target_width, mode)`: the pad target is computed, never typed in.

    The bake leaves `intermediate_size` at `base + 1`, so the default is the next multiple of the block
    size. Asking an operator to type `17664` is how a run ends up with a width that does not match the
    model, so the arithmetic belongs here; `--pad-mlp-to` overrides it and `--no-pad` keeps the tree's
    own width for a lineage artifact.
    """
    source_width = _need_int(text, "intermediate_size")
    if no_pad:
        return source_width, source_width, "none"
    if pad_mlp_to is not None:
        if pad_mlp_to < source_width:
            raise ConversionRefused(f"--pad-mlp-to {pad_mlp_to} is narrower than the tree's {source_width}")
        if pad_mlp_to % QK_K:
            raise ConversionRefused(f"--pad-mlp-to {pad_mlp_to} is not a multiple of {QK_K}")
        return source_width, pad_mlp_to, "explicit"
    if source_width % QK_K == 0:
        return source_width, source_width, "none"
    return source_width, aligned_width(source_width), "auto"


def plan_tensors(
    tree: Path, *, family: Family, text: Mapping[str, Any], block_count: int, hidden_size: int,
    target_width: int, with_mtp: bool,
) -> tuple[list[Job], int]:
    """Every tensor to write, with its final shape. Reads safetensors headers only.

    The family answers every question the tree's spelling decides -- which sources are skipped, what each
    name becomes, what dtype it needs and what has to happen to its values -- so this loop is the same
    whichever family the registry resolved. `text` is the language stack's config, which is where a family
    reads its own geometry.
    """
    np, _torch, _types, _writer, safe_open = _deps()
    linear = family.linear_attention(text)
    index_path = tree / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ConversionRefused(
            f"{index_path} is missing; this converter needs a sharded tree with its index "
            "(a single-file tree can pass --tree with the index written beside it)"
        )
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ConversionRefused(f"{index_path} has no usable weight_map")

    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = tree / shard
        if not shard_path.is_file():
            raise ConversionRefused(f"the index names a shard that is not in the tree: {shard_path}")
        with safe_open(str(shard_path), framework="pt") as handle:
            for key in handle.keys():
                shapes[key] = tuple(int(n) for n in handle.get_slice(key).get_shape())

    jobs: list[Job] = []
    vision_skipped = 0
    for source in sorted(weight_map):
        if source.startswith(family.vision_prefix):
            vision_skipped += 1
            continue
        is_draft = source.startswith(family.draft_prefix)
        if is_draft and not with_mtp:
            continue
        target = family.map_name(source, block_count, is_draft)
        if target is None:
            raise ConversionRefused(
                f"no mapping for tensor {source!r}; refusing to guess (add a rule or skip the tensor "
                "deliberately)"
            )
        shape = shapes[source]
        squeeze = target.endswith("ssm_conv1d.weight") and len(shape) == 3 and shape[1] == 1
        out_shape = (shape[0], shape[2]) if squeeze else shape
        f32 = family.is_f32(target, out_shape)
        transform = family.plan_transform(source, shape, linear)
        if transform.needs_float32 and not f32:
            raise ConversionRefused(
                f"{target} ({source}) needs float32 arithmetic but is not written as F32"
            )
        pad_axis = None
        if is_mlp_weight(target):
            # The requested width binds *every* MLP, the draft block included: llama.cpp allocates every
            # block, `nextn` included, from `feed_forward_length`, and a draft head left at its own
            # width makes the file unloadable. The appended zeros are inert.
            axes = [axis for axis, size in enumerate(out_shape) if size != hidden_size]
            if len(axes) != 1:
                raise ConversionRefused(
                    f"{source} does not expose exactly one axis that is not the hidden size "
                    f"{hidden_size} (shape {out_shape})"
                )
            axis = axes[0]
            # The GGUF-level repair in `prepare-quant` pads along the *pinned* axis from the shared
            # table. Two answers to "which axis is intermediate" is how one command silently transposes
            # a projection the other pads, so the derived axis must agree with the pinned one.
            pinned = MLP_AXIS.get(projection_of(target))
            if pinned is not None and pinned != axis:
                raise ConversionRefused(
                    f"{source} ({target}) carries its intermediate dimension on axis {axis} by shape, "
                    f"but llama.cpp's pinned layout for {projection_of(target)} puts it on axis "
                    f"{pinned}; refusing to pad along a different axis than the repair would"
                )
            if out_shape[axis] > target_width:
                raise ConversionRefused(
                    f"{source} is {out_shape[axis]} wide, wider than the target {target_width}"
                )
            if out_shape[axis] < target_width:
                pad_axis = axis
                out_shape = tuple(
                    target_width if index == axis else size for index, size in enumerate(out_shape)
                )
        jobs.append(
            Job(
                source=source, target=target, shard=weight_map[source], shape=shape,
                out_shape=tuple(out_shape), f32=f32, pad_axis=pad_axis, squeeze=squeeze,
                transform=transform,
            )
        )
    return jobs, vision_skipped


def payload(handle: Any, job: Job, family: Family) -> Any:
    """The tensor's bytes in the dtype the writer expects, transformed and copied once.

    The value transforms belong to the family, so they are applied by it -- here, at the one point where
    the tensor's values are in hand and before the bf16 reinterpretation.
    """
    np, torch, _types, _writer, _open = _deps()
    tensor = handle.get_tensor(job.source)
    if job.squeeze:
        tensor = tensor.squeeze(1)
    tensor = family.apply_transform(tensor, job.transform)
    if job.f32:
        array = tensor.float().numpy()
    elif tensor.dtype in (torch.bfloat16, torch.float16):
        # The whole point: bf16 -> GGUF BF16 is a reinterpretation, not a conversion.
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.float().numpy()
    if job.pad_axis is not None:
        pads = [(0, 0)] * array.ndim
        pads[job.pad_axis] = (0, job.out_shape[job.pad_axis] - array.shape[job.pad_axis])
        array = np.pad(array, pads, mode="constant")
    return np.ascontiguousarray(array)


def set_metadata(
    writer: Any, *, family: Family, config: Mapping[str, Any], generation: Mapping[str, Any],
    target_width: int, name: str, draft_layers: int,
) -> None:
    """The `<arch>.*` keys llama.cpp reads, refusing a missing one rather than writing a default."""
    text = text_config(config)
    block_count = _need_int(text, "num_hidden_layers")
    head_dim = _optional_int(
        text.get("head_dim"),
        _need_int(text, "hidden_size") // _need_int(text, "num_attention_heads"),
        "head_dim",
    )
    rope = dict(text.get("rope_parameters") or {})
    sections = list(rope.get("mrope_section") or [])
    interval = _optional_int(text.get("full_attention_interval"), 4, "full_attention_interval")

    writer.add_type("model")
    writer.add_name(name)
    # The draft layers count as blocks: llama.cpp's dummy layer index is the trunk's layer count.
    writer.add_block_count(block_count + draft_layers)
    writer.add_context_length(_need_int(text, "max_position_embeddings"))
    writer.add_embedding_length(_need_int(text, "hidden_size"))
    writer.add_feed_forward_length(target_width)
    writer.add_head_count(_need_int(text, "num_attention_heads"))
    writer.add_head_count_kv(_need_int(text, "num_key_value_heads"))
    writer.add_layer_norm_rms_eps(_need_float(text, "rms_norm_eps"))
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_freq_base(
        _optional_float(rope.get("rope_theta") or text.get("rope_theta"), 0.0, "rope_theta")
    )
    writer.add_rope_dimension_count(
        round(head_dim * _optional_float(text.get("partial_rotary_factor"), 1.0, "partial_rotary_factor"))
    )
    if sections:
        # llama.cpp expects four sections; the config carries three.
        writer.add_rope_dimension_sections(
            [_optional_int(v, 0, "mrope_section") for v in sections] + [0] * (4 - len(sections))
        )
    writer.add_ssm_conv_kernel(_need_int(text, "linear_conv_kernel_dim"))
    writer.add_ssm_state_size(_need_int(text, "linear_key_head_dim"))
    writer.add_ssm_group_count(_need_int(text, "linear_num_key_heads"))
    writer.add_ssm_time_step_rank(_need_int(text, "linear_num_value_heads"))
    writer.add_ssm_inner_size(
        _need_int(text, "linear_num_value_heads") * _need_int(text, "linear_value_head_dim")
    )
    writer.add_array(
        f"{family.arch}.attention.recurrent_layers",
        family.recurrent_layers(block_count, interval, draft_layers),
    )
    writer.add_uint32(f"{family.arch}.full_attention_interval", interval)
    if draft_layers:
        writer.add_nextn_predict_layers(draft_layers)
    writer.add_file_type(FILE_TYPE_BF16)
    for key in ("top_k", "top_p", "temperature"):
        value = generation.get(key)
        if value is None:
            continue
        gguf_key = "general.sampling." + ("temp" if key == "temperature" else key)
        if isinstance(value, int) and not isinstance(value, bool):
            writer.add_uint32(gguf_key, value)
        else:
            writer.add_float32(gguf_key, _optional_float(value, 0.0, gguf_key))
    writer.add_string("tree_conversion.utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _vocab_size(vocab: Mapping[str, int], added: Mapping[int, Any], text: Mapping[str, Any]) -> int:
    """The token table's length: the declared size, else the tokenizer's own range."""
    highest = max([max(vocab.values(), default=-1), *added]) + 1
    declared = _optional_int(text.get("vocab_size"), 0, "vocab_size")
    if declared and declared < highest:
        raise ConversionRefused(
            f"config.json declares vocab_size {declared} but the tokenizer reaches id {highest - 1}; "
            "one of the two is wrong and the token table would be truncated"
        )
    return declared or highest


def _looks_special(token: str) -> bool:
    """Upstream's heuristic for a token an added-token entry failed to mark as special."""
    return (
        token in ("<pad>", "<mask>", "<2mass>", "[@BOS@]")
        or (token.startswith("<|") and token.endswith("|>"))
        or (token.startswith("<｜") and token.endswith("｜>"))
        or (token.startswith("<unused") and token.endswith(">"))
    )


def _merge_entries(merges: Sequence[Any]) -> list[str]:
    """The merges as GGUF strings, escaping a space inside a part as upstream escapes it."""
    pairs = [pair for pair in merges if isinstance(pair, list)]
    if any(" " in part for pair in pairs for part in pair):
        return [
            " ".join("".join(chr(ord(c) + 256) if c == " " else c for c in part) for part in pair)
            for pair in pairs
        ]
    return [" ".join(str(part) for part in pair) for pair in pairs] + [
        str(pair) for pair in merges if not isinstance(pair, list)
    ]


def special_token_ids(
    family: Family, tokenizer_config: Mapping[str, Any], added: Mapping[int, Any],
    config: Mapping[str, Any],
) -> dict[str, int]:
    """The special-token ids, resolved the way llama.cpp's `SpecialVocab` resolves them."""
    by_content = {str(entry.get("content")): index for index, entry in added.items()}
    ids: dict[str, int] = {}
    for token_type in family.special_tokens:
        entry = tokenizer_config.get(f"{token_type}_token")
        content = entry if isinstance(entry, str) else None
        if content is None and isinstance(entry, dict):
            content = entry.get("content")
        if isinstance(content, str) and content in by_content:
            ids[token_type] = by_content[content]
    text = text_config(config)
    for token_type in family.special_tokens:
        if token_type in ids:
            continue
        value = config.get(f"{token_type}_token_id")
        if value is None and "text_config" in config:
            value = text.get(f"{token_type}_token_id")
        if isinstance(value, int) and not isinstance(value, bool):
            ids[token_type] = value
    return ids


def set_vocab(writer: Any, tree: Path, family: Family, config: Mapping[str, Any]) -> None:
    """The gpt2-style vocabulary, read from tokenizer.json rather than a tokenizer library."""
    from gguf import TokenType

    tokenizer = _read_json(tree / "tokenizer.json")
    model = tokenizer.get("model") or {}
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ConversionRefused("tokenizer.json has no model.vocab")
    added = {
        _optional_int(entry.get("id"), -1, "added_tokens.id"): entry
        for entry in tokenizer.get("added_tokens") or []
    }
    added.pop(-1, None)
    kind = (tokenizer.get("normalizer") or {}).get("type")
    if kind not in (None, "NFC"):
        raise ConversionRefused(
            f"tokenizer.json normalizes with {kind!r}; upstream re-normalizes an added token through "
            "the tokenizer and this converter does not -- use llama.cpp's own converter"
        )
    for index, entry in added.items():
        content = str(entry.get("content", ""))
        if not entry.get("normalized") and unicodedata.normalize("NFC", content) != content:
            raise ConversionRefused(
                f"added token {index} ({content!r}) is not in NFC and is not marked normalized; "
                "upstream would re-normalize it -- use llama.cpp's own converter"
            )

    size = _vocab_size(vocab, added, text_config(config))
    tokens: list[str] = [f"[PAD{index}]" for index in range(size)]
    types: list[int] = [TokenType.UNUSED] * size
    for token, index in vocab.items():
        tokens[index] = token
        types[index] = TokenType.NORMAL
    for index, entry in added.items():
        token = str(entry.get("content", tokens[index]))
        if entry.get("special") or _looks_special(token):
            types[index] = TokenType.CONTROL
        else:
            types[index] = TokenType.USER_DEFINED
            token = token.replace(family.space_marker, " ")
        tokens[index] = token

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(family.arch)
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(_merge_entries(model.get("merges") or []))

    tokenizer_config_path = tree / "tokenizer_config.json"
    tokenizer_config = _read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    for token_type, token_id in special_token_ids(family, tokenizer_config, added, config).items():
        adder = getattr(writer, f"add_{token_type}_token_id", None)
        if adder is not None:  # `cls` has no handler in gguf, exactly as upstream
            adder(token_id)
    for token_type in family.special_tokens:
        value = tokenizer_config.get(f"add_{token_type}_token")
        adder = getattr(writer, f"add_add_{token_type}_token", None)
        if isinstance(value, bool) and adder is not None:
            adder(value)
    template = tokenizer_config.get("chat_template")
    if template is None:
        jinja = tree / "chat_template.jinja"
        template = jinja.read_text(encoding="utf-8") if jinja.is_file() else None
    if isinstance(template, (str, list)):
        writer.add_chat_template(template)


def _elements(shape: Sequence[int]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count




def reserve_output(out: Path) -> Path:
    """Claim a final path before conversion so a peer cannot publish over it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    reservation = out.with_name(f".{out.name}.apostate-reservation")
    if out.exists():
        raise ConversionRefused(f"{out} already exists; refusing to replace it")
    try:
        with reservation.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError as error:
        raise ConversionRefused(f"{out} is reserved by another conversion: {reservation}") from error
    return reservation
def convert(
    tree: Path | str,
    out: Path | str,
    *,
    pad_mlp_to: int | None = None,
    with_mtp: bool = False,
    no_pad: bool = False,
    dry_run: bool = False,
) -> Receipt:
    """Read the tree, write the GGUF, and report exactly what was done."""
    import time

    np, _torch, types, writer_type, safe_open = _deps()
    tree, out = Path(tree), Path(out)
    if not tree.is_dir():
        raise ConversionRefused(f"{tree} is not a directory")

    config = _read_json(tree / "config.json")
    family = family_for_config(config)
    arch = family.arch
    text = text_config(config)
    block_count = _need_int(text, "num_hidden_layers")
    source_width, target_width, pad_mode = resolve_widths(text, pad_mlp_to, no_pad)
    jobs, vision_skipped = plan_tensors(
        tree, family=family, text=text, block_count=block_count,
        hidden_size=_need_int(text, "hidden_size"), target_width=target_width, with_mtp=with_mtp,
    )
    if with_mtp and not any(job.source.startswith(family.draft_prefix) for job in jobs):
        raise ConversionRefused(f"--with-mtp but {tree} holds no {family.draft_prefix}* tensors")
    draft_layers = family.draft_layer_count(text, (job.source for job in jobs)) if with_mtp else 0
    receipt = Receipt(
        tree=str(tree), out=str(out), architecture=arch, block_count=block_count + draft_layers,
        source_width=source_width, target_width=target_width, padded=pad_mode, tensors=len(jobs),
        padded_tensors=sum(1 for job in jobs if job.pad_axis is not None),
        draft_tensors=sum(1 for job in jobs if job.source.startswith(family.draft_prefix)),
        vision_tensors_skipped=vision_skipped,
    )
    if pad_mode == "none" and source_width % QK_K:
        receipt.warnings.append(f"the tree's {unaligned_note(source_width)} (drop --no-pad to pad it)")
    if pad_mode != "none":
        receipt.warnings.append(
            f"MLP padded {source_width} -> {target_width} ({pad_mode}); the draft (MTP) block moves with "
            "the decoder, because llama.cpp sizes every block from feed_forward_length"
        )
    if dry_run:
        return receipt

    reservation = reserve_output(out)
    started = time.time()
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent)) / out.name
    writer = None
    try:
        generation_path = tree / "generation_config.json"
        writer = writer_type(str(stage), arch)
        set_metadata(
            writer, family=family, config=config,
            generation=_read_json(generation_path) if generation_path.is_file() else {},
            target_width=target_width, name=tree.name, draft_layers=draft_layers,
        )
        set_vocab(writer, tree, family, config)
        for job in jobs:
            dtype = np.float32 if job.f32 else np.uint16
            raw = types.F32 if job.f32 else types.BF16
            itemsize = 4 if job.f32 else 2
            writer.add_tensor_info(
                job.target, list(job.out_shape), dtype, _elements(job.out_shape) * itemsize, raw_dtype=raw
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for job in jobs:
            with safe_open(str(tree / job.shard), framework="pt") as handle:
                array = payload(handle, job, family)
            writer.write_tensor_data(array)
            receipt.bytes_written += int(array.nbytes)
        writer.close()
        os.replace(stage, out)
    except BaseException:
        try:
            if writer is not None:
                writer.close()
        finally:
            stage.unlink(missing_ok=True)
            try:
                stage.parent.rmdir()
            except OSError:
                pass
            reservation.unlink(missing_ok=True)
        raise
    reservation.unlink(missing_ok=True)
    receipt.seconds = time.time() - started
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tree", required=True, type=Path, help="the baked HF tree (safetensors + config)")
    parser.add_argument("--out", required=True, type=Path, help="the BF16 GGUF to create")
    parser.add_argument(
        "--pad-mlp-to", type=int, default=None,
        help="override the computed width; the default pads to the next multiple of 256 whenever the "
             "tree's own width is not one, so a 17409 bake lands at 17664 without anyone typing it",
    )
    parser.add_argument("--no-pad", action="store_true", help="keep the tree's own width (lineage artifact)")
    parser.add_argument("--with-mtp", action="store_true", help="include the draft (MTP) block")
    parser.add_argument("--receipt", type=Path, default=None, help="write the JSON receipt here")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.receipt is not None and not args.dry_run and args.receipt.exists():
        print(f"convert-tree: receipt already exists: {args.receipt}", file=sys.stderr)
        return 2
    try:
        receipt = convert(
            args.tree, args.out, pad_mlp_to=args.pad_mlp_to, with_mtp=args.with_mtp,
            no_pad=args.no_pad, dry_run=args.dry_run,
        )
    except ConversionRefused as refusal:
        print(f"convert-tree: {refusal}", file=sys.stderr)
        return 2
    document = receipt.document()
    document["dry_run"] = args.dry_run
    if args.receipt is not None and not args.dry_run:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    if not args.dry_run:
        print(
            f"convert-tree: {receipt.tensors} tensors, MLP {receipt.source_width} -> "
            f"{receipt.target_width} ({receipt.padded}), {receipt.seconds:.1f}s -> {receipt.out}"
        )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
