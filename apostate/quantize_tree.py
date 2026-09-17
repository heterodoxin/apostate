"""One call from a bake's HF tree to a quantized GGUF, with the MTP and vision options carried over.

`convert-tree`, `prepare-quant` and `quantize-gguf` each do one step and each refuse rather than guess,
which is right -- but an operator is then left gluing three receipts, three output names and two
decisions that no single command can make for them: whether the draft (MTP) head is quantized at a type
of its own, and whether a vision projector is exported and at what type. This is that chain in one
invocation:

  (a) convert the tree **once**, at the padded width, with this fork's own converter, writing the draft
      block only when the tree holds `mtp.*` and `--no-mtp` was not given;
  (b) resolve an importance matrix -- a local file, or the base model's published one -- and grow it
      against the converted trunk, because no published matrix carries statistics for the neuron an
      additive bake appended;
  (c) optionally export the vision projector with llama.cpp's own converter and quantize it;
  (d) quantize the trunk through llama-quantize with the grown matrix and the tensor-type pins;
  (e) print one receipt, and write it if asked.

No leg is re-implemented here: the conversion is `apostate.convert_tree`, the matrix work is
`apostate.prepare_quant`, the argv and the draft pin are `apostate.quantize_gguf`, and every layout fact is
`apostate.gguf_layout`, reached through those legs and never re-derived here. That is the point of the
command -- one place decides the order, and each step still refuses in its own name.

**MTP.** `--mtp-quantization TYPE` pins the draft block with `blk.<draft>.*=TYPE`, where the draft index
is `block_count - nextn_predict_layers` read out of the *converted file's* own header (a trunk of 64
decoder layers with one draft layer declares 65 blocks and puts the draft at 64 -- never a literal).
The pin is then checked against that file's tensor names and refused when it names none, because
llama-quantize matches `--tensor-type` patterns with `std::regex_search` and says **nothing** when a
pattern matches no tensor: measured against the build in `D:\\AI\\loaders\\llamacpp`
(`0.4.0-dev, build 10845, commit dbeb37548`), a recipe entry matching nothing exits 0, prints no
warning, and quantizes every tensor as if it had not been passed. A pin landing one block past the draft
would therefore ship a draft block at the base type in silence.

The derived pin is composed *after* any explicit `--tensor-type`, because llama-quantize applies the
first pattern that matches a tensor name and then stops: an explicit pin for a tensor inside the block is
the narrower, deliberate statement and wins, while the derived pin still covers the block's other
tensors -- a default, not an override. The receipt records the pins in the order they were sent, each
with the flag that asked for it, so a reader can see which one won. llama.cpp stores every 1-D tensor --
the block's norms -- as F32 whatever the pin says; that is expected and not a bug.

This fork has no donor-head graft: the converter writes the tree's *own* `mtp.*` head, so there is no
`--mtp-source`, and a draft head taken from a different checkpoint is a separate feature rather than
something quietly missing here.

**Vision.** `--export-mmproj` needs llama.cpp's `convert_hf_to_gguf.py`, resolved from
`--llama-cpp-source`, else `APOSTATE_LLAMA_CPP_SOURCE` (the checkout that holds it, for a machine that
should not have llama.cpp on PATH), else PATH, and the projector source (the tree, else `--mmproj-source`)
must index a vision tower before any conversion work. `F16` exports the projector as it stands; `Q8_0`
reproduces the tooling repository's hybrid rule exactly (2-D weights whose `ne[0]` is a multiple of 32 ->
`Q8_0`, everything else left at its source type) and quantizes *from* the F16 export, which is kept only
with `--keep-intermediate`. The receipt records whether llama-quantize read the recipe as a file or as
flags, and which mechanism named each of the two llama.cpp tools.

**Paths.** The receipt names artifacts by basename, because it is the document that leaves the machine;
the argv is the one place a local path is needed to reproduce a run. The two deliberate exceptions are the
resolved llama.cpp tools (`mmproj.converter`, `quantize.quantizer`): there the absolute path *is* the
fact, because "which binary ran" is provenance this project names, and a machine whose profile still
exports an old build must be visible in the artifact rather than merely suspected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import convert_tree, prepare_quant, quantize_gguf
# `_need_int` is the shared config reader the registry hands out for exactly this: a command that is not
# the engine still has to read a tree's own geometry the same way the engine does.
from .conversion import ConversionRefused, _need_int, family_for_config

SCHEMA = "apostate.tree-quantization.v1"

#: llama.cpp's own converter: the one file a `--llama-cpp-source` checkout -- or an
#: `APOSTATE_LLAMA_CPP_SOURCE` -- must hold. Named once so the flag's help, the refusal and the PATH
#: lookup all speak about the same script.
CONVERTER = "convert_hf_to_gguf.py"

#: The published-imatrix cache `prepare-quant` uses; one cache, so a chain and a one-off run share it.
IMATRIX_CACHE = Path("~/.cache/apostate/imatrix")

#: The vision spellings the tooling repository's preflight recognises. A tower is what makes a projector
#: source a projector source, and there is no point converting one that has none.
VISION_MARKERS = (
    "model.visual.",
    "model.vision_tower.",
    "vision_tower.",
    "vision_model.",
    "multi_modal_projector.",
)

PROJECTOR_QUANTIZATIONS = ("F16", "Q8_0")


class TreeQuantizationRefused(RuntimeError):
    """The chain cannot be run safely. Every leg's refusal reaches the operator as this command's own."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TreeQuantizationRefused(f"{path} is missing") from error
    except (OSError, json.JSONDecodeError) as error:
        raise TreeQuantizationRefused(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise TreeQuantizationRefused(f"{path} does not contain a JSON object")
    return document


def _basename(value: Path | str) -> str:
    return Path(value).name


def tree_facts(tree: Path) -> tuple[Any, dict[str, str]]:
    """The family that owns the tree, and its weight map -- read before a byte is converted.

    The index is also where the draft (MTP) head is visible: the converter writes the tree's own `mtp.*`
    tensors, so their presence here is what decides whether there is a draft block to quantize at all.
    """
    config = _read_json(tree / "config.json")
    declared = config.get("quantization_config")
    if isinstance(declared, dict) and declared:
        method = declared.get("quant_method") or declared.get("quantization_method") or "unknown"
        raise TreeQuantizationRefused(
            f"{tree} declares quantization_config ({method}): this chain converts a bf16/f16/f32 tree, "
            "and quantizing an already-quantized checkpoint re-quantizes from a lossy source. Point "
            "--tree at the bake's own tree"
        )
    family = family_for_config(config)
    index_path = tree / "model.safetensors.index.json"
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TreeQuantizationRefused(f"{index_path} has no usable weight_map")
    return family, {str(name): str(shard) for name, shard in weight_map.items()}


def quantization_kind(value: str | None) -> str | None:
    """A ggml type name for a flag that names one, refused when it could not be one.

    llama-quantize refuses an *unknown* type itself (`parse_ggml_type` prints `invalid ggml_type`), so
    the only job here is to catch a value that is not a type token at all before it is composed into an
    argv that also carries a `=` and a tensor pattern.
    """
    if value is None:
        return None
    kind = value.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", kind) is None:
        raise TreeQuantizationRefused(
            f"{value!r} is not a ggml type name: letters, digits and underscores, e.g. Q8_0 or BF16"
        )
    return kind.upper()


def predicted_draft_index(config: Mapping[str, Any]) -> int:
    """The draft index the conversion *will* write, for the `--dry-run` plan only.

    `draft_threshold` reads the real index out of the converted file as
    `block_count - nextn_predict_layers`, which is exactly the trunk's decoder-layer count; before that
    file exists the count can only come from the tree's config. A real run never trusts this number.
    """
    return _need_int(convert_tree.text_config(config), "num_hidden_layers")


def preflight_mmproj(source: Path) -> dict[str, Any]:
    """Refuse a projector source with no vision tower, before anything is converted.

    The index is the artifact: `config.json` claiming a vision component is a claim about tensors that
    may not be in the tree, while the weight map names the tensors themselves. The marker spellings are
    the tooling repository's preflight's, so a source that passes one passes the other.
    """
    if not source.is_dir():
        raise TreeQuantizationRefused(f"the projector source is not a directory: {source}")
    index_path = source / "model.safetensors.index.json"
    weight_map = _read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise TreeQuantizationRefused(f"{index_path} has no usable weight_map")
    vision = [str(name) for name in weight_map if any(marker in str(name) for marker in VISION_MARKERS)]
    if not vision:
        raise TreeQuantizationRefused(
            f"{source} holds no vision tower: its index names no tensor under any of "
            f"{', '.join(VISION_MARKERS)} -- point --mmproj-source at a compatible multimodal "
            "checkpoint, or drop --export-mmproj for a text-only bake"
        )
    missing = sorted(
        {str(weight_map[name]) for name in vision if not (source / str(weight_map[name])).is_file()}
    )
    if missing:
        raise TreeQuantizationRefused(
            "the projector source names shards that are not in it: " + ", ".join(missing)
        )
    return {
        "source": source.name,
        "tensors": len(vision),
        "shards": sorted({str(weight_map[name]) for name in vision}),
    }


def resolve_converter(llama_cpp_source: str) -> quantize_gguf.Resolved:
    """llama.cpp's own HF-to-GGUF converter: `--llama-cpp-source`, then `APOSTATE_LLAMA_CPP_SOURCE`, PATH.

    The fork bundles no llama.cpp, and the projector is the one leg that needs it: this project's own
    converter cannot write an mmproj. Resolution is `quantize_gguf.resolve_tool`'s, so both llama.cpp
    tools share one rule and one shape of provenance -- the difference between them is only that this
    one's flag and variable name a *checkout* and the tool is `convert_hf_to_gguf.py` inside it.
    """
    return quantize_gguf.resolve_tool(
        flag="--llama-cpp-source",
        explicit=llama_cpp_source,
        variable=quantize_gguf.LLAMA_CPP_SOURCE_VARIABLE,
        program=CONVERTER,
        usage="/path/to/llama.cpp (the checkout that holds convert_hf_to_gguf.py)",
        target="that checkout",
        member=CONVERTER,
        context=", so --export-mmproj cannot run",
        refusal=TreeQuantizationRefused,
    )


def projector_paths(out: Path, quantization: str) -> dict[str, Path]:
    """The tooling repository's projector names, rooted beside `--out`.

    `--out`'s stem normally ends in the quantization it was asked for, and the tooling's projector names
    carry the model alone -- so that suffix is dropped, and a trunk and its projector share one stem.
    """
    stem = out.stem
    suffix = "-" + quantization.lower()
    if stem.lower().endswith(suffix):
        stem = stem[: -len(suffix)]
    return {
        "f16": out.with_name(f"{stem}-mmproj-F16.gguf"),
        "hybrid": out.with_name(f"{stem}-mmproj-hybrid-Q8_0-F16.gguf"),
    }


def draft_pin(converted: Path, kind: str) -> str:
    """`blk.<draft>.*:TYPE` for a converted trunk, refused as this command's own error.

    The derivation is `quantize_gguf.mtp_pin`'s -- one implementation for both commands, and the one that
    checks the pin against the file's own tensor names. What this name adds is the refusal type: an
    operator who typed `quantize-tree` is not told that `quantize-gguf` refused.
    """
    return quantize_gguf.mtp_pin(converted, kind, TreeQuantizationRefused)


def pin_entries(derived: str | None, explicit: Sequence[str]) -> list[dict[str, str]]:
    """The tensor-type pins in the order llama-quantize will see them, each with the flag that asked.

    Order is precedence, not presentation: llama-quantize walks the patterns in argv order and stops at
    the first that matches a tensor name, so the *derived* draft pin goes last and an explicit
    `--tensor-type` for a tensor inside the draft block is the one that wins -- a computed, block-wide
    default must not silently beat a deliberate, narrower statement. The derived pin still covers every
    other tensor of the block, which is what makes it a default rather than an override.

    The names are spelled here so the receipt can say which pin came from where; the argv wants only the
    pins themselves.
    """
    entries = [{"pin": value, "origin": "--tensor-type"} for value in explicit]
    if derived is not None:
        entries.append({"pin": derived, "origin": "--mtp-quantization"})
    return entries


def resolve_matrix(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """The importance matrix to grow, and its provenance.

    A local file is taken as given and hashed. A publisher is discovered and cached by `prepare-quant`,
    whose provenance (`publisher`, `repository`, `filename`, `revision`, `sha256`, `origin`) is recorded
    as it returns it -- including the `null` revision of a cache hit, which cannot prove the commit it
    came from and says so rather than implying an audit that never happened.
    """
    if args.imatrix is not None:
        path = Path(args.imatrix).expanduser()
        if not path.is_file():
            raise TreeQuantizationRefused(f"imatrix not found: {path}")
        return path, {
            "publisher": "local",
            "filename": path.name,
            "revision": None,
            "sha256": prepare_quant.sha256(path),
            "origin": "argument",
        }
    return prepare_quant.resolve_published_imatrix(
        args.imatrix_source, args.base_model, IMATRIX_CACHE
    )


def _run(argv: Sequence[str]) -> None:
    """Run one external leg, echoing its argv; the receipt is the record, the console is the trail."""
    print("+ " + quantize_gguf.render_command(argv))
    completed = subprocess.run(list(argv))
    if completed.returncode != 0:
        raise TreeQuantizationRefused(
            f"{Path(argv[0]).name} exited {completed.returncode}; the receipt is not written"
        )


def _recipe_beside(projector: Path, recipe: Sequence[tuple[str, str]]) -> Path:
    """A recipe file in the projector's own directory, deleted once llama-quantize has read it.

    A dot-file beside the artifact rather than in `tempfile.gettempdir()`: it has to be on a filesystem
    llama-quantize can read, and a projector inside a 20 GB output directory may be on a different volume
    from `/tmp`.
    """
    handle, name = tempfile.mkstemp(prefix=f".{projector.stem}-hybrid-recipe.", suffix=".txt", dir=projector.parent)
    os.close(handle)
    return quantize_gguf.write_recipe(Path(name), recipe)


def export_projector(
    converter: quantize_gguf.Resolved,
    source: Path,
    paths: Mapping[str, Path],
    quantization: str,
    quantizer: quantize_gguf.Resolved,
    threads: int,
) -> tuple[dict[str, Any], list[Path]]:
    """Export the projector with llama.cpp's converter, then quantize it when asked.

    Returns the receipt leg and every path this leg created. The F16 export is always written first --
    even when only the hybrid is wanted, because the hybrid is quantized *from* it -- so a `Q8_0` run
    leaves an F16 projector behind too, which the caller keeps or deletes with its other intermediates.

    `converter` is the resolved tool, not a bare path: the leg records where the converter is *and* which
    mechanism named it, so an operator reading the receipt later can tell a `--dry-run` from a run against
    the wrong checkout.
    """
    f16 = paths["f16"]
    export_argv = [
        sys.executable, str(converter.path), str(source), "--mmproj", "--outtype", "f16",
        "--outfile", str(f16),
    ]
    created = [f16]
    try:
        _run(export_argv)
        leg: dict[str, Any] = {
            "source": source.name,
            "converter": converter.record(),
            "export_argv": export_argv,
            "f16": f16.name,
            "hybrid": None,
            "recipe": None,
        }
        if quantization == "Q8_0":
            hybrid = paths["hybrid"]
            recipe = quantize_gguf.hybrid_recipe(prepare_quant.open_gguf(f16))
            recipe_file = (
                _recipe_beside(f16, recipe)
                if quantize_gguf.supports_tensor_type_file(quantizer.path)
                else None
            )
            argv = quantize_gguf.projector_command(
                quantizer.path, f16, hybrid, "Q8_0", recipe, threads, recipe_file=recipe_file
            )
            try:
                _run(argv)
            finally:
                if recipe_file is not None:
                    recipe_file.unlink(missing_ok=True)
            created.append(hybrid)
            leg["hybrid"] = hybrid.name
            leg["quantize_argv"] = argv
            leg["recipe"] = {
                "mechanism": "tensor-type-file" if recipe_file is not None else "tensor-type",
                "entries": len(recipe),
                "q8_0": sum(1 for _name, kind in recipe if kind == "q8_0"),
            }
    except BaseException:
        # A partial projector is worse than none: the no-overwrite rule would block the identical retry.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return leg, created


def _check_receipt_path(receipt: Path, artifacts: Sequence[Path]) -> None:
    """A receipt is metadata: it may never land on an input or on an artifact of this run."""
    resolved = receipt.expanduser().resolve()
    for other in artifacts:
        if other.expanduser().resolve() == resolved:
            raise TreeQuantizationRefused(f"the receipt path collides with {other.name}")
    if prepare_quant.entry_exists(receipt):
        raise TreeQuantizationRefused(f"receipt already exists: {receipt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apostate quantize-tree",
        description=(
            "Convert a bake's HF tree to GGUF, grow the base model's importance matrix against it, "
            "export the vision projector if asked, and quantize the trunk -- one receipt for the chain."
        ),
    )
    parser.add_argument("--tree", type=Path, required=True, help="the bake's HF tree (safetensors + config)")
    parser.add_argument("--out", type=Path, required=True, help="the quantized GGUF to create")
    parser.add_argument("--quantization", required=True, help="llama.cpp quantization type, e.g. Q4_K_M")
    matrix_source = parser.add_mutually_exclusive_group()
    matrix_source.add_argument("--imatrix", type=Path, help="local importance matrix to grow")
    matrix_source.add_argument(
        "--imatrix-source",
        choices=("mradermacher", "bartowski"),
        help="discover and cache the base model's published imatrix",
    )
    parser.add_argument("--base-model", help="owner/model coordinates used with --imatrix-source")
    parser.add_argument(
        "--imatrix-out",
        type=Path,
        help="grown imatrix to create (default: sibling of --out named <out-stem>.imatrix.gguf)",
    )
    parser.add_argument(
        "--no-imatrix",
        action="store_true",
        help="quantize with no importance matrix at all, saying so in the receipt. Refused by default: "
             "a static build shipped under a name that implies an imatrix is the silent downgrade this "
             "command exists to prevent",
    )
    parser.add_argument(
        "--refuse-unweighted",
        action="store_true",
        help="refuse when the matrix has no statistics for draft MLP tensors -- pass it when the matrix "
             "was collected from this model and should have covered the draft head",
    )
    parser.add_argument(
        "--mtp-quantization",
        help="ggml type for the draft (MTP) block, e.g. Q8_0; pinned as blk.<draft>.* and checked "
             "against the converted file. Omitted: the block keeps the trunk's quantization",
    )
    parser.add_argument("--no-mtp", action="store_true", help="convert the trunk only, dropping the draft block")
    parser.add_argument("--export-mmproj", action="store_true", help="also export the vision projector")
    parser.add_argument(
        "--mmproj-source",
        type=Path,
        help="HF checkpoint the projector is exported from (default: --tree)",
    )
    parser.add_argument(
        "--mmproj-quantization",
        choices=PROJECTOR_QUANTIZATIONS,
        default="F16",
        help="projector quantization: F16 exports it as converted, Q8_0 adds the tooling repository's "
             "hybrid (2-D weights whose ne[0] is a multiple of 32 -> Q8_0, the rest at their source type)",
    )
    parser.add_argument(
        "--quantizer",
        default="",
        help="external llama-quantize path; overrides APOSTATE_LLAMA_QUANTIZE, and both override PATH. An "
             "empty variable counts as unset; a wrong one is refused by name, not fallen through to PATH",
    )
    parser.add_argument(
        "--llama-cpp-source",
        default="",
        help="llama.cpp checkout holding convert_hf_to_gguf.py; overrides APOSTATE_LLAMA_CPP_SOURCE, and "
             "both override PATH. An empty variable counts as unset; a checkout that holds no converter "
             "is refused by name, not fallen through to PATH",
    )
    parser.add_argument("--threads", type=int, default=0, help="llama-quantize threads; 0 lets llama.cpp choose")
    parser.add_argument(
        "--tensor-type",
        action="append",
        default=[],
        help="extra name:TYPE pin for the trunk quantize; repeatable",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep the BF16 trunk and the grown matrix (and the F16 projector a Q8_0 export quantized "
             "from) instead of deleting them",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the whole plan; write nothing")
    parser.add_argument("--receipt", type=Path, help="write the JSON receipt here (never overwriting one)")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Decide, then do: every refusal happens before the first byte is converted."""
    import time

    tree, out = Path(args.tree), Path(args.out)
    if not tree.is_dir():
        raise TreeQuantizationRefused(f"{tree} is not a directory: --tree takes a bake's HF tree")
    if prepare_quant.entry_exists(out):
        raise TreeQuantizationRefused(f"refusing to overwrite output: {out}")
    family, weight_map = tree_facts(tree)
    draft_sources = sorted(name for name in weight_map if name.startswith(family.draft_prefix))
    if args.mtp_quantization is not None and args.no_mtp:
        raise TreeQuantizationRefused(
            "--mtp-quantization with --no-mtp: --no-mtp drops the draft block, so there would be nothing "
            "left to pin; drop one of the two"
        )
    if args.mtp_quantization is not None and not draft_sources:
        raise TreeQuantizationRefused(
            f"--mtp-quantization {args.mtp_quantization}, but {tree} holds no {family.draft_prefix}* "
            "tensors: there is no draft block to quantize"
        )
    mtp_kind = quantization_kind(args.mtp_quantization)
    with_mtp = bool(draft_sources) and not args.no_mtp
    if not args.no_imatrix and args.imatrix is None and args.imatrix_source is None:
        raise TreeQuantizationRefused(
            "no importance matrix to grow: pass --imatrix PATH, or --imatrix-source mradermacher with "
            "--base-model OWNER/MODEL, or --no-imatrix to ship a static build and say so"
        )

    trunk = out.with_name(f"{out.stem}.bf16.gguf")
    matrix_out = (
        Path(args.imatrix_out).expanduser()
        if args.imatrix_out is not None
        else out.with_name(f"{out.stem}.imatrix.gguf")
    )
    projectors = projector_paths(out, args.quantization) if args.export_mmproj else {}
    wanted = (
        ("BF16 trunk", trunk),
        ("grown imatrix", matrix_out),
        ("F16 projector", projectors.get("f16")),
        ("hybrid projector", projectors.get("hybrid") if args.mmproj_quantization == "Q8_0" else None),
    )
    for label, path in wanted:
        if path is not None and prepare_quant.entry_exists(path):
            raise TreeQuantizationRefused(f"refusing to overwrite the {label}: {path}")

    quantizer = quantize_gguf.resolve_quantizer(args.quantizer)
    converter = resolve_converter(args.llama_cpp_source) if args.export_mmproj else None
    projector_source = Path(args.mmproj_source).expanduser() if args.mmproj_source else tree
    projector_preflight = preflight_mmproj(projector_source) if args.export_mmproj else None
    resolved, provenance = (None, None) if args.no_imatrix else resolve_matrix(args)
    if args.receipt is not None:
        _check_receipt_path(
            args.receipt,
            [tree, out, trunk, matrix_out, *projectors.values(), *([resolved] if resolved else [])],
        )

    started = time.time()
    conversion = convert_tree.convert(tree, trunk, with_mtp=with_mtp, dry_run=args.dry_run)
    if args.dry_run:
        return _plan(
            args=args, tree=tree, out=out, trunk=trunk, matrix_out=matrix_out, projectors=projectors,
            conversion=conversion, draft_sources=draft_sources, mtp_kind=mtp_kind, with_mtp=with_mtp,
            provenance=provenance, projector_preflight=projector_preflight, projector_source=projector_source,
            quantizer=quantizer, converter=converter, seconds=time.time() - started,
        )

    created: list[Path] = [trunk]
    intermediates: list[Path] = []
    try:
        growth = None
        if resolved is not None:
            growth = prepare_quant.adapt_matrix(
                resolved, trunk, matrix_out, refuse_unweighted=args.refuse_unweighted
            )
            created.append(matrix_out)
            intermediates.append(matrix_out)
        projector_leg = None
        if args.export_mmproj:
            projector_leg, projector_created = export_projector(
                converter, projector_source, projectors, args.mmproj_quantization, quantizer, args.threads
            )
            created.extend(projector_created)
            # The F16 projector is a deliverable only when F16 was asked for; a Q8_0 run quantizes *from*
            # it, which is exactly the intermediate the tooling repository leaves in its work directory.
            if args.mmproj_quantization == "Q8_0":
                intermediates.append(projectors["f16"])
        intermediates.append(trunk)

        derived = draft_pin(trunk, mtp_kind) if mtp_kind is not None else None
        pins = pin_entries(derived, args.tensor_type)
        argv = quantize_gguf.command(
            quantizer.path, trunk, out, args.quantization, matrix_out if resolved is not None else None,
            [entry["pin"] for entry in pins], args.threads,
        )
        _run(argv)
        created.append(out)

        unweighted = (
            growth["unweighted_draft_tensors"]
            if growth is not None
            else prepare_quant.unweighted_draft_mlp_tensors(trunk, {})
        )
        document = {
            "schema": SCHEMA,
            "dry_run": False,
            "tree": _tree_reference(tree, len(draft_sources)),
            "width": _width_decision(conversion),
            "conversion": _child(conversion.document(), ("tree", "out")),
            "mtp": {
                "converted": with_mtp,
                "tensors": len(draft_sources),
                "quantization": mtp_kind,
                "pin": derived,
            },
            "imatrix": _matrix_reference(matrix_out, provenance, growth),
            "mmproj": projector_leg,
            "quantize": {
                "argv": argv,
                "pins": pins,
                "quantizer": quantizer.record(),
                "output": out.name,
                "threads": args.threads,
            },
            "unweighted_draft_tensors": unweighted,
            "intermediates": {
                "kept": bool(args.keep_intermediate),
                "paths": sorted(_basename(path) for path in intermediates),
            },
            "output": {
                "path": out.name,
                "sha256": prepare_quant.sha256(out),
                "bytes": out.stat().st_size,
            },
            "seconds": round(time.time() - started, 2),
        }
        if args.receipt is not None:
            prepare_quant.write_receipt(args.receipt, document)
    except BaseException:
        # A half-built chain is worse than none: the no-overwrite rule would then block the identical
        # retry that would have produced all of it.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    if not args.keep_intermediate:
        for path in intermediates:
            path.unlink(missing_ok=True)
    return document


def _tree_reference(tree: Path, draft_tensors: int) -> dict[str, Any]:
    index = tree / "model.safetensors.index.json"
    return {
        "path": tree.name,
        "index": index.name,
        "index_sha256": prepare_quant.sha256(index),
        "draft_tensors": draft_tensors,
    }


def _width_decision(conversion: convert_tree.Receipt) -> dict[str, Any]:
    return {
        "source": conversion.source_width,
        "target": conversion.target_width,
        "padding": conversion.padded,
        "note": conversion.warnings[0] if conversion.warnings else None,
    }


def _child(document: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """A child command's receipt with its own path fields reduced to basenames.

    The legs record absolute paths because each of them may be run alone; this document is the one that
    leaves the machine, so it keeps the facts and drops the local layout.
    """
    reduced = dict(document)
    for key in paths:
        if reduced.get(key):
            reduced[key] = _basename(str(reduced[key]))
    return reduced


def _matrix_reference(
    grown: Path | None, provenance: Mapping[str, Any] | None, growth: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The matrix leg: where it came from, where it was written, and what growing it did.

    `None` provenance is `--no-imatrix`, which is a fact about the run rather than a missing field.
    """
    if provenance is None:
        return None
    return {
        "source": dict(provenance),
        "grown": None if (grown is None or growth is None) else _basename(grown),
        "growth": _child(growth, ("source", "target", "out")) if growth is not None else None,
    }


def _plan(
    *,
    args: argparse.Namespace,
    tree: Path,
    out: Path,
    trunk: Path,
    matrix_out: Path,
    projectors: Mapping[str, Path],
    conversion: convert_tree.Receipt,
    draft_sources: Sequence[str],
    mtp_kind: str | None,
    with_mtp: bool,
    provenance: Mapping[str, Any] | None,
    projector_preflight: Mapping[str, Any] | None,
    projector_source: Path,
    quantizer: quantize_gguf.Resolved,
    converter: quantize_gguf.Resolved | None,
    seconds: float,
) -> dict[str, Any]:
    """The whole plan, spelled the way the real run would spell it, and nothing written.

    Every decision the run makes is here: which width the conversion will pad to, which draft index the
    pin will name, which matrix was resolved and from which revision, how the projector recipe will be
    fed to llama-quantize, and the exact trunk argv. What a plan cannot know without doing the work --
    the matrix growth against a trunk that does not exist yet, and the file's own tensor names -- is
    named as such instead of guessed.
    """
    index = predicted_draft_index(_read_json(tree / "config.json")) if mtp_kind is not None else None
    derived = f"blk.{index}.*:{mtp_kind}" if index is not None else None
    pins = pin_entries(derived, args.tensor_type)
    argv = quantize_gguf.command(
        quantizer.path, trunk, out, args.quantization, matrix_out if provenance is not None else None,
        [entry["pin"] for entry in pins], args.threads,
    )
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "tree": _tree_reference(tree, len(draft_sources)),
        "width": _width_decision(conversion),
        "conversion": _child(conversion.document(), ("tree", "out")),
        "mtp": {
            "converted": with_mtp,
            "tensors": len(draft_sources),
            "quantization": mtp_kind,
            "draft_index": index,
            "pin": derived,
            "checked_against": "the converted file's block_count and nextn_predict_layers at run time",
        },
        "imatrix": (
            {
                "source": dict(provenance),
                "grown": _basename(matrix_out),
                "growth": "computed against the converted trunk; a plan has no trunk to grow against yet",
            }
            if provenance is not None
            else None
        ),
        "mmproj": (
            {
                "source": projector_preflight["source"],
                "tensors": projector_preflight["tensors"],
                "converter": converter.record(),
                "export_argv": [
                    sys.executable, str(converter.path), str(projector_source), "--mmproj",
                    "--outtype", "f16", "--outfile", str(projectors["f16"]),
                ],
                "f16": _basename(projectors["f16"]),
                "hybrid": _basename(projectors["hybrid"]) if args.mmproj_quantization == "Q8_0" else None,
                "recipe": (
                    "one name=type line per projector tensor, 2-D with ne[0] a multiple of 32 -> Q8_0, "
                    "the rest at their source type; fed as --tensor-type-file when this build documents "
                    f"it, else as repeated --tensor-type flags ({_basename(quantizer.path)})"
                )
                if args.mmproj_quantization == "Q8_0"
                else None,
            }
            if projector_preflight is not None
            else None
        ),
        "quantize": {
            "argv": argv,
            "pins": pins,
            "quantizer": quantizer.record(),
            "output": out.name,
            "threads": args.threads,
        },
        # A fact about a converted trunk, and a plan has none: the trunk's own tensor names are what say
        # which draft MLP weights the matrix does not cover.
        "unweighted_draft_tensors": None,
        "intermediates": {
            "kept": bool(args.keep_intermediate),
            "paths": sorted({_basename(trunk), _basename(matrix_out)}),
        },
        "output": {"path": out.name, "sha256": None, "bytes": None},
        "seconds": round(seconds, 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.imatrix_source is not None and not args.base_model:
        parser.error("--base-model is required with --imatrix-source")
    if args.no_imatrix and (args.imatrix is not None or args.imatrix_source is not None):
        parser.error("--no-imatrix cannot be combined with --imatrix or --imatrix-source")
    if (args.mmproj_source is not None or args.mmproj_quantization != "F16") and not args.export_mmproj:
        parser.error("--mmproj-source and --mmproj-quantization require --export-mmproj")
    try:
        document = run(args)
    except (
        TreeQuantizationRefused,
        ConversionRefused,
        prepare_quant.PreparationRefused,
        quantize_gguf.QuantizationRefused,
    ) as error:
        print(f"quantize-tree: refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
