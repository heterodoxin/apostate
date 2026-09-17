"""Quantize a prepared GGUF through an external llama-quantize executable.

This module never bundles llama.cpp. Apostate owns model and imatrix preparation;
llama.cpp owns K-quant and imatrix-weighted quantization.

It also owns the two argv shapes that are llama.cpp's rather than this project's: the trunk's, where a
tensor-type pin is one `--tensor-type name=TYPE`, and the vision projector's, where the same recipe is a
whole file of `name=type` lines. Both are spelled here so `apostate quantize-tree` composes a call
instead of a copy, and so the one place that decides which form a build accepts is next to the code that
has to live with the answer.

**MTP.** The one thing this module reads out of a source file is the *index* of its draft (MTP) block:
`--mtp-quantization TYPE` resolves `blk.<block_count - nextn_predict_layers>.*=TYPE` from the file's own
header through `apostate.gguf_layout`, and refuses when that pin names no tensor in it. The index is a
property of the artifact, so it is not something an operator can be asked to know per model -- the
hand-written `blk.64.*` that a recipe carries is right for one model and silently wrong for the next, and
llama-quantize accepts a pattern that matches nothing without a word. Nothing else is discovered: no
matrix is resolved here, and no conversion is performed.

**Pin precedence**, read out of llama.cpp rather than assumed, because the argv order of the pins is a
decision and not a detail. `parse_tensor_type` (`tools/quantize/quantize.cpp`) splits an entry at the
first `=`, which is why the composed argv spells a pin `name=TYPE` (the flag's own `name:TYPE` is this
project's shorthand), lowercases the name, and refuses an unknown ggml type itself. `quantize_state_impl`
(`src/llama-quant.cpp`) then compiles every pattern once and, per tensor, walks them in argv order with
`std::regex_search`, applying the first one that matches and `break`ing. The first entry that matches a
name is therefore the one that wins, so any explicit `--tensor-type` is composed *before* a broader
pattern that would also have covered it, and a pattern that matches nothing at all is silent: measured
against the build in `D:\\AI\\loaders\\llamacpp` (`0.4.0-dev`, build 10845, commit dbeb37548), an entry
matching no tensor exits 0, prints no warning, and quantizes every tensor as if it had not been passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn, Sequence

from . import gguf_layout, prepare_quant


class QuantizationRefused(RuntimeError):
    """The requested external quantization cannot be run safely."""


def resolve_quantizer(explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise QuantizationRefused(f"llama-quantize not found: {path}")
        return path
    found = shutil.which("llama-quantize")
    if found:
        return Path(found)
    raise QuantizationRefused(
        "llama-quantize was not found on PATH; install llama.cpp or pass --quantizer /path/to/llama-quantize"
    )


def command(
    quantizer: Path,
    source: Path,
    output: Path,
    quantization: str,
    imatrix: Path | None,
    tensor_types: Sequence[str],
    threads: int,
) -> list[str]:
    argv = [str(quantizer)]
    if imatrix is not None:
        argv.extend(["--imatrix", str(imatrix)])
    for value in tensor_types:
        name, separator, kind = value.partition(":")
        if not separator or not name or not kind:
            raise QuantizationRefused(f"--tensor-type requires name:TYPE, got {value!r}")
        argv.extend(["--tensor-type", f"{name}={kind.upper()}"])
    return [*argv, str(source), str(output), quantization.lower(), str(threads)]


def render_command(argv: Sequence[str]) -> str:
    """A lossless, non-shell receipt for an external invocation."""
    return json.dumps(list(argv))


def _header_integer(refusal: type[Exception]) -> Callable[[Any, str], int]:
    """`gguf_layout`'s integer reader, refused through the calling command's own exception type.

    Zero is accepted where a positive would seem right: a file may declare `nextn_predict_layers` as 0,
    which `draft_threshold` reads as "no draft head". A negative is a corrupt header and is not.
    """

    def integer(value: Any, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise refusal(f"{label} is not an integer: {value!r}") from error
        if parsed < 0:
            raise refusal(f"{label} is negative: {parsed}")
        return parsed

    return integer


def mtp_pin(
    source: Path | str, quantization: str, refusal: type[Exception] = QuantizationRefused
) -> str:
    """`blk.<draft>.*:TYPE` for `source`, read out of its own header, refused when it names no tensor.

    The draft index is a property of the artifact -- `block_count - nextn_predict_layers` -- and not
    something an operator can be expected to know per model. `gguf_layout.draft_threshold` is the one rule
    for it, so a family that counts its draft layer as a block and one that leaves the count to the trunk
    both land on the same index.

    The check is the point of the function. llama-quantize matches a `--tensor-type` pattern with
    `std::regex_search` and reports a pattern that matched no tensor with complete silence: measured
    against the build in `D:\\AI\\loaders\\llamacpp` (`0.4.0-dev`, build 10845), a recipe whose entries
    match nothing exits 0, prints no warning, and quantizes every tensor as if it had not been passed. A
    pin landing one block past the draft would therefore ship the draft block at the base type in silence.

    `refusal` is the calling command's own exception *type*: an operator who typed `quantize-tree` is not
    told that `quantize-gguf` refused. A type and not `gguf_layout`'s refusal callable, because building
    an exception does not raise it -- a class passed where a raiser is expected refuses nothing -- and
    this function is the last place to be silently permissive. The colon form is what `command` accepts;
    the `name=TYPE` spelling is the argv writer's business.
    """

    def refuse(message: str) -> NoReturn:
        """`gguf_layout`'s refusal callable, raising rather than constructing."""
        raise refusal(message)

    path = Path(source)
    try:
        reader = prepare_quant.open_gguf(path)
    except prepare_quant.PreparationRefused as error:
        # A source `is_file` yet unreadable as GGUF -- truncated, foreign, or read without gguf-py. The
        # operator typed *this* command, so the refusal is spelled in its name rather than surfacing
        # another module's exception type.
        refuse(str(error))
    threshold = gguf_layout.draft_threshold(
        prepare_quant.gguf_fields(reader), _header_integer(refusal), refuse
    )
    if threshold is None:
        refuse(
            f"{path.name} declares no block_count, so the draft (MTP) index cannot be read from it; "
            "quantize a file a converter wrote, or drop --mtp-quantization"
        )
    for tensor in reader.tensors:
        match = gguf_layout.BLOCK.match(str(tensor.name))
        if match is not None and int(match.group(1)) == threshold:
            return f"blk.{threshold}.*:{quantization}"
    refuse(
        f"--mtp-quantization {quantization} pins blk.{threshold}.*, but {path.name} has no tensor in "
        f"block {threshold}: the source holds no draft (MTP) block at that index, which is what a "
        "trunk-only file looks like -- and so does one whose draft block was dropped at conversion. "
        "llama-quantize would apply the pin to nothing and quantize as if it had not been passed, "
        "without a word. Quantize a source that carries the draft block, or drop --mtp-quantization"
    )


#: The block a projector weight's `ne[0]` must be a multiple of to be worth Q8_0. `GGUFReader` reports
#: `ne[0]` as `shape[0]`, so this reads exactly as the tooling repository's recipe writer spells it.
MMPROJ_BLOCK = 32

#: The source types the projector recipe may name. A projector converted from a bf16 checkpoint is
#: entirely one of these three; a tensor that is already quantized is left out of the recipe, because
#: naming it would ask llama-quantize to requantize it.
RECIPE_SOURCE_TYPES = ("F16", "BF16", "F32")


def hybrid_recipe(reader: Any, kind: str = "Q8_0") -> list[tuple[str, str]]:
    """The tooling repository's projector rule, as `(tensor, ggml type)` pairs in file order.

    `kind` for an F16/BF16 tensor that is 2-D with `shape[0]` a multiple of `MMPROJ_BLOCK`, otherwise the
    source type unchanged, and nothing at all for a tensor that is already quantized.

    F32 is deliberately not promoted to `kind`: `converter/make_mmproj_quant_recipe.py` selects only
    f16/bf16, and a projector that promoted the F32 norms would weigh more than the published one while
    claiming the same name. Reproducing the reference exactly is the point -- the hybrid name is a claim
    about which tensors were converted, and two answers to it make the name meaningless.
    """
    recipe: list[tuple[str, str]] = []
    for tensor in reader.tensors:
        source = str(tensor.tensor_type.name).upper()
        if source not in RECIPE_SOURCE_TYPES:
            continue
        shape = [int(value) for value in tensor.shape]
        if source in ("F16", "BF16") and len(shape) == 2 and shape[0] % MMPROJ_BLOCK == 0:
            recipe.append((str(tensor.name), kind.lower()))
        else:
            recipe.append((str(tensor.name), source.lower()))
    return recipe


def write_recipe(path: Path, recipe: Sequence[tuple[str, str]]) -> Path:
    """Write a recipe as one `name=type` per line, ASCII with LF endings.

    That is llama-quantize's own format -- `parse_tensor_type_file` reads whitespace-separated
    `name=type` tokens -- and fixing the bytes on one line each keeps a Windows run and a Linux run
    identical.
    """
    path.write_text(
        "".join(f"{name}={kind}\n" for name, kind in recipe), encoding="ascii", newline="\n"
    )
    return path


def projector_command(
    quantizer: Path,
    source: Path,
    output: Path,
    kind: str,
    recipe: Sequence[tuple[str, str]],
    threads: int,
    *,
    recipe_file: Path | None = None,
) -> list[str]:
    """The projector's argv: the trunk's `command`, but with a per-tensor recipe.

    `recipe_file` is passed when this build reads `--tensor-type-file`; the identical recipe is otherwise
    sent as repeated `--tensor-type` flags. Both fill the same list inside llama-quantize
    (`parse_tensor_type_file` calls `parse_tensor_type` per token), so the choice is argv length and not
    capability -- which is why the flags form is a safe fallback for any build.
    """
    argv = [str(quantizer)]
    if recipe_file is not None:
        argv.extend(["--tensor-type-file", str(recipe_file)])
    else:
        for name, tensor_kind in recipe:
            argv.extend(["--tensor-type", f"{name}={tensor_kind}"])
    return [*argv, str(source), str(output), kind.lower(), str(threads)]


def supports_tensor_type_file(quantizer: Path) -> bool:
    """Whether this llama-quantize build documents `--tensor-type-file`.

    Probed rather than assumed: apostate does not bundle llama.cpp, so the binary is whatever the machine
    has. A build without the flag still takes the same recipe as repeated `--tensor-type` flags, so a
    probe that cannot run at all -- a non-executable file, a foreign architecture, a binary that will not
    answer `--help` -- answers False and costs nothing.
    """
    try:
        probe = subprocess.run(
            [str(quantizer), "--help"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--tensor-type-file" in f"{probe.stdout}{probe.stderr}"



def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", required=True, type=Path, help="prepared BF16/F16/F32 GGUF")
    parser.add_argument("--out", required=True, type=Path, help="quantized GGUF to create")
    parser.add_argument("--quantization", required=True, help="llama.cpp quantization type, e.g. Q4_K_M")
    parser.add_argument("--imatrix", type=Path, help="compatible (possibly adapted) importance matrix")
    parser.add_argument("--tensor-type", action="append", default=[], help="name:TYPE override; repeatable")
    parser.add_argument(
        "--mtp-quantization",
        help="ggml type for the draft (MTP) block, e.g. Q8_0: the block index is read from --source's own "
             "block_count and nextn_predict_layers, and refused when the source holds no tensor there, so "
             "no hand-written layer index is needed. It is appended after --tensor-type, and llama-quantize "
             "applies the first pattern that matches a tensor name, so an explicit --tensor-type for a "
             "tensor in the draft block is the one that wins",
    )
    parser.add_argument("--threads", type=int, default=0, help="llama-quantize threads; 0 lets llama.cpp choose")
    parser.add_argument("--quantizer", default="", help="external llama-quantize path; default resolves PATH")
    parser.add_argument("--dry-run", action="store_true", help="print the command without invoking it")
    args = parser.parse_args(argv)
    try:
        if not args.source.is_file():
            raise QuantizationRefused(f"source GGUF not found: {args.source}")
        if args.out.exists():
            raise QuantizationRefused(f"refusing to overwrite output: {args.out}")
        if args.imatrix is not None and not args.imatrix.is_file():
            raise QuantizationRefused(f"imatrix not found: {args.imatrix}")
        quantizer = resolve_quantizer(args.quantizer)
        pins = list(args.tensor_type)
        if args.mtp_quantization is not None:
            # Appended, not prepended: llama-quantize walks the `--tensor-type` patterns in argv order and
            # stops at the first one that matches a tensor name, so an explicit pin stays the sharper
            # answer to a tensor the block-wide pin would also have covered.
            pins.append(mtp_pin(args.source, args.mtp_quantization))
        invocation = command(
            quantizer, args.source, args.out, args.quantization, args.imatrix, pins, args.threads
        )
        print("+ " + render_command(invocation))
        if args.dry_run:
            return 0
        return subprocess.run(invocation).returncode
    except QuantizationRefused as error:
        print(f"quantize-gguf: refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
