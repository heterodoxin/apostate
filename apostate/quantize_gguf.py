"""Quantize a prepared GGUF through an external llama-quantize executable.

This module never bundles llama.cpp. Apostate owns model and imatrix preparation;
llama.cpp owns K-quant and imatrix-weighted quantization.

It also owns the two argv shapes that are llama.cpp's rather than this project's: the trunk's, where a
tensor-type pin is one `--tensor-type name=TYPE`, and the vision projector's, where the same recipe is a
whole file of `name=type` lines. Both are spelled here so `apostate quantize-tree` composes a call
instead of a copy, and so the one place that decides which form a build accepts is next to the code that
has to live with the answer.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


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
        invocation = command(
            quantizer, args.source, args.out, args.quantization, args.imatrix, args.tensor_type, args.threads
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
