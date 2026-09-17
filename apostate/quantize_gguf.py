"""Quantize a prepared GGUF through an external llama-quantize executable.

This module never bundles llama.cpp. Apostate owns model and imatrix preparation;
llama.cpp owns K-quant and imatrix-weighted quantization.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import sys


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
