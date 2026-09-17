# Apostate CLI entrypoint

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import shlex
from pathlib import Path

from . import discover

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

HELP = """\
apostate            interactive menu (default)
apostate setup      install python deps, check gpu
apostate doctor     verify the GPU can run a kernel before any big load (cuda/rocm)
apostate ablate --model M --out D   build a diode checkpoint (default method)
apostate diode  --model M --out D   build a conditional-abliteration (diode) checkpoint
apostate ccv   --model M --out D   build a predictive/contrastive co-vector checkpoint
apostate kcrn   --model M --out D   build a projected fixed-weight KCRN checkpoint
apostate ticv   --model D --out D2  bake soft-deflection removal (TICV) into an abliterated checkpoint
apostate finetune --model M --out D [--data path.jsonl] [--steps N]  QLoRA finetune (train alias)
apostate talk   --model D [--backend vllm]   chat
apostate test   --model D --base M  benchmark (--suite humaneval,mbpp,gsm8k,refusal,all)
apostate prepare-quant --model M.gguf --out-model M-aligned.gguf [--imatrix I --out-imatrix O]
apostate convert-tree --tree D --out M.gguf [--with-mtp] [--pad-mlp-to N] [--no-pad]
apostate quantize-gguf --source M.gguf --out Q.gguf --quantization Q4_K_M [--quantizer PATH]
apostate quantize-tree --tree D --out Q.gguf --quantization Q4_K_M [--imatrix I | --imatrix-source mradermacher --base-model O/M] [--mtp-quantization T] [--export-mmproj [--mmproj-source D] [--mmproj-quantization F16|Q8_0]] [--quantizer PATH]
apostate list       show cached hf models + local checkpoints
"""


def run_module(mod_args, label=None) -> int:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
    if label:
        env["APOSTATE_COMMAND"] = label
    return subprocess.run([sys.executable, *mod_args], env=env).returncode


def _flag(args, name, default=None):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def _strip(args, names):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in names:
            skip = True
        else:
            out.append(a)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[0] if argv else "tui"
    args = argv[1:]

    if cmd in ("-h", "--help"):
        print(HELP)
        return 0

    if cmd == "tui":
        from .tui import run as tui_run
        return tui_run()

    if cmd == "setup":
        from .setup_wizard import main as setup_main
        return setup_main(args)

    if cmd == "doctor":
        return run_module(["-m", "apostate.doctor", *args], "apostate doctor")

    if cmd in ("ablate", "diode", "boost", "ccv", "kcrn"):
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", _flag(args, "--output-dir", "out"))
        rest = _strip(args, ["--model", "--out", "--output-dir"])
        label = "apostate " + cmd + " --model " + shlex.quote(model) + " --out " + shlex.quote(out)
        if rest:
            label += " " + " ".join(shlex.quote(x) for x in rest)
        method = {"ablate": "diode", "diode": "diode", "kcrn": "kcrn", "ccv": "ccv"}.get(cmd, "legacy")
        return run_module(
            ["-m", "apostate.cli", "--method", method, "--optimize", "--model", model, "--output-dir", out, *rest],
            label)

    if cmd == "turbo":
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", "out")
        label = f"apostate turbo --model {model} --out {out}"
        print("step 1: finetune")
        run_module(["-m", "apostate.finetune", "--model", model, "--out", out + "_ft"], label)
        print("step 2: abliterate")
        run_module(["-m", "apostate.cli", "--method", "legacy", "--optimize", "--model", out + "_ft", "--output-dir", out], label)
        print("step 3: cleanup")
        shutil.rmtree(out + "_ft", ignore_errors=True)
        print("step 4: verify")
        run_module(["-m", "apostate.benchcode", "--model", out, "--base", model], label)
        return 0

    if cmd == "ticv":
        return run_module(["-m", "apostate.ticv", *args], f"apostate ticv {' '.join(args)}".strip())
    if cmd == "test":
        return run_module(["-m", "apostate.benchcode", *args], f"apostate test {' '.join(args)}".strip())
    if cmd == "talk":
        return run_module(["-m", "apostate.chat", *args], f"apostate talk {' '.join(args)}".strip())
    if cmd == "prepare-quant":
        return run_module(["-m", "apostate.prepare_quant", *args], f"apostate prepare-quant {' '.join(args)}".strip())
    if cmd == "convert-tree":
        # The bake's own output is an HF tree, not a GGUF: this is the step that turns it into one, with
        # the appended neuron's width repaired in the same pass so `prepare-quant` has nothing left to fix.
        return run_module(["-m", "apostate.convert_tree", *args], f"apostate convert-tree {' '.join(args)}".strip())
    if cmd == "quantize-gguf":
        return run_module(
            ["-m", "apostate.quantize_gguf", *args], f"apostate quantize-gguf {' '.join(args)}".strip()
        )
    if cmd == "quantize-tree":
        # The whole chain -- convert, grow the matrix, export the projector, quantize -- so an operator
        # does not glue three receipts together and, on the way, lose the MTP and projector options the
        # tooling repo already had.
        return run_module(
            ["-m", "apostate.quantize_tree", *args], f"apostate quantize-tree {' '.join(args)}".strip()
        )
    if cmd == "quantize":
        return run_module(["-m", "apostate.quant", *args])
    if cmd in ("train", "finetune"):
        return run_module(["-m", "apostate.finetune", *args])

    if cmd == "list":
        print("hf cache:")
        for m in discover.hf_models():
            print("  " + m)
        print("\ncheckpoints:")
        for c in discover.checkpoints():
            print("  " + c)
        return 0

    print("unknown command: " + cmd, file=sys.stderr)
    print(HELP, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
