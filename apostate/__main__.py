"""apostate cli. no args opens the tui; subcommands run the engine."""

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
apostate ablate --model M --out D   remove refusals (--resume reuses activation cache)
apostate test   --model D --base M  benchmark (--suite humaneval,mbpp,gsm8k,refusal,all)
apostate talk   --model D [--backend vllm]   chat
apostate list       show cached hf models + local checkpoints
"""


def run_module(mod_args, label=None) -> int:
    """run a python -m subcommand with the engine on the path."""
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

    if cmd in ("ablate", "boost"):
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", _flag(args, "--output-dir", "out"))
        rest = _strip(args, ["--model", "--out", "--output-dir"])
        label = "apostate ablate --model " + shlex.quote(model) + " --out " + shlex.quote(out)
        if rest:
            label += " " + " ".join(shlex.quote(x) for x in rest)
        return run_module(
            ["-m", "apostate.cli", "--optimize", "--model", model, "--output-dir", out, *rest],
            label)

    if cmd == "turbo":
        model = _flag(args, "--model", DEFAULT_MODEL)
        out = _flag(args, "--out", "out")
        label = f"apostate turbo --model {model} --out {out}"
        print("step 1: finetune")
        run_module(["-m", "apostate.finetune", "--model", model, "--out", out + "_ft"], label)
        print("step 2: abliterate")
        run_module(["-m", "apostate.cli", "--optimize", "--model", out + "_ft", "--output-dir", out], label)
        print("step 3: cleanup")
        shutil.rmtree(out + "_ft", ignore_errors=True)
        print("step 4: verify")
        run_module(["-m", "apostate.benchcode", "--model", out, "--base", model], label)
        return 0

    if cmd == "test":
        return run_module(["-m", "apostate.benchcode", *args], f"apostate test {' '.join(args)}".strip())
    if cmd == "talk":
        return run_module(["-m", "apostate.chat", *args], f"apostate talk {' '.join(args)}".strip())
    if cmd == "quantize":
        return run_module(["-m", "apostate.quant", *args])
    if cmd == "train":
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
