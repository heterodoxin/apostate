"""setup wizard: install python deps, check gpu, optionally provision vllm."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform.startswith("win")
PY_DEPS = ["torch", "transformers", "datasets", "safetensors", "optuna", "bitsandbytes", "textual"]


def _run(args) -> bool:
    print("  $ " + " ".join(args))
    return subprocess.run(args).returncode == 0


def _ask(q) -> str:
    try:
        return input(q).strip().lower()
    except EOFError:
        return ""


def _to_wsl(p) -> str:
    p = str(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def main(argv=None) -> int:
    print("\n=== apostate setup ===")
    print(f"os: {sys.platform} {platform.machine()} | python: {platform.python_version()}\n")

    if _ask(f"[1/3] install python deps ({' '.join(PY_DEPS)})? [Y/n] ") not in ("n", "no"):
        _run([sys.executable, "-m", "pip", "install", "-U", "--quiet", *PY_DEPS])

    print("\n[2/3] gpu ...")
    subprocess.run([sys.executable, "-c",
                    "import torch;print('  cuda', torch.cuda.is_available(),"
                    "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'))"])

    # vLLM is optional and heavy; on Windows it lives in WSL
    q = "[3/3] set up vLLM now? (several GB) [y/N] "
    if _ask("\n" + q) in ("y", "yes"):
        if IS_WIN:
            if subprocess.run(["wsl", "-e", "echo", "ok"], capture_output=True).returncode == 0:
                _run(["wsl", "-u", "root", "bash", _to_wsl(ROOT / "apostate" / "vllm_serve.sh"), "setup"])
            else:
                print("  WSL not ready. In an admin PowerShell run `wsl --install`, reboot, then re-run setup.")
        else:
            _run([sys.executable, "-m", "pip", "install", "-q", "vllm"])

    print("\nready. run: apostate\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
