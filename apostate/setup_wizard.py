# setup wizard: install python deps for the right GPU backend, check gpu, optionally provision vllm.
# AMD/ROCm cards need the ROCm wheel index; plain pip torch pulls a CUDA build that can't see AMD GPUs.

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = sys.platform.startswith("win")

TORCH_PKGS = ["torch", "torchvision", "torchaudio"]
# deps that are backend-agnostic; bitsandbytes is added for CUDA only (below)
BASE_DEPS = ["transformers", "datasets", "safetensors", "optuna", "peft",
             "accelerate", "requests", "sentencepiece", "protobuf", "scipy", "textual"]

# default ROCm wheel index. RDNA4 / gfx12xx needs >= 6.4; bump if you have newer.
DEFAULT_ROCM_INDEX = "https://download.pytorch.org/whl/rocm6.4"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def _run(args) -> bool:
    print("  $ " + " ".join(str(a) for a in args))
    return subprocess.run(args).returncode == 0


def _pip(*args) -> bool:
    return _run([sys.executable, "-m", "pip", *args])


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


def _is_amd_gpu() -> bool:
    # kernel-level signal, no torch needed: the amdgpu compute device node.
    return os.path.exists("/dev/kfd")


def _choose_backend() -> str:
    amd = _is_amd_gpu()
    if amd:
        print("  detected an AMD GPU (/dev/kfd present).")
        default, prompt = "rocm", "backend? [ROCm/cuda/cpu] "
    elif IS_WIN or sys.platform.startswith("linux"):
        default, prompt = "cuda", "backend? [CUDA/rocm/cpu] "
    else:
        default, prompt = "cpu", "backend? [cuda/rocm/CPU] "
    ans = _ask("  " + prompt) or default
    if ans.startswith("r"):
        return "rocm"
    if ans.startswith("cu") or ans == "c":
        return "cuda"
    if ans.startswith("cp"):
        return "cpu"
    return default


def _install_torch(backend: str) -> bool:
    if backend == "rocm":
        idx = _ask(f"  ROCm wheel index [{DEFAULT_ROCM_INDEX}] (RDNA4 needs >=rocm6.4): ") \
            or DEFAULT_ROCM_INDEX
        print("  installing ROCm torch (bundles its own ROCm runtime; no system ROCm needed) ...")
        return _pip("install", "-U", "--index-url", idx, *TORCH_PKGS)
    if backend == "cuda":
        return _pip("install", "-U", "--index-url", CUDA_INDEX, *TORCH_PKGS)
    return _pip("install", "-U", *TORCH_PKGS)


def main(argv=None) -> int:
    print("\n=== apostate setup ===")
    print(f"os: {sys.platform} {platform.machine()} | python: {platform.python_version()}\n")

    pv = sys.version_info
    if pv >= (3, 13):
        print(f"  ! python {pv.major}.{pv.minor}: GPU torch wheels may not exist for this "
              "version yet.\n    if torch install fails, use python 3.10-3.12 (e.g. a venv).\n")

    backend = _choose_backend()
    deps = list(BASE_DEPS)
    if backend == "cuda":
        deps.append("bitsandbytes")  # 4-bit; CUDA-first
    elif backend == "rocm":
        print("  note: skipping bitsandbytes (4-bit is experimental on ROCm). bf16 is the\n"
              "        recommended path; a 24GB+ card runs <=14B models in bf16 comfortably.\n"
              "        for ROCm 4-bit later: install a ROCm build of bitsandbytes manually.")

    if _ask(f"[1/3] install torch ({backend}) + deps ({' '.join(deps)})? [Y/n] ") not in ("n", "no"):
        if not _install_torch(backend):
            print("  ! torch install failed -- check the wheel index / python version above.")
            return 1
        _pip("install", "-U", "--quiet", *deps)

    print("\n[2/3] gpu check ...")
    # guarded: a broken/unsupported runtime prints a message instead of crashing.
    subprocess.run([sys.executable, "-c",
                    "import torch\n"
                    "try:\n"
                    "    ok = torch.cuda.is_available()\n"
                    "    name = torch.cuda.get_device_name(0) if ok else 'cpu-only'\n"
                    "    print('  torch', torch.__version__, '| build hip=', getattr(torch.version,'hip',None),\n"
                    "          'cuda=', getattr(torch.version,'cuda',None))\n"
                    "    print('  gpu visible:', ok, '|', name)\n"
                    "except Exception as e:\n"
                    "    print('  gpu probe error:', type(e).__name__, e)\n"])
    print("  next: run `apostate doctor` to verify the GPU can actually execute a kernel\n"
          "        (catches RDNA4/gfx12xx-on-old-ROCm BEFORE a big model load).")

    # vLLM is optional and heavy; on Windows it lives in WSL
    q = "[3/3] set up vLLM now? (several GB) [y/N] "
    if _ask("\n" + q) in ("y", "yes"):
        if IS_WIN:
            if subprocess.run(["wsl", "-e", "echo", "ok"], capture_output=True).returncode == 0:
                _run(["wsl", "-u", "root", "bash", _to_wsl(ROOT / "apostate" / "vllm_serve.sh"), "setup"])
            else:
                print("  WSL not ready. In an admin PowerShell run `wsl --install`, reboot, then re-run setup.")
        else:
            _pip("install", "-q", "vllm")

    print("\nready. run: apostate\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
