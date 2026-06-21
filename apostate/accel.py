# accelerator backend detection + anti-freeze guards (cuda / rocm / cpu).
# ROCm exposes via torch.cuda (HIP masquerades as CUDA); device string stays "cuda" on ROCm.
# Guards fail fast on CPU before any GPU allocation — gfx12xx on old ROCm hangs instead of raising.

from __future__ import annotations

import os
from typing import Optional

import torch


def gpu_backend() -> Optional[str]:
    # Returns 'rocm', 'cuda', or None based on the torch build, not GPU reachability.
    # Reads string attributes only — never initializes a runtime, safe on half-broken drivers.
    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return None


def is_rocm() -> bool:
    return gpu_backend() == "rocm"


def gpu_available() -> bool:
    # Like torch.cuda.is_available() but swallows HIP/CUDA init failures instead of raising.
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_name(index: int = 0) -> str:
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return "unknown gpu"


def gpu_arch() -> Optional[str]:
    # gfx target of the first GPU (e.g. 'gfx1201' for RDNA4) or 'sm_XX' on CUDA.
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None
    arch = getattr(props, "gcnArchName", None)  # rocm
    if arch:
        return str(arch).split(":")[0]  # strip feature flags like ':sramecc+:xnack-'
    major = getattr(props, "major", None)
    minor = getattr(props, "minor", None)
    if major is not None and minor is not None:
        return f"sm_{major}{minor}"
    return None


def resolve_device(device: Optional[str]) -> str:
    # Turn 'auto'/None/'' into a concrete device string.
    # ROCm resolves to 'cuda' — that's the device string the ROCm runtime expects.
    if device not in ("auto", None, ""):
        return device
    if gpu_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                return "xpu"
        except Exception:
            pass
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        try:
            if mps.is_available():
                return "mps"
        except Exception:
            pass
    return "cpu"


def install_hint() -> str:
    # Build-aware pip install hint. Uses /dev/kfd to detect AMD even before a GPU-capable torch is installed.
    backend = gpu_backend()
    if backend == "rocm" or _looks_like_amd():
        return (
            "this is an AMD GPU: install a ROCm torch wheel new enough for your "
            "card (RDNA4 / gfx12xx needs ROCm >= 6.4, ideally 7.0+), e.g.\n"
            "  python -m pip install --force-reinstall \\\n"
            "    --index-url https://download.pytorch.org/whl/rocm6.4 \\\n"
            "    torch torchvision torchaudio\n"
            "see https://pytorch.org/get-started/locally for the current ROCm index."
        )
    return (
        "install a CUDA torch wheel, e.g.\n"
        "  python -m pip install --force-reinstall \\\n"
        "    --index-url https://download.pytorch.org/whl/cu128 \\\n"
        "    torch torchvision torchaudio"
    )


def _looks_like_amd() -> bool:
    # kernel-level signal that doesn't need torch's GPU runtime: the amdgpu
    # compute device node. Lets us give AMD-specific advice before a GPU-capable
    # torch is even installed.
    try:
        return os.path.exists("/dev/kfd")
    except Exception:
        return False


def require_gpu(device: str) -> None:
    # Raise a clear backend-aware error if a GPU device was asked for but torch can't see one.
    if device in ("cpu", "mps", "xpu"):
        return
    if gpu_available():
        return
    backend = gpu_backend() or "no gpu build"
    raise RuntimeError(
        f"device '{device}' requested but torch cannot see a gpu "
        f"(torch={torch.__version__}, build={backend}, "
        f"hip={getattr(torch.version, 'hip', None)}, "
        f"cuda={getattr(torch.version, 'cuda', None)}).\n" + install_hint()
    )


def bitsandbytes_status() -> tuple[bool, str]:
    # Returns (usable, reason). Only checks import-level — launching a bnb GPU kernel to test it
    # could hang on unsupported archs; a broken bnb will raise cleanly at load time instead.
    try:
        import importlib.util
        if importlib.util.find_spec("bitsandbytes") is None:
            return False, "bitsandbytes is not installed"
    except Exception as e:
        return False, f"bitsandbytes lookup failed: {e}"
    try:
        import bitsandbytes as bnb  # importing also probes the native lib
    except Exception as e:
        return False, f"bitsandbytes import failed ({type(e).__name__}: {e})"
    # multi-backend builds (>=0.45) register backends; CUDA-only builds on ROCm
    # will not have a rocm/hip backend. Treat unknown layouts as 'usable' and
    # let the real load surface any error cleanly.
    backends = getattr(bnb, "backends", None)
    if isinstance(backends, dict) and is_rocm():
        keys = ",".join(backends) or "<none>"
        if not any(k in keys for k in ("rocm", "hip", "cuda")):
            return False, f"bitsandbytes has no rocm backend (registered: {keys})"
    return True, "ok"


def _bytes_per_param(load_in_4bit: bool, compute_dtype: str) -> float:
    if load_in_4bit:
        # nf4 ~0.5 B/param + quant state; rounded up so we never under-estimate (under-estimate risks a hang).
        return 0.6
    return {"float16": 2.0, "bfloat16": 2.0, "float32": 4.0}.get(compute_dtype, 2.0)


def estimate_param_count(model_id: str, trust_remote_code: bool = True) -> Optional[int]:
    # Param count via meta-device build — no real memory or GPU touch. Returns None on failure.
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception:
        return None
    try:
        hf = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        with init_empty_weights():
            m = AutoModelForCausalLM.from_config(hf, trust_remote_code=trust_remote_code)
        n = sum(p.numel() for p in m.parameters())
        del m
        return int(n)
    except Exception:
        return None


def preflight_vram(
    device: str,
    *,
    model_id: str,
    load_in_4bit: bool,
    compute_dtype: str,
    batch_size: int = 24,
    n_params: Optional[int] = None,
    log=print,
) -> None:
    # Refuse to load if the model clearly won't fit in free VRAM before any big GPU allocation.
    # Best-effort: warns and proceeds if measurement fails; set APOSTATE_STRICT_VRAM=1 to make that fatal.
    if device != "cuda":
        return
    strict = os.environ.get("APOSTATE_STRICT_VRAM", "").lower() in ("1", "true", "yes")

    if n_params is None:
        n_params = estimate_param_count(model_id)
    if n_params is None:
        msg = ("could not estimate model size for the VRAM preflight "
               "(meta-device build failed). loading without a memory guard.")
        if strict:
            raise RuntimeError(msg + " (APOSTATE_STRICT_VRAM is set)")
        log("warning: " + msg)
        return

    try:
        free, total = torch.cuda.mem_get_info()
    except Exception as e:
        msg = f"could not read free VRAM for the preflight ({type(e).__name__}: {e})."
        if strict:
            raise RuntimeError(msg + " (APOSTATE_STRICT_VRAM is set)")
        log("warning: " + msg + " loading without a memory guard.")
        return

    bpp = _bytes_per_param(load_in_4bit, compute_dtype)
    weights = n_params * bpp
    overhead = weights * 0.15                      # loader copies / temporary buffers
    headroom = max(1.5e9, batch_size * 5e7)        # activations + kv cache, scales a bit
    need = weights + overhead + headroom

    g = 1e9
    log(f"vram preflight: model ~{n_params/1e9:.2f}B params, "
        f"need ~{need/g:.1f} GB ({'4bit' if load_in_4bit else compute_dtype}), "
        f"free {free/g:.1f}/{total/g:.1f} GB on {device_name()}")
    if need > free:
        raise RuntimeError(
            f"refusing to load: estimated need ~{need/g:.1f} GB exceeds free VRAM "
            f"~{free/g:.1f} GB (of {total/g:.1f} GB). loading anyway would likely "
            f"hang the GPU and freeze the machine. options:\n"
            f"  - enable 4-bit:        --load-in-4bit   (needs working bitsandbytes)\n"
            f"  - smaller model:       --model <smaller>\n"
            f"  - smaller batch:       --batch-size {max(1, batch_size // 2)}\n"
            f"  - free VRAM / close other GPU apps\n"
            f"set APOSTATE_SKIP_VRAM_CHECK=1 only if you are certain this estimate is wrong."
        )


def maybe_preflight(device: str, **kw) -> None:
    # preflight_vram unless APOSTATE_SKIP_VRAM_CHECK is set.
    if os.environ.get("APOSTATE_SKIP_VRAM_CHECK", "").lower() in ("1", "true", "yes"):
        return
    preflight_vram(device, **kw)


def gpu_smoke_test(device: str, log=print) -> bool:
    # Runs a tiny GPU op to confirm the runtime executes kernels for this arch before any large load.
    # On RDNA4 with old ROCm, kernels can hang rather than raise — cheaper to discover that on 8 bytes.
    if device != "cuda":
        return True
    if os.environ.get("APOSTATE_SKIP_GPU_SMOKE", "").lower() in ("1", "true", "yes"):
        return True
    arch = gpu_arch()
    log(f"gpu smoke test on {device_name()} ({arch or 'arch?'}) ...")
    try:
        x = torch.ones(8, device="cuda")
        y = (x * 2).sum().item()
        torch.cuda.synchronize()
    except Exception as e:
        raise RuntimeError(
            f"GPU smoke test failed on {device_name()} ({arch}): "
            f"{type(e).__name__}: {e}\n"
            "the runtime cannot execute a basic kernel on this GPU -- almost "
            "always a ROCm/arch mismatch (e.g. RDNA4/gfx12xx on ROCm < 6.4).\n"
            + install_hint()
        ) from e
    if y != 16.0:
        raise RuntimeError(
            f"GPU smoke test returned wrong result ({y} != 16.0) on "
            f"{device_name()} ({arch}). the kernel ran but produced garbage -- "
            "this arch is not correctly supported by the installed ROCm/torch. "
            "do NOT proceed; results would be silently corrupt.\n" + install_hint()
        )
    log("gpu smoke test ok")
    return True
