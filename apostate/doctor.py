# apostate doctor: safe GPU readiness check. Runs only a tiny kernel (8 floats).
# Exit 0 = ready, 1 = problem found.

from __future__ import annotations

import sys

from . import accel


def main(argv=None) -> int:
    import torch

    print("=== apostate doctor ===")
    print(f"torch        : {torch.__version__}")
    print(f"build        : backend={accel.gpu_backend()} "
          f"hip={getattr(torch.version, 'hip', None)} "
          f"cuda={getattr(torch.version, 'cuda', None)}")

    dev = accel.resolve_device("auto")
    print(f"resolved dev : {dev}")
    if dev != "cuda":
        print(f"\nno GPU resolved -- would run on '{dev}'. "
              "if you expected a GPU:\n" + accel.install_hint())
        return 0

    print(f"gpu          : {accel.device_name()} ({accel.gpu_arch()})")
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"vram free    : {free/1e9:.1f} / {total/1e9:.1f} GB")
    except Exception as e:
        print(f"vram         : could not read ({type(e).__name__}: {e})")

    ok, why = accel.bitsandbytes_status()
    print(f"bitsandbytes : {'usable' if ok else 'NOT usable -> use bf16 / --no-load-in-4bit'} ({why})")

    print("\nrunning tiny GPU smoke test (8 floats) ...")
    try:
        accel.gpu_smoke_test(dev, log=lambda m: print("  " + m))
    except Exception as e:
        print("\nFAILED:\n" + str(e))
        return 1

    print("\nall good: this GPU executes kernels correctly. safe to load a model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
