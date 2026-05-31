"""vllm backend"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

SERVED = "apostate"
_TURBOQUANT_PREFIX = "turboquant_"


def _have_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _wsl_check():
    """wsl check"""
    try:
        r = subprocess.run(["wsl", "-e", "echo", "ok"], capture_output=True, timeout=30)
    except FileNotFoundError:
        return False, "WSL not installed. Admin PowerShell: wsl --install   then reboot."
    except Exception as e:
        return False, f"WSL check failed: {e}"
    if r.returncode == 0:
        return True, ""
    err = (r.stdout + r.stderr).decode("utf-16le", "replace").replace("\x00", "").strip()
    return False, ("WSL is installed but not starting:\n  " + err[:300] +
                   "\n  repair: wsl --unregister Ubuntu  then  wsl --install -d Ubuntu")


def _to_wsl_path(win_path: str) -> str:
    """wsl path"""
    p = win_path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def _wait_ready(base: str, proc, timeout: int = 1800) -> bool:
    import requests
    print("starting vllm server (first run installs + downloads, slow) ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print("vllm server exited early.", flush=True)
            return False
        try:
            if requests.get(base + "/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _repl(v1: str, served: str, temperature: float, max_tokens: int):
    import requests
    messages = []
    print("\nchat ready (vllm).  /reset  /exit\n", flush=True)
    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit", "/q"):
            break
        if user == "/reset":
            messages = []
            print("(conversation cleared)\n")
            continue
        messages.append({"role": "user", "content": user})
        print("\033[35mmodel>\033[0m ", end="", flush=True)
        acc = ""
        payload = {"model": served, "messages": messages, "temperature": temperature, "stream": True}
        if max_tokens and max_tokens > 0:   # token cap
            payload["max_tokens"] = max_tokens
        try:
            with requests.post(v1 + "/chat/completions", json=payload, stream=True, timeout=600) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    s = line.decode("utf-8", "replace")
                    if s.startswith("data: "):
                        s = s[6:]
                    if s.strip() == "[DONE]":
                        break
                    try:
                        d = json.loads(s)["choices"][0]["delta"].get("content", "")
                    except Exception:
                        continue
                    if d:
                        acc += d
                        print(d, end="", flush=True)
        except Exception as e:
            print(f"\n[server error: {e}]", flush=True)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": acc})
        print("\n")
    print("bye.")


import tempfile

SERVER_LOG = os.path.join(tempfile.gettempdir(), "apostate_vllm.log")


def _launch(args: list):
    # detach stdin
    kw = dict(stdin=subprocess.DEVNULL, stdout=open(SERVER_LOG, "wb"), stderr=subprocess.STDOUT)
    if sys.platform.startswith("win"):
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(args, **kw)


def _cleanup_wsl_vllm(port: int, shutdown_wsl: bool):
    try:
        subprocess.run(
            ["wsl", "-u", "root", "bash", "-lc",
             "pkill -f 'vllm.entrypoints.openai.api_server.*--port %d' || true" % int(port)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
    except Exception:
        pass
    if shutdown_wsl and os.environ.get("APOSTATE_KEEP_WSL", "").lower() not in ("1", "true", "yes"):
        try:
            subprocess.run(
                ["wsl", "--shutdown"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
        except Exception:
            pass


def _kv_args(kv_cache_dtype: str | None) -> list:
    kv = (kv_cache_dtype or "auto").strip().lower()
    if not kv or kv in ("auto", "bf16", "bfloat16"):
        return []
    return ["--kv-cache-dtype", kv]


def _serve_via_wsl(
    model: str, temperature: float, max_tokens: int, port: int,
    kv_cache_dtype: str | None, shutdown_wsl: bool,
) -> bool:
    ok, msg = _wsl_check()
    if not ok:
        print("vllm needs WSL on Windows.\n  " + msg, flush=True)
        return False
    script = _to_wsl_path(os.path.join(os.path.dirname(__file__), "vllm_serve.sh"))
    model_wsl = _to_wsl_path(model)
    kv = kv_cache_dtype or "auto"
    if kv.startswith(_TURBOQUANT_PREFIX):
        print(f"routing vllm through WSL with TurboQuant KV cache ({kv}) ...", flush=True)
    else:
        print("routing vllm through WSL (first run auto-installs uv + vllm, slow) ...", flush=True)
    proc = _launch(["wsl", "-u", "root", "bash", script, model_wsl, str(port), kv])
    return _drive(
        proc, port, temperature, max_tokens,
        cleanup=lambda: _cleanup_wsl_vllm(port, shutdown_wsl),
    )


def _serve_native(
    model: str, temperature: float, max_tokens: int, port: int,
    kv_cache_dtype: str | None,
) -> bool:
    if not _have_vllm():
        print("setting up vllm (one-time) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
        if not _have_vllm():
            print("vllm install failed.", flush=True)
            return False
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # avoid jit
    args = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", SERVED, "--port", str(port),
            "--enforce-eager", *_kv_args(kv_cache_dtype)]
    proc = _launch(args)
    return _drive(proc, port, temperature, max_tokens)


def _drive(proc, port, temperature, max_tokens, cleanup=None) -> bool:
    base = f"http://localhost:{port}"
    try:
        if not _wait_ready(base, proc):
            print(f"vllm server did not become ready. log: {SERVER_LOG}", flush=True)
            return False
        _repl(base + "/v1", SERVED, temperature, max_tokens)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        if cleanup is not None:
            cleanup()
    return True


def serve_and_chat(
    model: str, temperature: float, max_tokens: int, port: int = 8000,
    kv_cache_dtype: str | None = "auto", shutdown_wsl: bool = True,
) -> bool:
    """serve chat"""
    if sys.platform.startswith("win"):
        return _serve_via_wsl(model, temperature, max_tokens, port, kv_cache_dtype, shutdown_wsl)
    return _serve_native(model, temperature, max_tokens, port, kv_cache_dtype)
