"""vllm backend: auto setup, serve, stream. windows routes through wsl."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

SERVED = "apostate"


def _have_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _wsl_check():
    """(ok, message). distinguishes missing vs broken wsl."""
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
    """pure-python C:\\a\\b -> /mnt/c/a/b (wslpath mangles backslashes through wsl.exe)."""
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
        payload = {"model": served, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": True}
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
    # no stdin (else the server steals the chat's keystrokes); logs to a file, not the chat
    return subprocess.Popen(args, stdin=subprocess.DEVNULL,
                            stdout=open(SERVER_LOG, "wb"), stderr=subprocess.STDOUT)


def _serve_via_wsl(model: str, temperature: float, max_tokens: int, port: int) -> bool:
    ok, msg = _wsl_check()
    if not ok:
        print("vllm needs WSL on Windows.\n  " + msg, flush=True)
        return False
    script = _to_wsl_path(os.path.join(os.path.dirname(__file__), "vllm_serve.sh"))
    model_wsl = _to_wsl_path(model)
    print("routing vllm through WSL (first run auto-installs uv + vllm, slow) ...", flush=True)
    proc = _launch(["wsl", "-u", "root", "bash", script, model_wsl, str(port)])
    return _drive(proc, port, temperature, max_tokens)


def _serve_native(model: str, temperature: float, max_tokens: int, port: int) -> bool:
    if not _have_vllm():
        print("setting up vllm (one-time) ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
        if not _have_vllm():
            print("vllm install failed.", flush=True)
            return False
    proc = _launch([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", model, "--served-model-name", SERVED, "--port", str(port)])
    return _drive(proc, port, temperature, max_tokens)


def _drive(proc, port, temperature, max_tokens) -> bool:
    base = f"http://localhost:{port}"
    try:
        if not _wait_ready(base, proc):
            print(f"vllm server did not become ready. log: {SERVER_LOG}", flush=True)
            return True
        _repl(base + "/v1", SERVED, temperature, max_tokens)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return True


def serve_and_chat(model: str, temperature: float, max_tokens: int, port: int = 8000) -> bool:
    """ensure vllm, serve, stream. windows -> wsl. returns False if unavailable."""
    if sys.platform.startswith("win"):
        return _serve_via_wsl(model, temperature, max_tokens, port)
    return _serve_native(model, temperature, max_tokens, port)
