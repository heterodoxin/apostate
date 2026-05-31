"""vllm backend: auto setup, serve, stream."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time


def _have_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None


def ensure_vllm() -> bool:
    """install vllm on first use (linux/wsl only)."""
    if _have_vllm():
        return True
    if sys.platform.startswith("win"):
        print("vllm runs on linux/wsl, not native windows.\n"
              "  open WSL and: pip install vllm", flush=True)
        return False
    print("setting up vllm (one-time) ...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    return _have_vllm()


def _wait_ready(base: str, proc, timeout: int = 600) -> bool:
    import requests
    print("starting vllm server ...", flush=True)
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
        time.sleep(2)
    return False


def _repl(v1: str, model: str, temperature: float, max_tokens: int):
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
        payload = {"model": model, "messages": messages, "temperature": temperature,
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


def serve_and_chat(model: str, temperature: float, max_tokens: int, port: int = 8000) -> bool:
    """ensure vllm, launch server, stream a chat. returns False if unavailable."""
    if not ensure_vllm():
        return False
    base = f"http://localhost:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", model, "--port", str(port)])
    try:
        if not _wait_ready(base, proc):
            print("vllm server did not become ready.", flush=True)
            return True   # handled (don't fall back to a slow local load)
        _repl(base + "/v1", model, temperature, max_tokens)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return True
