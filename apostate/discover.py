"""find local models: huggingface cache ids and baked checkpoint dirs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _hf_cache_roots() -> List[Path]:
    roots: List[Path] = []

    def add(p):
        if not p:
            return
        p = Path(p).resolve()
        if p.exists() and p not in roots:
            roots.append(p)

    add(os.environ.get("HUGGINGFACE_HUB_CACHE"))
    if os.environ.get("HF_HOME"):
        add(Path(os.environ["HF_HOME"]) / "hub")
    add(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _has_snapshot(d: Path) -> bool:
    if (d / "config.json").exists():
        return True
    snaps = d / "snapshots"
    if not snaps.is_dir():
        return False
    markers = ("config.json", "tokenizer_config.json", "processor_config.json")
    return any(
        s.is_dir() and any((s / m).exists() for m in markers)
        for s in snaps.iterdir()
    )


def hf_models() -> List[str]:
    """repo ids sitting in the HF cache (models--org--name -> org/name)."""
    out, seen = [], set()
    for root in _hf_cache_roots():
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for d in entries:
            if not d.is_dir() or not d.name.startswith("models--"):
                continue
            mid = d.name[len("models--"):].replace("--", "/")
            if not mid or mid.lower() in seen or not _has_snapshot(d):
                continue
            seen.add(mid.lower())
            out.append(mid)
    return sorted(out, key=str.lower)


def checkpoints() -> List[str]:
    """local dirs that look like a saved model (config + safetensors)."""
    out, seen = [], set()
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        try:
            entries = list(base.iterdir())
        except OSError:
            continue
        for d in entries:
            if not d.is_dir() or d.name.endswith("_merged") or str(d) in seen:
                continue
            try:
                files = {f.name for f in d.iterdir()}
            except OSError:
                continue
            if "config.json" in files and any(f.endswith(".safetensors") for f in files):
                seen.add(str(d))
                out.append(str(d))
    return out
