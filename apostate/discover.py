# find local models: huggingface cache ids and baked checkpoint dirs.

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


def _is_model_dir(files) -> bool:
    return "config.json" in files and any(f.endswith(".safetensors") for f in files)


def is_apostate_dir(path: Path, files=None) -> bool:
    try:
        files = files if files is not None else {f.name for f in Path(path).iterdir()}
    except OSError:
        return False
    if not _is_model_dir(files):
        return False
    return (Path(path).name.endswith("-apostate")
            or "apostate_config.json" in files or "report.json" in files)


def apostate_checkpoints() -> List[str]:
    return [c for c in checkpoints() if is_apostate_dir(Path(c))]


# big or irrelevant trees we never descend into during a scan
_SKIP = {
    "windows", "program files", "program files (x86)", "programdata",
    "$recycle.bin", "system volume information", "node_modules", "__pycache__",
    "appdata", ".cache", "recovery", "perflogs", "msocache", "windows.old", "venv",
}


def _env_roots() -> List[Path]:
    raw = os.environ.get("APOSTATE_MODEL_ROOTS", "")
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def _scan_roots() -> List[Path]:
    # every mounted drive, plus the cwd/repo drive and any APOSTATE_MODEL_ROOTS.
    roots = list(_env_roots())
    for anchor in (Path.cwd(), Path(__file__).resolve().parent.parent, Path.home()):
        try:
            roots.append(Path(anchor.anchor or os.sep))
        except OSError:
            pass
    if os.name == "nt":
        roots += [Path(f"{d}:\\") for d in "CDEFGHIJKLMNOP"]
    else:
        roots.append(Path(os.sep))
    out, seen = [], set()
    for r in roots:
        try:
            ok = r.exists() and r.is_dir()
        except OSError:
            ok = False
        key = str(r).lower()
        if ok and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _walk_for_apostate(root: Path, seen: set, found: List[str]):
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _SKIP
            and not d.startswith(".")
            and not d.startswith("$")
            and not d.startswith("models--")
        ]
        files = set(filenames)
        if _is_model_dir(files):
            if is_apostate_dir(Path(dirpath), files):
                found.append(dirpath)
            dirnames[:] = []  # a model dir has no nested models


def scan_apostate(root: str = None) -> List[str]:
    # dynamically walk every drive for baked models, pruning system and cache trees.
    roots = [Path(root)] if root is not None else _scan_roots()
    seen, found = set(), []
    for r in roots:
        _walk_for_apostate(r, seen, found)
    return sorted(set(found))


def apostate_models() -> List[str]:
    # everything talk/test offers: local checkpoints + a full drive scan.
    out, seen = [], set()
    for m in apostate_checkpoints() + scan_apostate():
        key = str(Path(m).resolve()).lower()
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out
