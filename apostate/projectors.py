from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import torch

from .model import ModelBundle


class ProjectionController:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.device = next(bundle.model.parameters()).device
        self.enabled = False
        self._handles = []
        self._hooked = set()
        self._modules: List[torch.nn.Module] = []
        self._layer_writers: List[Tuple[torch.nn.Module, torch.nn.Module]] = []
        self._ple_writers: List[Tuple[torch.nn.Module, ...]] = []
        self._embed: Optional[torch.nn.Module] = None
        self._ple_embed: Optional[torch.nn.Module] = None
        self._ple_model_projection: Optional[torch.nn.Module] = None
        self._final: Optional[torch.nn.Module] = None
        self.edits: List[dict] = []
        self._register()
        self.add_edit("primary", sign=-1.0, default_alpha=1.0)
        self.add_edit("head_token", sign=-1.0, default_alpha=0.0)
        if any(self._ple_writers):
            self.add_edit("ple", sign=-1.0, default_alpha=0.0, kind="ple_gate")
        if self._ple_embed is not None:
            self.add_edit("ple_embed", sign=-1.0, default_alpha=0.0, kind="ple_embed")
        if self._ple_model_projection is not None:
            self.add_edit("ple_model_projection", sign=-1.0, default_alpha=0.0, kind="ple_model_projection")
        self.set_head_alpha(0.0)

    def _register(self):
        b = self.bundle
        self._embed = b.embed()
        self._ple_embed = b.ple_embed()
        self._ple_model_projection = b.ple_model_projection()
        self._modules = [self._embed]
        self._final = b.final_norm()
        for layer in b.layers():
            writers = b.layer_writers(layer)
            ple = tuple(b.ple_writers(layer))
            self._layer_writers.append(writers)
            self._ple_writers.append(ple)
            self._modules.extend(writers)
        if self._final is not None:
            self._modules.append(self._final)
        seen, uniq = set(), []
        for m in self._modules:
            if id(m) not in seen:
                seen.add(id(m))
                uniq.append(m)
        self._modules = uniq
        for m in self._modules:
            self._ensure_hook(m)

    def _ensure_hook(self, module: torch.nn.Module):
        mid = id(module)
        if mid in self._hooked:
            return
        self._hooked.add(mid)
        self._handles.append(module.register_forward_hook(self._make_hook(module)))

    def _modules_for_kind(self, kind: str) -> List[torch.nn.Module]:
        if kind == "ple_gate":
            return [m for mods in self._ple_writers for m in mods]
        if kind == "ple_embed":
            return [self._ple_embed] if self._ple_embed is not None else []
        if kind == "ple_model_projection":
            return [self._ple_model_projection] if self._ple_model_projection is not None else []
        return self._modules

    def _cast(self, edit: dict, dtype, device):
        cache = edit.get("_cast")
        if cache is None:
            cache = {}
            edit["_cast"] = cache
        key = (dtype, device)
        Rd = cache.get(key)
        if Rd is None:
            Rd = edit["R"].to(dtype=dtype, device=device)
            cache[key] = Rd
        return Rd

    def _make_hook(self, module):
        mod_id = id(module)

        def hook(_mod, _inp, out):
            if not self.enabled:
                return out
            t = out[0] if isinstance(out, tuple) else out
            delta = None
            for e in self.edits:
                R = e["R"]
                if R is None:
                    continue
                a = e["alpha"].get(mod_id, 0.0)
                if a == 0.0:
                    continue
                if R.shape[0] != t.shape[-1]:
                    continue
                Rd = self._cast(e, t.dtype, t.device)
                term = (t @ Rd) @ Rd.t()
                contrib = (e["sign"] * a) * term
                delta = contrib if delta is None else delta + contrib
            if delta is None:
                return out
            t2 = t + delta
            if isinstance(out, tuple):
                return (t2,) + tuple(out[1:])
            return t2

        return hook

    def add_edit(self, name: str, sign: float, default_alpha: float = 0.0, kind: str = "hidden"):
        alpha = {id(m): default_alpha for m in self._modules_for_kind(kind)}
        self.edits.append({"name": name, "kind": kind, "sign": float(sign), "R": None, "alpha": alpha})

    def _edit(self, name: str) -> dict:
        for e in self.edits:
            if e["name"] == name:
                return e
        raise KeyError(f"no edit named {name!r}")

    def set_edit_subspace(self, name: str, R: torch.Tensor):
        e = self._edit(name)
        e["R"] = R.to(self.device).float()
        e["_cast"] = None

    def set_edit_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        for m in self._layer_writers[layer_idx]:
            e["alpha"][id(m)] = value

    def set_edit_embed_alpha(self, name: str, value: float):
        self._edit(name)["alpha"][id(self._embed)] = value

    def set_edit_head_alpha(self, name: str, value: float):
        if self._final is not None:
            self._edit(name)["alpha"][id(self._final)] = value

    def set_edit_ple_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        for m in self._ple_writers[layer_idx]:
            e["alpha"][id(m)] = value

    def set_edit_uniform_alpha(self, name: str, value: float):
        e = self._edit(name)
        final_id = id(self._final) if self._final is not None else None
        for k in e["alpha"]:
            if k == final_id:
                continue
            e["alpha"][k] = value

    def get_edit_layer_alpha(self, name: str, layer_idx: int) -> float:
        return self._edit(name)["alpha"][id(self._layer_writers[layer_idx][0])]

    @property
    def R(self):
        return self.edits[0]["R"]

    @property
    def alpha(self):
        return self.edits[0]["alpha"]

    @alpha.setter
    def alpha(self, value):
        self.edits[0]["alpha"] = value

    def set_subspace(self, R: torch.Tensor):
        self.edits[0]["R"] = R.to(self.device).float()
        self.edits[0]["_cast"] = None

    def set_uniform_alpha(self, value: float):
        self.set_edit_uniform_alpha("primary", value)

    def set_layer_alpha(self, layer_idx: int, value: float):
        self.set_edit_layer_alpha("primary", layer_idx, value)

    def set_embed_alpha(self, value: float):
        self.set_edit_embed_alpha("primary", value)

    def set_head_alpha(self, value: float):
        self.set_edit_head_alpha("primary", value)

    def set_head_token_subspace(self, R: torch.Tensor):
        self.set_edit_subspace("head_token", R)

    def set_head_token_alpha(self, value: float):
        self.set_edit_head_alpha("head_token", value)

    def get_head_token_alpha(self) -> float:
        if self._final is None:
            return 0.0
        return self._edit("head_token")["alpha"].get(id(self._final), 0.0)

    def has_ple(self) -> bool:
        return any(self._ple_writers)

    def clear_ple(self):
        if not self.has_ple():
            pass
        else:
            e = self._edit("ple")
            e["R"] = None
            e["_cast"] = None
            for mods in self._ple_writers:
                for m in mods:
                    e["alpha"][id(m)] = 0.0
        for name, mod in (
            ("ple_embed", self._ple_embed),
            ("ple_model_projection", self._ple_model_projection),
        ):
            if mod is None:
                continue
            e = self._edit(name)
            e["R"] = None
            e["_cast"] = None
            e["alpha"][id(mod)] = 0.0

    def set_ple_subspace(self, R: torch.Tensor):
        e = self._edit("ple")
        for mods in self._ple_writers:
            for mod in mods:
                self._ensure_hook(mod)
        e["R"] = R.to(self.device).float()
        e["_cast"] = None

    def set_ple_layer_alpha(self, layer_idx: int, value: float):
        self.set_edit_ple_layer_alpha("ple", layer_idx, value)

    def get_ple_layer_alpha(self, layer_idx: int) -> float:
        if not self.has_ple() or not self._ple_writers[layer_idx]:
            return 0.0
        return self._edit("ple")["alpha"].get(id(self._ple_writers[layer_idx][0]), 0.0)

    def set_ple_embed_subspace(self, R: torch.Tensor):
        e = self._edit("ple_embed")
        if self._ple_embed is not None:
            self._ensure_hook(self._ple_embed)
        e["R"] = R.to(self.device).float()
        e["_cast"] = None

    def set_ple_embed_alpha(self, value: float):
        if self._ple_embed is not None:
            self._edit("ple_embed")["alpha"][id(self._ple_embed)] = value

    def get_ple_embed_alpha(self) -> float:
        if self._ple_embed is None:
            return 0.0
        return self._edit("ple_embed")["alpha"].get(id(self._ple_embed), 0.0)

    def set_ple_model_projection_subspace(self, R: torch.Tensor):
        e = self._edit("ple_model_projection")
        if self._ple_model_projection is not None:
            self._ensure_hook(self._ple_model_projection)
        e["R"] = R.to(self.device).float()
        e["_cast"] = None

    def set_ple_model_projection_alpha(self, value: float):
        if self._ple_model_projection is not None:
            self._edit("ple_model_projection")["alpha"][id(self._ple_model_projection)] = value

    def get_ple_model_projection_alpha(self) -> float:
        if self._ple_model_projection is None:
            return 0.0
        return self._edit("ple_model_projection")["alpha"].get(id(self._ple_model_projection), 0.0)

    def get_embed_alpha(self) -> float:
        return self._edit("primary")["alpha"][id(self._embed)]

    def get_head_alpha(self) -> float:
        if self._final is None:
            return 0.0
        return self._edit("primary")["alpha"].get(id(self._final), 0.0)

    def get_layer_alpha(self, layer_idx: int) -> float:
        return self.get_edit_layer_alpha("primary", layer_idx)

    def isolate_layer(self, layer_idx: int):
        self.set_uniform_alpha(0.0)
        self.set_layer_alpha(layer_idx, 1.0)

    @property
    def num_layers(self) -> int:
        return len(self._layer_writers)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    @contextmanager
    def active(self):
        prev = self.enabled
        self.enabled = True
        try:
            yield self
        finally:
            self.enabled = prev

    @contextmanager
    def bypassed(self):
        prev = self.enabled
        self.enabled = False
        try:
            yield self
        finally:
            self.enabled = prev

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._hooked = set()

    def export(self) -> dict:
        out_edits = []
        for e in self.edits:
            if e["R"] is None:
                continue
            if e.get("kind") == "ple_gate":
                layer_alphas = [
                    e["alpha"].get(id(self._ple_writers[i][0]), 0.0) if self._ple_writers[i] else 0.0
                    for i in range(len(self._ple_writers))
                ]
                embed_alpha = 0.0
                head_alpha = 0.0
            elif e.get("kind") == "ple_embed":
                layer_alphas = [0.0 for _ in range(len(self._layer_writers))]
                embed_alpha = e["alpha"].get(id(self._ple_embed), 0.0) if self._ple_embed is not None else 0.0
                head_alpha = 0.0
            elif e.get("kind") == "ple_model_projection":
                layer_alphas = [0.0 for _ in range(len(self._layer_writers))]
                embed_alpha = e["alpha"].get(id(self._ple_model_projection), 0.0) if self._ple_model_projection is not None else 0.0
                head_alpha = 0.0
            else:
                layer_alphas = [
                    e["alpha"][id(self._layer_writers[i][0])] for i in range(len(self._layer_writers))
                ]
                embed_alpha = e["alpha"][id(self._embed)]
                head_alpha = e["alpha"].get(id(self._final), 0.0) if self._final is not None else 0.0
            out_edits.append({
                "name": e["name"],
                "kind": e.get("kind", "hidden"),
                "sign": e["sign"],
                "R": e["R"].detach().cpu(),
                "embed_alpha": embed_alpha,
                "head_alpha": head_alpha,
                "layer_alphas": layer_alphas,
            })
        return {"edits": out_edits}
