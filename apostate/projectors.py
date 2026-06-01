"""projection hooks"""

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
        self._modules: List[torch.nn.Module] = []
        self._layer_writers: List[Tuple[torch.nn.Module, torch.nn.Module]] = []
        self._embed: Optional[torch.nn.Module] = None
        self._final: Optional[torch.nn.Module] = None
        # edits
        self.edits: List[dict] = []
        self._register()
        self.add_edit("primary", sign=-1.0, default_alpha=1.0)  # refusal edit
        self.set_head_alpha(0.0)

    # registration
    def _register(self):
        b = self.bundle
        self._embed = b.embed()
        self._modules = [self._embed]
        self._final = b.final_norm()
        for layer in b.layers():
            writers = b.layer_writers(layer)   # residual writers
            self._layer_writers.append(writers)
            self._modules.extend(writers)
        if self._final is not None:
            self._modules.append(self._final)
        # dedup modules
        seen, uniq = set(), []
        for m in self._modules:
            if id(m) not in seen:
                seen.add(id(m))
                uniq.append(m)
        self._modules = uniq
        for m in self._modules:
            self._handles.append(m.register_forward_hook(self._make_hook(m)))

    # cast cache
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

    # edit state
    def add_edit(self, name: str, sign: float, default_alpha: float = 0.0):
        alpha = {id(m): default_alpha for m in self._modules}
        self.edits.append({"name": name, "sign": float(sign), "R": None, "alpha": alpha})

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

    def set_edit_uniform_alpha(self, name: str, value: float):
        e = self._edit(name)
        final_id = id(self._final) if self._final is not None else None
        for k in e["alpha"]:
            if k == final_id:
                continue
            e["alpha"][k] = value

    def get_edit_layer_alpha(self, name: str, layer_idx: int) -> float:
        return self._edit(name)["alpha"][id(self._layer_writers[layer_idx][0])]

    # primary api
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

    def get_embed_alpha(self) -> float:
        return self._edit("primary")["alpha"][id(self._embed)]

    def get_head_alpha(self) -> float:
        if self._final is None:
            return 0.0
        return self._edit("primary")["alpha"].get(id(self._final), 0.0)

    def get_layer_alpha(self, layer_idx: int) -> float:
        return self.get_edit_layer_alpha("primary", layer_idx)

    def isolate_layer(self, layer_idx: int):
        """isolate layer"""
        self.set_uniform_alpha(0.0)
        self.set_layer_alpha(layer_idx, 1.0)

    @property
    def num_layers(self) -> int:
        return len(self._layer_writers)

    # control
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

    # export
    def export(self) -> dict:
        out_edits = []
        for e in self.edits:
            if e["R"] is None:
                continue
            layer_alphas = [
                e["alpha"][id(self._layer_writers[i][0])] for i in range(len(self._layer_writers))
            ]   # layer alpha
            out_edits.append({
                "name": e["name"],
                "sign": e["sign"],
                "R": e["R"].detach().cpu(),
                "embed_alpha": e["alpha"][id(self._embed)],
                "head_alpha": e["alpha"].get(id(self._final), 0.0) if self._final is not None else 0.0,
                "layer_alphas": layer_alphas,
            })
        return {"edits": out_edits}
