# runtime edit controller: forward hooks that project refusal directions out; off == original model.

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import torch

from .model import ModelBundle


def oblique_vectors(R: torch.Tensor, mu: torch.Tensor, strength: float, denom_floor: float = 0.2):
    # E = I - Rbake U^T. U = R minus strength*its harmless-mean part (E mu = mu); Rbake = R(U^T R)^-1
    # (E R = 0). strength clamps down if U^T R is ill-conditioned (R near-parallel to mu).
    R = R.float()
    mu = mu.float()
    mhat = mu / (mu.norm() + 1e-8)
    c = mhat @ R  # [r]
    s = float(max(0.0, strength))
    while s > 1e-3:
        U = R - s * torch.outer(mhat, c)
        UtR = U.t() @ R
        sym = 0.5 * (UtR + UtR.t())
        if float(torch.linalg.eigvalsh(sym).min()) >= denom_floor:
            try:
                Rbake = R @ torch.linalg.inv(UtR)
                return U, Rbake
            except RuntimeError:
                pass
        s -= 0.05
    return R, R  # fell back to symmetric


def predictive_covector(R: torch.Tensor, harmless: torch.Tensor, ridge: float = 1e-2):
    # read co-vector D = R - W, W = harmless ridge-predictor of R^T x from x orthogonal to R.
    # harmless R-variance is predictable (W reproduces it) so D^T x ~ 0 there -> preserved;
    # the harmful refusal excursion is OOD for W -> D^T x large -> removed. E R = 0 (W _|_ R),
    # E mu ~ mu, but preserves harmless VARIANCE along R, not just the mean. bake: U = D, Rbake = R.
    R = R.float()
    X = harmless.float()
    Xp = X - (X @ R) @ R.t()          # harmless with the R-subspace removed
    try:
        G = Xp.t() @ Xp
        lam = float(ridge) * (float(torch.diagonal(G).mean()) + 1e-8)
        G = G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        W = torch.linalg.solve(G, Xp.t() @ (X @ R))   # [hidden, r]
        W = W - R @ (R.t() @ W)       # keep W orthogonal to R
        D = R - W
        if not torch.isfinite(D).all():
            return None, None
        return D, R                   # U = D, Rbake = R
    except RuntimeError:
        return None, None


class ProjectionController:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self.device = next(bundle.model.parameters()).device
        self.enabled = False
        # post-norm models (gemma2/3/4) renormalize writer outputs, so we ablate the
        # reader side (q/k/v/gate/up inputs) instead via pre-hooks. see model.uses_post_norm.
        self.reader_mode = bool(bundle.uses_post_norm())
        self._handles = []
        self._pre_handles = []
        self._hooked = set()
        self._pre_hooked = set()
        self._modules: List[torch.nn.Module] = []
        self._residual_layers: List[torch.nn.Module] = []
        self._layer_writers: List[Tuple[torch.nn.Module, torch.nn.Module]] = []
        self._reader_modules: List[Tuple[torch.nn.Module, ...]] = []
        self._module_layer: dict = {}  # reader module id -> layer index, for per-layer dirs
        self._kv_writers: List[Tuple[Tuple[str, torch.nn.Module], ...]] = []
        self._query_writers: List[Tuple[torch.nn.Module, ...]] = []
        self._ple_writers: List[Tuple[torch.nn.Module, ...]] = []
        self._ple_projection_writers: List[Tuple[torch.nn.Module, ...]] = []
        self._embed: Optional[torch.nn.Module] = None
        self._ple_embed: Optional[torch.nn.Module] = None
        self._ple_model_projection: Optional[torch.nn.Module] = None
        self._final: Optional[torch.nn.Module] = None
        self.edits: List[dict] = []
        self._oblique_mean: Optional[torch.Tensor] = None
        self._oblique_strength: float = 1.0
        self._oblique_floor: float = 0.2
        self._oblique_writers_only: bool = True
        self._register()
        # embed/final aren't residual writers -> stay symmetric under writers-only.
        self._non_writer_ids = {id(m) for m in (self._embed, self._final) if m is not None}
        self.add_edit("primary", sign=-1.0, default_alpha=1.0,
                      kind="reader" if self.reader_mode else "hidden")
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
        for li, layer in enumerate(b.layers()):
            writers = b.layer_writers(layer)
            ple = tuple(b.ple_writers(layer))
            ple_proj = tuple(b.ple_projection_writers(layer))
            kv = tuple(b.kv_writers(layer))
            query = tuple(b.query_writers(layer))
            self._residual_layers.append(layer)
            self._layer_writers.append(writers)
            for m in writers:  # writer module -> layer index, for per-layer oblique co-vectors
                self._module_layer[id(m)] = li
            self._reader_modules.append(tuple(b.reader_modules(layer)))
            self._kv_writers.append(kv)
            self._query_writers.append(query)
            self._ple_writers.append(ple)
            self._ple_projection_writers.append(ple_proj)
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
        if self.reader_mode:
            for li, mods in enumerate(self._reader_modules):
                for m in mods:
                    self._module_layer[id(m)] = li
                    self._ensure_pre_hook(m)

    def _ensure_hook(self, module: torch.nn.Module):
        mid = id(module)
        if mid in self._hooked:
            return
        self._hooked.add(mid)
        self._handles.append(module.register_forward_hook(self._make_hook(module)))

    def _ensure_pre_hook(self, module: torch.nn.Module):
        mid = id(module)
        if mid in self._pre_hooked:
            return
        self._pre_hooked.add(mid)
        self._pre_handles.append(module.register_forward_pre_hook(self._make_pre_hook(module)))

    def _modules_for_kind(self, kind: str) -> List[torch.nn.Module]:
        if kind == "reader":
            return [m for mods in self._reader_modules for m in mods]
        if kind == "ple_gate":
            return [m for mods in self._ple_writers for m in mods]
        if kind == "ple_residual":
            return [m for mods in self._ple_projection_writers for m in mods]
        if kind == "ple_embed":
            return [self._ple_embed] if self._ple_embed is not None else []
        if kind == "ple_model_projection":
            return [self._ple_model_projection] if self._ple_model_projection is not None else []
        if kind == "residual":
            return self._residual_layers
        if kind == "kv":
            return [m for mods in self._kv_writers for _part, m in mods]
        if kind == "kv_key":
            return [m for mods in self._kv_writers for part, m in mods if part == "k"]
        if kind == "kv_value":
            return [m for mods in self._kv_writers for part, m in mods if part == "v"]
        if kind == "query":
            return [m for mods in self._query_writers for m in mods]
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

    def _cast_field(self, edit: dict, field: str, cache_key: str, dtype, device):
        cache = edit.get(cache_key)
        if cache is None:
            cache = {}
            edit[cache_key] = cache
        key = (dtype, device)
        v = cache.get(key)
        if v is None:
            v = edit[field].to(dtype=dtype, device=device)
            cache[key] = v
        return v

    def _project_pair(self, edit: dict, dtype, device, mod_id=None):
        # (left, right) for removal (t @ right) @ left.t(): oblique = (Rbake, U), else (R, R).
        oblique = edit.get("Rbake") is not None and edit.get("U") is not None
        if oblique and self._oblique_writers_only and mod_id in self._non_writer_ids:
            oblique = False
        if oblique:
            left = self._cast_field(edit, "Rbake", "_rbcast", dtype, device)
            U_layers = edit.get("U_layers")
            if U_layers is not None and mod_id is not None:
                L = self._module_layer.get(mod_id)
                if L is not None and L in U_layers:
                    cache = edit.get("_ucast_layers")
                    if cache is None:
                        cache = {}
                        edit["_ucast_layers"] = cache
                    key = (L, dtype, device)
                    right = cache.get(key)
                    if right is None:
                        right = U_layers[L].to(dtype=dtype, device=device)
                        cache[key] = right
                    return left, right
            right = self._cast_field(edit, "U", "_ucast", dtype, device)
            return left, right
        Rd = self._cast(edit, dtype, device)
        return Rd, Rd

    def _make_hook(self, module):
        mod_id = id(module)

        def hook(_mod, _inp, out):
            if not self.enabled:
                return out
            t = out[0] if isinstance(out, tuple) else out
            delta = None
            for e in self.edits:
                if e.get("kind") == "reader":  # reader edits run on inputs, via pre-hooks
                    continue
                R = e["R"]
                if R is None:
                    continue
                a = e["alpha"].get(mod_id, 0.0)
                if a == 0.0:
                    continue
                if R.shape[0] != t.shape[-1]:
                    continue
                left, right = self._project_pair(e, t.dtype, t.device, mod_id)
                term = (t @ right) @ left.t()
                contrib = (e["sign"] * a) * term
                delta = contrib if delta is None else delta + contrib
            if delta is None:
                return out
            t2 = t + delta
            if isinstance(out, tuple):
                return (t2,) + tuple(out[1:])
            return t2

        return hook

    def _make_pre_hook(self, module):
        # remove the refusal direction from a reader's input (q/k/v/gate/up etc.)
        mod_id = id(module)

        layer = self._module_layer.get(mod_id)

        def pre_hook(_mod, args):
            if not self.enabled or not args:
                return None
            x = args[0]
            if not isinstance(x, torch.Tensor):
                return None
            delta = None
            for e in self.edits:
                if e.get("kind") != "reader":
                    continue
                a = e["alpha"].get(mod_id, 0.0)
                if a == 0.0:
                    continue
                R = self._reader_R(e, layer)  # per-layer direction (falls back to single R)
                if R is None or R.shape[0] != x.shape[-1]:
                    continue
                Rd = R.to(dtype=x.dtype, device=x.device)
                contrib = (e["sign"] * a) * ((x @ Rd) @ Rd.t())
                delta = contrib if delta is None else delta + contrib
            if delta is None:
                return None
            return (x + delta,) + tuple(args[1:])

        return pre_hook

    def _reader_R(self, edit: dict, layer):
        rl = edit.get("R_layers")
        if rl is not None and layer is not None and layer < len(rl) and rl[layer] is not None:
            return rl[layer]
        return edit["R"]

    def add_edit(self, name: str, sign: float, default_alpha: float = 0.0, kind: str = "hidden"):
        alpha = {id(m): default_alpha for m in self._modules_for_kind(kind)}
        self.edits.append({"name": name, "kind": kind, "sign": float(sign), "R": None, "alpha": alpha,
                           "U": None, "Rbake": None})

    def enable_oblique(self, mu: torch.Tensor, strength: float = 1.0, denom_floor: float = 0.2,
                       writers_only: bool = True, predictive: bool = False, harmless=None, ridge: float = 1e-2,
                       harmless_layers=None):
        # store the harmless mean; oblique vectors refresh against whatever R is set later.
        self._oblique_mean = mu.to(self.device).float()
        self._oblique_strength = float(strength)
        self._oblique_floor = float(denom_floor)
        self._oblique_writers_only = bool(writers_only)
        self._oblique_predictive = bool(predictive)
        self._oblique_ridge = float(ridge)
        self._oblique_harmless = harmless.to(self.device).float() if harmless is not None else None
        # per-layer harmless activations (kept on cpu; streamed to gpu per-layer during refresh).
        self._oblique_harmless_layers = (
            [a.detach().float().cpu() if a is not None else None for a in harmless_layers]
            if harmless_layers is not None else None)
        for e in self.edits:
            self._refresh_oblique(e)

    def disable_oblique(self):
        self._oblique_mean = None
        for e in self.edits:
            e["U"] = None
            e["Rbake"] = None
            e["_ucast"] = None
            e["_rbcast"] = None

    def _refresh_oblique(self, edit: dict):
        # primary writer edit only; recomputed whenever R changes (U/Rbake depend on R).
        mu = getattr(self, "_oblique_mean", None)
        edit["_ucast"] = None
        edit["_rbcast"] = None
        edit["_ucast_layers"] = None
        edit["U_layers"] = None
        if mu is None or edit.get("kind") != "hidden" or edit.get("name") != "primary" or edit.get("R") is None:
            edit["U"] = None
            edit["Rbake"] = None
            return
        U = Rbake = None
        predictive = getattr(self, "_oblique_predictive", False)
        ridge = getattr(self, "_oblique_ridge", 1e-2)
        hl_layers = getattr(self, "_oblique_harmless_layers", None)
        if predictive and hl_layers is not None:
            # per-layer D_L = R - W_L so each writer layer preserves ITS OWN harmless variance.
            U_layers = {}
            for L, hl in enumerate(hl_layers):
                if hl is None:
                    continue
                D_L, _ = predictive_covector(edit["R"], hl.to(self.device), ridge)
                if D_L is not None:
                    U_layers[L] = D_L.to(self.device).float()
            if U_layers:
                edit["U_layers"] = U_layers
                U = next(iter(U_layers.values()))  # fallback for any layer without per-layer D
                Rbake = edit["R"]
        elif predictive and getattr(self, "_oblique_harmless", None) is not None:
            U, Rbake = predictive_covector(edit["R"], self._oblique_harmless, ridge)
        if U is None:
            U, Rbake = oblique_vectors(edit["R"], mu, self._oblique_strength, self._oblique_floor)
        edit["U"] = U.to(self.device).float()
        edit["Rbake"] = Rbake.to(self.device).float()

    def _edit(self, name: str) -> dict:
        for e in self.edits:
            if e["name"] == name:
                return e
        raise KeyError(f"no edit named {name!r}")

    def set_edit_subspace(self, name: str, R: torch.Tensor):
        e = self._edit(name)
        e["R"] = R.to(self.device).float()
        e["_cast"] = None
        self._refresh_oblique(e)

    def set_reader_layer_subspace(self, layer_idx: int, R: torch.Tensor, name: str = "primary"):
        # per-layer refusal direction for the reader edit (post-norm models need this)
        e = self._edit(name)
        rl = e.get("R_layers")
        if rl is None:
            rl = [None] * len(self._reader_modules)
            e["R_layers"] = rl
        Rd = R.to(self.device).float()
        rl[layer_idx] = Rd
        if e["R"] is None:
            e["R"] = Rd  # representative so export() doesn't skip the edit

    def get_reader_layer_subspace(self, layer_idx: int, name: str = "primary"):
        rl = self._edit(name).get("R_layers")
        r = rl[layer_idx] if rl is not None else None
        return r.detach().cpu() if r is not None else None

    def _layer_targets(self, edit: dict, layer_idx: int):
        # reader edits act on the layer's readers; everything else on its writers
        if edit.get("kind") == "reader":
            return self._reader_modules[layer_idx]
        return self._layer_writers[layer_idx]

    def set_edit_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        for m in self._layer_targets(e, layer_idx):
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

    def set_edit_ple_residual_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        for m in self._ple_projection_writers[layer_idx]:
            e["alpha"][id(m)] = value

    def set_edit_residual_layer_alpha(self, name: str, layer_idx: int, value: float):
        self._edit(name)["alpha"][id(self._residual_layers[layer_idx])] = value

    def set_edit_kv_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        kind = e.get("kind")
        for part, mod in self._kv_writers[layer_idx]:
            if kind == "kv_key" and part != "k":
                continue
            if kind == "kv_value" and part != "v":
                continue
            e["alpha"][id(mod)] = value

    def set_edit_query_layer_alpha(self, name: str, layer_idx: int, value: float):
        e = self._edit(name)
        for mod in self._query_writers[layer_idx]:
            e["alpha"][id(mod)] = value

    def set_edit_uniform_alpha(self, name: str, value: float):
        e = self._edit(name)
        final_id = id(self._final) if self._final is not None else None
        for k in e["alpha"]:
            if k == final_id:
                continue
            e["alpha"][k] = value

    def get_edit_layer_alpha(self, name: str, layer_idx: int) -> float:
        e = self._edit(name)
        targets = self._layer_targets(e, layer_idx)
        if not targets:
            return 0.0
        return e["alpha"].get(id(targets[0]), 0.0)

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
        self._refresh_oblique(self.edits[0])

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

    def set_residual_subspace(self, R: torch.Tensor):
        self.set_edit_residual_subspace("residual", R)

    def set_edit_residual_subspace(self, name: str, R: torch.Tensor):
        if not any(e["name"] == "residual" for e in self.edits):
            self.add_edit("residual", sign=-1.0, default_alpha=0.0, kind="residual")
        if not any(e["name"] == name for e in self.edits):
            self.add_edit(name, sign=-1.0, default_alpha=0.0, kind="residual")
        for layer in self._residual_layers:
            self._ensure_hook(layer)
        self.set_edit_subspace(name, R)

    def set_edit_kv_subspace(self, name: str, R: torch.Tensor, kind: str):
        if not any(e["name"] == name for e in self.edits):
            self.add_edit(name, sign=-1.0, default_alpha=0.0, kind=kind)
        for mod in self._modules_for_kind(kind):
            self._ensure_hook(mod)
        self.set_edit_subspace(name, R)

    def set_edit_query_subspace(self, name: str, R: torch.Tensor):
        if not any(e["name"] == name for e in self.edits):
            self.add_edit(name, sign=-1.0, default_alpha=0.0, kind="query")
        for mod in self._modules_for_kind("query"):
            self._ensure_hook(mod)
        self.set_edit_subspace(name, R)

    def clear_kv(self):
        for e in self.edits:
            if not str(e.get("kind", "")).startswith("kv"):
                continue
            e["R"] = None
            e["_cast"] = None
            for k in e["alpha"]:
                e["alpha"][k] = 0.0

    def clear_query(self):
        for e in self.edits:
            if e.get("kind") != "query":
                continue
            e["R"] = None
            e["_cast"] = None
            for k in e["alpha"]:
                e["alpha"][k] = 0.0

    def set_residual_layer_alpha(self, layer_idx: int, value: float):
        self.set_edit_residual_layer_alpha("residual", layer_idx, value)

    def get_residual_layer_alpha(self, layer_idx: int) -> float:
        try:
            return self._edit("residual")["alpha"].get(id(self._residual_layers[layer_idx]), 0.0)
        except KeyError:
            return 0.0

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
        for e in self.edits:
            if e.get("kind") != "ple_residual":
                continue
            e["R"] = None
            e["_cast"] = None
            for k in e["alpha"]:
                e["alpha"][k] = 0.0

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

    def set_edit_ple_residual_subspace(self, name: str, R: torch.Tensor):
        if not any(e["name"] == name for e in self.edits):
            self.add_edit(name, sign=-1.0, default_alpha=0.0, kind="ple_residual")
        for mods in self._ple_projection_writers:
            for mod in mods:
                self._ensure_hook(mod)
        self.set_edit_subspace(name, R)

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
        # reader-mode primary has no embed key; tolerate it like the other getters
        return self._edit("primary")["alpha"].get(id(self._embed), 0.0)

    def get_head_alpha(self) -> float:
        if self._final is None:
            return 0.0
        return self._edit("primary")["alpha"].get(id(self._final), 0.0)

    def get_layer_alpha(self, layer_idx: int) -> float:
        return self.get_edit_layer_alpha("primary", layer_idx)

    def alpha_state(self) -> dict:
        return {e["name"]: dict(e["alpha"]) for e in self.edits}

    def set_alpha_state(self, state: dict):
        for e in self.edits:
            vals = state.get(e["name"])
            if vals is not None:
                e["alpha"] = dict(vals)

    def scale_alpha_state(self, state: dict, scale: float, cap: Optional[float] = None):
        for e in self.edits:
            vals = state.get(e["name"])
            if vals is None:
                continue
            out = {}
            for mid, alpha in vals.items():
                value = alpha * scale
                out[mid] = min(cap, value) if cap is not None else value
            e["alpha"] = out

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
        for h in self._handles + self._pre_handles:
            h.remove()
        self._handles = []
        self._pre_handles = []
        self._hooked = set()
        self._pre_hooked = set()

    def export(self) -> dict:
        out_edits = []
        for e in self.edits:
            if e["R"] is None:
                continue
            if e.get("kind") == "reader":
                layer_alphas = [
                    e["alpha"].get(id(self._reader_modules[i][0]), 0.0) if self._reader_modules[i] else 0.0
                    for i in range(len(self._reader_modules))
                ]
                embed_alpha = 0.0
                head_alpha = 0.0
                rl = e.get("R_layers")
                if rl is not None:
                    out_edits.append({
                        "name": e["name"], "kind": "reader", "sign": e["sign"],
                        "R": e["R"].detach().cpu(),
                        "R_layers": [(r.detach().cpu() if r is not None else None) for r in rl],
                        "embed_alpha": 0.0, "head_alpha": 0.0, "layer_alphas": layer_alphas,
                    })
                    continue
            elif e.get("kind") == "ple_gate":
                layer_alphas = [
                    e["alpha"].get(id(self._ple_writers[i][0]), 0.0) if self._ple_writers[i] else 0.0
                    for i in range(len(self._ple_writers))
                ]
                embed_alpha = 0.0
                head_alpha = 0.0
            elif e.get("kind") == "ple_residual":
                layer_alphas = [
                    e["alpha"].get(id(self._ple_projection_writers[i][0]), 0.0)
                    if self._ple_projection_writers[i] else 0.0
                    for i in range(len(self._ple_projection_writers))
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
            elif e.get("kind") == "residual":
                layer_alphas = [
                    e["alpha"].get(id(self._residual_layers[i]), 0.0)
                    for i in range(len(self._residual_layers))
                ]
                embed_alpha = 0.0
                head_alpha = 0.0
            elif str(e.get("kind", "")).startswith("kv"):
                layer_alphas = []
                for i, mods in enumerate(self._kv_writers):
                    vals = [e["alpha"].get(id(m), 0.0) for _part, m in mods]
                    layer_alphas.append(max(vals, key=abs) if vals else 0.0)
                embed_alpha = 0.0
                head_alpha = 0.0
            elif e.get("kind") == "query":
                layer_alphas = []
                for mods in self._query_writers:
                    vals = [e["alpha"].get(id(m), 0.0) for m in mods]
                    layer_alphas.append(max(vals, key=abs) if vals else 0.0)
                embed_alpha = 0.0
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
                "U": e["U"].detach().cpu() if e.get("U") is not None else None,
                "Rbake": e["Rbake"].detach().cpu() if e.get("Rbake") is not None else None,
                "U_layers": ([
                    e["U_layers"][i].detach().cpu() if i in e["U_layers"] else None
                    for i in range(len(self._layer_writers))
                ] if e.get("U_layers") else None),
                "oblique_writers_only": self._oblique_writers_only,
            })
        return {"edits": out_edits}
