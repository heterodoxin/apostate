# the pipeline: load, collect, build directions, search, guard, de-escalate KL, bake.

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from typing import Optional

import torch

from .config import ApostateConfig
from .model import load_model
from .data import resolve_prompts
from .activations import (
    collect_activations,
    collect_kv_activations,
    collect_query_activations,
    collect_response_kv_activations,
    collect_response_query_activations,
    collect_response_activations,
    collect_ple_gate_activations,
    collect_ple_projection_activations,
    collect_ple_embed_activations,
)
from .directions import refusal_subspace, preservation_subspace, gram_schmidt_remove, separation, causal_layer_scores
from .projectors import ProjectionController
from .guard import run_guard, run_reader_guard
from .evaluate import (
    generate,
    judge_strict_refusal,
    refusal_logit_margin,
    strict_refusal_rate as refusal_rate,
    strict_refusal_rate_bounded as refusal_rate_bounded,
    kl_harmless,
)
from .optimize import optimize_profile, _head_token_subspace, _has_optuna
from .bake import bake
from .reports import write_model_card, write_run_report


def _log(msg: str):
    print(f"[apostate] {msg}", flush=True)


def _prompt_hash(instructions) -> str:
    h = hashlib.sha256()
    for p in instructions:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def _activation_cache_dir(cfg: ApostateConfig) -> str:
    return cfg.activation_cache_dir or os.path.join(cfg.output_dir, "activation_cache")


def _cached_collect(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    if not cfg.cache_activations:
        return collect_activations(bundle, instructions, batch_size, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "hidden_size": bundle.hidden_size,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if acts is not None and acts.shape[:1] == (bundle.num_layers,):
                    _log(f"activation cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"activation cache ignored for {name}: {e}")
    acts = collect_activations(bundle, instructions, batch_size, preformatted=preformatted)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"activation cache saved: {name}")
    except Exception as e:
        _log(f"activation cache save failed for {name}: {e}")
    return acts


def _cached_collect_ple(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    if not cfg.cache_activations:
        return collect_ple_gate_activations(bundle, instructions, batch_size, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "ple_gate",
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if acts is not None and acts.shape[:1] == (bundle.num_layers,):
                    _log(f"ple cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"ple cache ignored for {name}: {e}")
    acts = collect_ple_gate_activations(bundle, instructions, batch_size, preformatted=preformatted)
    if acts is None:
        return None
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"ple cache saved: {name}")
    except Exception as e:
        _log(f"ple cache save failed for {name}: {e}")
    return acts


def _cached_collect_ple_projection(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    if not cfg.cache_activations:
        return collect_ple_projection_activations(bundle, instructions, batch_size, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "ple_projection",
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "hidden_size": bundle.hidden_size,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if acts is not None and acts.shape[:1] == (bundle.num_layers,):
                    _log(f"ple residual cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"ple residual cache ignored for {name}: {e}")
    acts = collect_ple_projection_activations(bundle, instructions, batch_size, preformatted=preformatted)
    if acts is None:
        return None
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"ple residual cache saved: {name}")
    except Exception as e:
        _log(f"ple residual cache save failed for {name}: {e}")
    return acts


def _cached_collect_response(bundle, instructions, responses, batch_size: int, cfg: ApostateConfig, name: str):
    if not cfg.cache_activations:
        return collect_response_activations(bundle, instructions, responses, batch_size)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "response_mean",
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "prompt_hash": _prompt_hash(instructions),
        "response_hash": _prompt_hash(responses),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if acts is not None and acts.shape[:1] == (bundle.num_layers,):
                    _log(f"response cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"response cache ignored for {name}: {e}")
    acts = collect_response_activations(bundle, instructions, responses, batch_size)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"response cache saved: {name}")
    except Exception as e:
        _log(f"response cache save failed for {name}: {e}")
    return acts


def _cached_collect_kv(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    layers = bundle.kv_source_layers()
    if not cfg.cache_activations:
        return collect_kv_activations(bundle, instructions, batch_size, layers=layers, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "kv_mean",
        "model": cfg.model,
        "layers": layers,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if isinstance(acts, dict):
                    _log(f"kv cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"kv cache ignored for {name}: {e}")
    acts = collect_kv_activations(bundle, instructions, batch_size, layers=layers, preformatted=preformatted)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"kv cache saved: {name}")
    except Exception as e:
        _log(f"kv cache save failed for {name}: {e}")
    return acts


def _cached_collect_response_kv(bundle, instructions, responses, batch_size: int, cfg: ApostateConfig, name: str):
    layers = bundle.kv_source_layers()
    if not cfg.cache_activations:
        return collect_response_kv_activations(bundle, instructions, responses, batch_size, layers=layers)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "response_kv_mean",
        "model": cfg.model,
        "layers": layers,
        "prompt_hash": _prompt_hash(instructions),
        "response_hash": _prompt_hash(responses),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if isinstance(acts, dict):
                    _log(f"response kv cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"response kv cache ignored for {name}: {e}")
    acts = collect_response_kv_activations(bundle, instructions, responses, batch_size, layers=layers)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"response kv cache saved: {name}")
    except Exception as e:
        _log(f"response kv cache save failed for {name}: {e}")
    return acts


def _cached_collect_query(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    layers = bundle.query_layer_candidates()
    if not cfg.cache_activations:
        return collect_query_activations(bundle, instructions, batch_size, layers=layers, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "query_mean",
        "model": cfg.model,
        "layers": layers,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if isinstance(acts, dict):
                    _log(f"query cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"query cache ignored for {name}: {e}")
    acts = collect_query_activations(bundle, instructions, batch_size, layers=layers, preformatted=preformatted)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"query cache saved: {name}")
    except Exception as e:
        _log(f"query cache save failed for {name}: {e}")
    return acts


def _cached_collect_response_query(bundle, instructions, responses, batch_size: int, cfg: ApostateConfig, name: str):
    layers = bundle.query_layer_candidates()
    if not cfg.cache_activations:
        return collect_response_query_activations(bundle, instructions, responses, batch_size, layers=layers)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "response_query_mean",
        "model": cfg.model,
        "layers": layers,
        "prompt_hash": _prompt_hash(instructions),
        "response_hash": _prompt_hash(responses),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if isinstance(acts, dict):
                    _log(f"response query cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"response query cache ignored for {name}: {e}")
    acts = collect_response_query_activations(bundle, instructions, responses, batch_size, layers=layers)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"response query cache saved: {name}")
    except Exception as e:
        _log(f"response query cache save failed for {name}: {e}")
    return acts


def _cached_collect_ple_embed(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str, preformatted: bool = False):
    if not cfg.cache_activations:
        return collect_ple_embed_activations(bundle, instructions, batch_size, preformatted=preformatted)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "kind": "ple_embed",
        "model": cfg.model,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.pt")
    if cfg.resume and os.path.isfile(path):
        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("meta") == meta:
                acts = obj.get("activations")
                if acts is not None:
                    _log(f"ple embed cache hit: {name}")
                    return acts
        except Exception as e:
            _log(f"ple embed cache ignored for {name}: {e}")
    acts = collect_ple_embed_activations(bundle, instructions, batch_size, preformatted=preformatted)
    if acts is None:
        return None
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"ple embed cache saved: {name}")
    except Exception as e:
        _log(f"ple embed cache save failed for {name}: {e}")
    return acts


def _cached_generate(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str):
    if not cfg.cache_activations:
        return generate(bundle, instructions, cfg.fit_response_tokens, batch_size)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "name": name,
        "model": cfg.model,
        "prompt_hash": _prompt_hash(instructions),
        "count": len(instructions),
        "tokens": cfg.fit_response_tokens,
    }
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{name}-{key}.json")
    if cfg.resume and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if obj.get("meta") == meta and len(obj.get("responses", [])) == len(instructions):
                _log(f"response cache hit: {name}")
                return obj["responses"]
        except Exception as e:
            _log(f"response cache ignored for {name}: {e}")
    responses = generate(bundle, instructions, cfg.fit_response_tokens, batch_size)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "responses": responses}, f)
        _log(f"response cache saved: {name}")
    except Exception as e:
        _log(f"response cache save failed for {name}: {e}")
    return responses


def _response_fit_activations(bundle, fit_harmful, harmless, cfg):
    n = max(1, min(cfg.fit_response_n, len(fit_harmful), len(harmless)))
    hprompts = fit_harmful[:n]
    lprompts = harmless[:n]
    _log(f"collecting response activations ({n}) ...")
    hresp = _cached_generate(bundle, hprompts, cfg.batch_size, cfg, "fit_harmful_responses")
    lresp = _cached_generate(bundle, lprompts, cfg.batch_size, cfg, "fit_harmless_responses")
    ah = _cached_collect_response(bundle, hprompts, hresp, cfg.batch_size, cfg, "fit_harmful_response_mean")
    al = _cached_collect_response(bundle, lprompts, lresp, cfg.batch_size, cfg, "fit_harmless_response_mean")
    return ah, al, hprompts, hresp, lprompts, lresp


def _cached_prompts(cfg: ApostateConfig, name: str, spec: str, n: int, seed: int):
    if not cfg.cache_activations:
        return resolve_prompts(spec, n, seed)
    cache_dir = _activation_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta = {"name": name, "spec": spec, "count": n, "seed": seed}
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    path = os.path.join(cache_dir, f"prompts-{name}-{key}.json")
    if cfg.resume and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            prompts = obj.get("prompts", [])
            if obj.get("meta") == meta and len(prompts) <= n:
                _log(f"prompt cache hit: {name}")
                return prompts[:n]
        except Exception as e:
            _log(f"prompt cache ignored for {name}: {e}")
    prompts = resolve_prompts(spec, n, seed)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "prompts": prompts}, f)
        _log(f"prompt cache saved: {name}")
    except Exception as e:
        _log(f"prompt cache save failed for {name}: {e}")
    return prompts


def _refusal_calibration_prompts() -> list[str]:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "refusal_calibration.txt")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _append_unique(items: list[str], extra: list[str]) -> list[str]:
    seen = set(items)
    out = list(items)
    for item in extra:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _persist_reports(cfg: ApostateConfig, report: dict, command: Optional[str]):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_run_report(cfg, report, command=command)
    write_model_card(cfg, report, command=command)


def _preservation_lookup(acts: Optional[torch.Tensor], rank: int):
    cache: dict = {}

    def lookup(layer_idx: int):
        if acts is None or rank <= 0:
            return None
        layer_idx = int(layer_idx)
        if layer_idx not in cache:
            cache[layer_idx] = preservation_subspace(acts[layer_idx], rank=rank)
        return cache[layer_idx]

    return lookup


def _combined_preservation_lookup(sources):
    # sources: list of (acts, rank). merge per-layer preservation subspaces (harmless + the
    # capability subspace), orthonormalized so overlapping directions collapse. ablating the
    # refusal direction then leaves both the harmless mean variance and the math/code
    # reasoning subspace untouched -- the superposition fix.
    cache: dict = {}

    def lookup(layer_idx: int):
        layer_idx = int(layer_idx)
        if layer_idx in cache:
            return cache[layer_idx]
        bases = []
        for acts, rank in sources:
            if acts is not None and rank > 0:
                b = preservation_subspace(acts[layer_idx], rank=rank)
                if b is not None and b.numel():
                    bases.append(b)
        if not bases:
            out = None
        elif len(bases) == 1:
            out = bases[0]
        else:
            M = torch.cat(bases, dim=1)
            Q, Rm = torch.linalg.qr(M)
            out = Q[:, torch.abs(torch.diagonal(Rm)) > 1e-6]
        cache[layer_idx] = out
        return out

    return lookup


def _kv_preservation_lookup(acts: Optional[dict], rank: int):
    cache: dict = {}

    def lookup(part: str, layer_idx: int):
        if not acts or rank <= 0:
            return None
        layer_idx = int(layer_idx)
        if part not in acts or layer_idx not in acts[part]:
            return None
        key = (part, layer_idx)
        if key not in cache:
            cache[key] = preservation_subspace(acts[part][layer_idx], rank=rank)
        return cache[key]

    return lookup


def _dict_preservation_lookup(acts: Optional[dict], rank: int):
    cache: dict = {}

    def lookup(layer_idx: int):
        if not acts or rank <= 0:
            return None
        layer_idx = int(layer_idx)
        if layer_idx not in acts:
            return None
        if layer_idx not in cache:
            cache[layer_idx] = preservation_subspace(acts[layer_idx], rank=rank)
        return cache[layer_idx]

    return lookup


def _alpha_state(controller):
    if hasattr(controller, "alpha_state"):
        return controller.alpha_state()
    return {"primary": dict(controller.alpha)}


def _set_alpha_state(controller, state):
    if hasattr(controller, "set_alpha_state"):
        controller.set_alpha_state(state)
    else:
        controller.alpha = dict(state["primary"])


def _scale_alpha_state(controller, state, scale: float, cap: Optional[float] = None):
    if hasattr(controller, "scale_alpha_state"):
        controller.scale_alpha_state(state, scale, cap)
        return
    out = {}
    for mid, alpha in state["primary"].items():
        value = alpha * scale
        out[mid] = min(cap, value) if cap is not None else value
    controller.alpha = out


def _refine_refusal(bundle, controller, cfg, eval_harmful, eval_harmless):
    base = _alpha_state(controller)
    es = eval_harmful[: max(48, cfg.opt_eval_n)]
    el = eval_harmless[: cfg.opt_eval_n]
    with controller.active():
        ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
    kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
    if ref <= cfg.target_refusal:
        if not cfg.refine_deescalate:
            return ref, kl
        best = (ref, kl, _alpha_state(controller))
        for s in (0.9, 0.8, 0.7):
            _scale_alpha_state(controller, base, s)
            with controller.active():
                new_ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
            if new_ref > cfg.target_refusal:
                break
            new_kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
            best = (new_ref, new_kl, _alpha_state(controller))
            _log(f"  refine(down): scale={s:.2f} refusal={new_ref:.3f} kl={new_kl:.3f} (kept)")
        _set_alpha_state(controller, best[2])
        return best[0], best[1]
    best = (ref, kl, _alpha_state(controller))
    step = (cfg.refine_max_scale - 1.0) / max(1, cfg.refine_steps)
    candidates = []
    for i in range(cfg.refine_steps, 0, -1):
        s = 1.0 + step * i
        _scale_alpha_state(controller, base, s, cap=cfg.refine_max_scale)
        new_kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
        if new_kl <= cfg.max_kl:
            candidates.append((s, new_kl, _alpha_state(controller)))

    for s, new_kl, alpha in candidates[: max(1, cfg.refine_scale_rerank_k * 3)]:
        _set_alpha_state(controller, alpha)
        with controller.active():
            new_ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
        if new_ref < best[0] - 1e-6:
            best = (new_ref, new_kl, _alpha_state(controller))
            _log(f"  refine: scale={s:.2f} refusal={new_ref:.3f} kl={new_kl:.3f} (kept)")
            if new_ref <= cfg.target_refusal:
                break
    _set_alpha_state(controller, best[2])
    return best[0], best[1]


def _minimize_kl_scale(bundle, controller, cfg, eval_harmful, eval_harmless):
    base_alpha = _alpha_state(controller)
    best_alpha = base_alpha
    best_ref = None
    best_kl = None
    lo, hi = 0.0, 1.0
    target = cfg.target_refusal + cfg.refine_refusal_slack

    def _apply(s: float):
        _scale_alpha_state(controller, base_alpha, s)

    for _ in range(cfg.refine_kl_steps):
        mid = 0.5 * (lo + hi)
        _apply(mid)
        with controller.active():
            ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        if ref <= target:
            hi = mid
            best_alpha = _alpha_state(controller)
            best_ref, best_kl = ref, kl
        else:
            lo = mid

    _set_alpha_state(controller, best_alpha)
    if best_ref is None:
        _apply(1.0)
        with controller.active():
            best_ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        best_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
    return best_ref, best_kl


def _alpha_get(controller, item: int) -> float:
    if item == -3:
        return controller.get_head_token_alpha()
    if item == -2:
        return controller.get_head_alpha()
    if item == -1:
        return controller.get_embed_alpha()
    return controller.get_layer_alpha(item)


def _alpha_set(controller, item: int, value: float):
    if item == -3:
        controller.set_head_token_alpha(value)
    elif item == -2:
        controller.set_head_alpha(value)
    elif item == -1:
        controller.set_embed_alpha(value)
    else:
        controller.set_layer_alpha(item, value)


def _alpha_label(item: int) -> str:
    if item == -3:
        return "head_token"
    if item == -2:
        return "head"
    return "embed" if item == -1 else f"L{item}"


def _minimize_kl_layers(bundle, controller, cfg, eval_harmful, eval_harmless):
    target = cfg.target_refusal + cfg.refine_refusal_slack
    hset = eval_harmful[: max(24, cfg.opt_eval_n)]
    lset = eval_harmless[: max(48, cfg.opt_eval_n)]
    best_alpha = _alpha_state(controller)
    with controller.active():
        best_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
    best_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    if best_ref > target or best_kl <= cfg.kl_target:
        return best_ref, best_kl, 0

    items = [-3, -2, -1] + list(range(controller.num_layers))
    kept = 0
    scales = (0.75, 0.50, 0.25, 0.0)
    for step in range(cfg.refine_kl_layer_steps):
        _set_alpha_state(controller, best_alpha)
        scored = []
        for item in items:
            a = _alpha_get(controller, item)
            if abs(a) < 1e-6:
                continue
            _alpha_set(controller, item, 0.0)
            trial_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
            _alpha_set(controller, item, a)
            drop = best_kl - trial_kl
            if drop > 1e-4:
                scored.append((drop, item))
        if not scored:
            break
        scored.sort(reverse=True)

        accepted = None
        for _, item in scored[: cfg.refine_kl_layer_candidates]:
            _set_alpha_state(controller, best_alpha)
            start = _alpha_get(controller, item)
            for scale in scales:
                _set_alpha_state(controller, best_alpha)
                _alpha_set(controller, item, start * scale)
                trial_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
                if trial_kl >= best_kl - 1e-4:
                    continue
                with controller.active():
                    trial_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
                if trial_ref <= target and (accepted is None or trial_kl < accepted[0]):
                    accepted = (trial_kl, trial_ref, item, scale, _alpha_state(controller))
            if accepted is not None and accepted[0] <= cfg.kl_target:
                break

        if accepted is None:
            break
        best_kl, best_ref, item, scale, best_alpha = accepted
        kept += 1
        _log(f"  kl trim {step + 1}: {_alpha_label(item)} x{scale:.2f} refusal={best_ref:.3f} kl={best_kl:.3f}")
        if best_kl <= cfg.kl_target:
            break

    _set_alpha_state(controller, best_alpha)
    return best_ref, best_kl, kept


def _repair_loss(ref: float, kl: float, cfg) -> float:
    ref_over = max(0.0, ref - cfg.target_refusal)
    if ref_over > 0.0:
        kl_budget_over = max(0.0, kl - cfg.max_kl)
        return (
            ref
            + cfg.refusal_target_weight * 2.5 * ref_over
            + cfg.refusal_quad_weight * 3.0 * ref_over * ref_over
            + cfg.kl_weight * 0.35 * kl
            + cfg.kl_over_budget_weight * kl_budget_over
        )
    kl_target_over = max(0.0, kl - cfg.kl_target)
    kl_budget_over = max(0.0, kl - cfg.max_kl)
    return (
        ref
        + cfg.refusal_target_weight * ref_over
        + cfg.refusal_quad_weight * ref_over * ref_over
        + cfg.kl_weight * kl
        + cfg.kl_target_weight * kl_target_over
        + cfg.kl_quad_weight * kl * kl
        + cfg.kl_over_budget_weight * kl_budget_over
    )


def _margin_refusal(bundle, instructions, batch_size: int) -> float:
    margin = refusal_logit_margin(bundle, instructions, batch_size)
    return 1.0 / (1.0 + math.exp(-margin))


def _head_sweep_enabled(bundle, cfg) -> bool:
    return bool(
        cfg.optimize and cfg.head_sweep
        and not bundle.can_edit_embed()
        and not bundle.has_ple()
        and bundle.final_norm() is not None
        and bundle.lm_head() is not None
    )


def _head_params(alpha: float) -> dict:
    return {
        "direction_source": "head_tokens",
        "direction_layer_frac": 0.58,
        "refusal_rank": 1,
        "strength": 0.0,
        "band_center": 0.58,
        "band_width": 0.78,
        "causal_mix": 0.0,
        "causal_power": 1.0,
        "direction_sign": 1.0,
        "ablate_embed": False,
        "ablate_head": True,
        "head_scale": 0.0,
        "head_alpha": float(alpha),
    }


def _head_alphas(cfg) -> list[float]:
    vals = set()
    x = float(cfg.head_sweep_min)
    hi = float(cfg.head_sweep_max)
    step = max(0.01, float(cfg.head_sweep_step))
    while x <= hi + 1e-9:
        vals.add(round(x, 4))
        x += step
    for x in (4.65, 4.85, 5.15):
        if float(cfg.head_sweep_min) <= x <= hi:
            vals.add(x)
    return sorted(vals)


def _prepare_head_profile(bundle, controller):
    controller.set_subspace(_head_token_subspace(bundle))
    for L in range(controller.num_layers):
        controller.set_layer_alpha(L, 0.0)
    controller.set_embed_alpha(0.0)


def _apply_head_alpha(controller, alpha: float):
    controller.set_head_alpha(float(alpha))


def _head_token_sweep(bundle, controller, cfg, eval_harmful, eval_harmless):
    _prepare_head_profile(bundle, controller)
    hprobe = eval_harmful[: max(4, min(len(eval_harmful), cfg.head_sweep_probe_n))]
    lprobe = eval_harmless[: max(8, min(len(eval_harmless), cfg.head_sweep_probe_n))]
    hset = eval_harmful[: max(8, min(len(eval_harmful), cfg.head_sweep_eval_n))]
    hset = _append_unique(hset, _refusal_calibration_prompts())
    lset = eval_harmless[: max(16, min(len(eval_harmless), cfg.head_sweep_eval_n))]
    rows = []
    best_score = float("inf")
    alphas = _head_alphas(cfg)
    for idx, alpha in enumerate(alphas, 1):
        _apply_head_alpha(controller, alpha)
        kl = kl_harmless(bundle, controller, lprobe, cfg.batch_size, positions=cfg.kl_positions)
        if cfg.head_sweep_probe_classifier:
            with controller.active():
                ref, complete = refusal_rate_bounded(
                    bundle, hprobe, cfg.opt_gen_tokens, cfg.batch_size,
                    should_stop=lambda floor, _seen, _total: _repair_loss(floor, kl, cfg) >= best_score - 1e-4,
                )
            scored_ref = min(1.0, ref + (0.15 if not complete else 0.0))
        else:
            with controller.active():
                ref = _margin_refusal(bundle, hprobe, cfg.batch_size)
            complete = True
            scored_ref = ref
        score = _repair_loss(scored_ref, kl, cfg)
        best_score = min(best_score, score)
        row = {
            "params": _head_params(alpha),
            "value": score,
            "refusal_proxy": round(ref, 4),
            "refusal_complete": complete,
            "kl": round(kl, 4),
        }
        rows.append(row)
        label = "refusal" if cfg.head_sweep_probe_classifier else "proxy"
        _log(f"  head {idx}/{len(alphas)}: alpha={alpha:.3f} {label}={ref:.3f} kl={kl:.3f}")

    exact = []
    top_n = max(1, cfg.head_sweep_top_k)
    seen = set()
    top = []

    def add_top(candidates, cap):
        for row in candidates:
            alpha = row["params"]["head_alpha"]
            if alpha in seen:
                continue
            seen.add(alpha)
            top.append(row)
            if len(top) >= cap:
                break

    anchors = [row for row in rows if row["params"]["head_alpha"] in (4.65, 4.85, 5.0)]
    add_top(anchors, min(top_n, len(anchors)))
    add_top(sorted(rows, key=lambda h: (h["value"], h["kl"])), max(1, top_n // 2))
    add_top(sorted(rows, key=lambda h: (
        h.get("refusal_proxy", 1.0) + (0.15 if h.get("refusal_complete") is False else 0.0),
        h["kl"],
    )), top_n)
    for idx, row in enumerate(top, 1):
        alpha = row["params"]["head_alpha"]
        _apply_head_alpha(controller, alpha)
        with controller.active():
            ref = refusal_rate(bundle, hset, cfg.opt_gen_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
        score = _repair_loss(ref, kl, cfg)
        item = {
            "params": _head_params(alpha),
            "value": score,
            "refusal": round(ref, 4),
            "refusal_complete": True,
            "kl": round(kl, 4),
        }
        exact.append(item)
        _log(f"  head exact {idx}/{len(top)}: alpha={alpha:.3f} refusal={ref:.3f} kl={kl:.3f}")

    exact_feasible = [h for h in exact if h.get("kl", 99.0) <= cfg.max_kl]
    if exact_feasible:
        best_ref = min(h.get("refusal", 1.0) for h in exact_feasible)
        ref_tol = max(0.005, float(cfg.refine_refusal_slack))
        near = [h for h in exact_feasible if h.get("refusal", 1.0) <= best_ref + ref_tol]
        if best_ref > cfg.target_refusal + ref_tol:
            best = min(near, key=lambda h: (-h["params"]["head_alpha"], h["kl"], h["value"]))
        else:
            best = min(near, key=lambda h: (h["kl"], h["value"]))
    else:
        best = min(exact or rows, key=lambda h: (h["value"], h["kl"]))
    _apply_head_alpha(controller, best["params"]["head_alpha"])
    attrs = {
        k: best[k]
        for k in ("refusal_proxy", "refusal", "kl")
        if k in best
    }
    return best["params"], attrs, rows + exact


def _repair_scales(best_ref: float, best_kl: float, cfg):
    target = cfg.target_refusal + cfg.refine_refusal_slack
    if best_ref > target:
        return (1.20, 1.45, 1.75, 2.10, 2.50)
    if best_kl > cfg.kl_target and best_ref <= target:
        return (0.0, 0.25, 0.50, 0.75, 0.90)
    return (0.0, 0.50, 0.75, 0.90, 1.10, 1.25)


def _repair_priority(start: float, scale: float, best_ref: float, best_kl: float, cfg) -> float:
    shrink = max(0.0, 1.0 - scale)
    grow = max(0.0, scale - 1.0)
    kl_need = max(0.0, best_kl - cfg.kl_target) + 2.0 * max(0.0, best_kl - cfg.max_kl)
    ref_need = max(0.0, best_ref - cfg.target_refusal)
    return abs(start) * (1.0 + shrink + grow + 8.0 * kl_need * shrink + 8.0 * ref_need * grow)


def _repair_alphas(bundle, controller, cfg, eval_harmful, eval_harmless, start_ref=None, start_kl=None):
    hset = eval_harmful[: max(24, cfg.repair_eval_n)]
    lset = eval_harmless[: max(48, cfg.repair_kl_n)]
    hprobe = hset[: max(4, min(len(hset), cfg.repair_probe_ref_n))]
    lprobe = lset[: max(8, min(len(lset), cfg.repair_probe_kl_n))]
    probe_positions = max(4, min(cfg.kl_positions, cfg.repair_probe_positions))
    if start_ref is None:
        with controller.active():
            best_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
    else:
        best_ref = start_ref
    if start_kl is None:
        best_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    else:
        best_kl = start_kl
    best_alpha = _alpha_state(controller)
    best_score = _repair_loss(best_ref, best_kl, cfg)
    steps = 0
    min_alpha = getattr(cfg, "repair_min_alpha", 1e-3)
    min_kl_gain = getattr(cfg, "repair_min_kl_gain", 0.003)
    min_ref_gain = getattr(cfg, "repair_min_refusal_gain", 0.005)
    min_score_gain = getattr(cfg, "repair_min_score_gain", 0.01)
    target = cfg.target_refusal + cfg.refine_refusal_slack

    for step in range(cfg.repair_steps):
        _set_alpha_state(controller, best_alpha)
        levels = [
            (item, abs(_alpha_get(controller, item)))
            for item in ([-3, -2, -1] + list(range(controller.num_layers)))
        ]
        max_level = max((v for _item, v in levels), default=0.0)
        floor = max(min_alpha, max_level * 0.01)
        active = [
            item for item, value in levels
            if value >= floor or (item == -2 and value > 1e-6)
        ]
        active.sort(key=lambda item: abs(_alpha_get(controller, item)), reverse=True)
        items = active[: cfg.repair_candidates]
        if not items:
            break

        scales = _repair_scales(best_ref, best_kl, cfg)
        candidates = []
        for item in items:
            start = _alpha_get(controller, item)
            for scale in scales:
                value = start * scale
                if abs(value - start) < 1e-6:
                    continue
                priority = _repair_priority(start, scale, best_ref, best_kl, cfg)
                candidates.append((priority, item, scale, value))
        if best_ref > target:
            for item, values in (
                (-1, (0.08, 0.14, 0.22, 0.32, 0.45)),
                (-2, (0.20, 0.45, 0.80, 1.20, 1.60)),
            ):
                start = _alpha_get(controller, item)
                for value in values:
                    if value <= start + 1e-6:
                        continue
                    scale = value / max(abs(start), 1.0)
                    priority = _repair_priority(max(abs(start), value), scale, best_ref, best_kl, cfg)
                    candidates.append((priority, item, scale, value))
        probe_n = max(1, cfg.repair_probe_candidates)
        exact_n = max(1, cfg.repair_rerank_k)
        if cfg.target_refusal <= 0.0 and best_ref > 0.0:
            probe_n = max(probe_n, 36)
            exact_n = max(exact_n, 12)
        candidates.sort(reverse=True)
        candidates = candidates[:probe_n]
        _log(
            f"  repair {step + 1}/{cfg.repair_steps}: "
            f"probe={len(candidates)} exact={exact_n} "
            f"refusal={best_ref:.3f} kl={best_kl:.3f}"
        )

        cheap = []
        for _priority, item, scale, value in candidates:
            _set_alpha_state(controller, best_alpha)
            _alpha_set(controller, item, value)
            trial_kl = kl_harmless(bundle, controller, lprobe, cfg.batch_size, positions=probe_positions)
            with controller.active():
                proxy_ref = _margin_refusal(bundle, hprobe, cfg.batch_size)
            proxy_score = _repair_loss(proxy_ref, trial_kl, cfg)
            cheap.append((proxy_score, trial_kl, item, scale, _alpha_state(controller)))

        accepted = None
        skipped = 0
        cheap.sort(key=lambda x: (x[0], x[1]))
        for _proxy_score, _probe_kl, item, scale, alpha in cheap[:exact_n]:
            _set_alpha_state(controller, alpha)
            trial_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
            if best_ref <= cfg.target_refusal and _repair_loss(0.0, trial_kl, cfg) >= best_score - 1e-4:
                skipped += 1
                continue
            with controller.active():
                trial_ref, complete = refusal_rate_bounded(
                    bundle, hset, cfg.max_new_tokens, cfg.batch_size,
                    should_stop=lambda floor, _seen, _total: _repair_loss(floor, trial_kl, cfg) >= best_score - 1e-4,
                )
            if not complete:
                skipped += 1
                continue
            if trial_ref > best_ref + cfg.repair_refusal_regress_slack and trial_kl <= cfg.max_kl:
                skipped += 1
                continue
            trial_score = _repair_loss(trial_ref, trial_kl, cfg)
            ref_gain = best_ref - trial_ref
            kl_gain = best_kl - trial_kl
            score_gain = best_score - trial_score
            if ref_gain < min_ref_gain and kl_gain < min_kl_gain and score_gain < min_score_gain:
                skipped += 1
                continue
            if trial_score < best_score - 1e-4:
                if accepted is None or trial_score < accepted[0]:
                    accepted = (trial_score, trial_ref, trial_kl, item, scale, _alpha_state(controller))

        if accepted is None:
            suffix = f" ({skipped} skipped)" if skipped else ""
            _log(f"  repair {step + 1}: no better exact candidate{suffix}")
            break
        best_score, best_ref, best_kl, item, scale, best_alpha = accepted
        steps += 1
        _log(f"  repair {step + 1}: {_alpha_label(item)} x{scale:.2f} refusal={best_ref:.3f} kl={best_kl:.3f}")
        if best_ref <= cfg.target_refusal and best_kl <= cfg.max_kl * cfg.repair_stop_kl_frac:
            break
        if best_ref <= cfg.target_refusal and best_kl <= cfg.kl_target:
            break

    _set_alpha_state(controller, best_alpha)
    return best_ref, best_kl, steps


def _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless):
    base_alpha = _alpha_state(controller)

    def apply_scale(s: float):
        _scale_alpha_state(controller, base_alpha, s)

    lo, hi = 0.0, 1.0
    steps = 0
    for steps in range(1, cfg.refine_kl_steps + 1):
        mid = 0.5 * (lo + hi)
        apply_scale(mid)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        if kl > cfg.max_kl:
            hi = mid
        else:
            lo = mid
        _log(f"  backoff {steps}: scale={mid:.3f} KL={kl:.3f}")
    apply_scale(lo)
    kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
    with controller.active():
        ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
    _log(f"  backoff final: scale={lo:.3f} KL={kl:.3f} refusal={ref:.3f}")
    return ref, kl, steps


def _push_refusal_scale(bundle, controller, cfg, eval_harmful, eval_harmless):
    base = _alpha_state(controller)
    best = None
    zero_over = None
    prev_scale = 1.0
    prev_ref = None
    best_ref_seen = 1.0
    no_improve = 0
    for scale in (1.10, 1.25, 1.40, 1.60, 1.85, 2.10):
        _scale_alpha_state(controller, base, scale)
        with controller.active():
            ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        # bail when refusal has plateaued and we are already over the kl budget: cranking
        # alpha further only raises kl without reaching target (the no-head / over-preserved
        # case). avoids sweeping every scale on the big test set for nothing.
        if ref < best_ref_seen - 0.005:
            best_ref_seen = ref
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 2 and ref > cfg.target_refusal and kl > cfg.max_kl:
            _log(f"  final push: refusal plateaued at {ref:.3f} over kl budget, stopping")
            break
        if kl <= cfg.max_kl and (best is None or (ref, kl) < (best[0], best[1])):
            best = (ref, kl, _alpha_state(controller), scale)
        if ref <= cfg.target_refusal and (zero_over is None or kl < zero_over[1]):
            zero_over = (ref, kl, _alpha_state(controller), scale)
        _log(f"  final push: scale={scale:.2f} refusal={ref:.3f} kl={kl:.3f}")
        if ref <= cfg.target_refusal and kl <= cfg.max_kl:
            lo = prev_scale if prev_ref is not None and prev_ref > cfg.target_refusal else 1.0
            hi = scale
            tight = (ref, kl, _alpha_state(controller), scale)
            for _ in range(3):
                mid = 0.5 * (lo + hi)
                _scale_alpha_state(controller, base, mid)
                with controller.active():
                    mid_ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
                mid_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
                _log(f"  final push refine: scale={mid:.3f} refusal={mid_ref:.3f} kl={mid_kl:.3f}")
                if mid_ref <= cfg.target_refusal and mid_kl <= cfg.max_kl:
                    hi = mid
                    tight = (mid_ref, mid_kl, _alpha_state(controller), mid)
                else:
                    lo = mid
            margin = float(getattr(cfg, "final_push_bake_margin", 0.0)) if cfg.bake else 0.0
            if margin > 0.0 and tight[3] < scale:
                margin_scale = min(scale, tight[3] * (1.0 + margin))
                _scale_alpha_state(controller, base, margin_scale)
                with controller.active():
                    margin_ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
                margin_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
                _log(f"  final push margin: scale={margin_scale:.3f} refusal={margin_ref:.3f} kl={margin_kl:.3f}")
                if margin_ref <= cfg.target_refusal and margin_kl <= cfg.max_kl:
                    tight = (margin_ref, margin_kl, _alpha_state(controller), margin_scale)
            if best is None or (tight[0], tight[1]) < (best[0], best[1]):
                best = tight
            break
        prev_scale = scale
        prev_ref = ref
    if getattr(cfg, "final_zero_trim", False) and zero_over is not None and zero_over[1] > cfg.max_kl:
        _set_alpha_state(controller, zero_over[2])
        _log(f"  final push trim: scale={zero_over[3]:.2f} start_kl={zero_over[1]:.3f}")
        ref, kl, _steps = _fast_zero_kl_trim(
            bundle, controller, cfg, eval_harmful, eval_harmless,
            start_ref=zero_over[0], start_kl=zero_over[1],
        )
        if ref <= cfg.target_refusal and (best is None or kl <= best[1] or best[0] > cfg.target_refusal):
            return ref, kl, zero_over[3]
    elif zero_over is not None and zero_over[1] > cfg.max_kl:
        _log(f"  final push: zero found at scale={zero_over[3]:.2f} kl={zero_over[1]:.3f} (over budget)")
    if best is not None:
        _set_alpha_state(controller, best[2])
        return best[0], best[1], best[3]
    _set_alpha_state(controller, base)
    with controller.active():
        ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
    kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
    return ref, kl, 1.0


def _fast_zero_kl_trim(bundle, controller, cfg, eval_harmful, eval_harmless, start_ref=None, start_kl=None):
    hset = eval_harmful[: max(48, min(len(eval_harmful), cfg.repair_eval_n))]
    lset = eval_harmless[: max(48, min(len(eval_harmless), cfg.repair_kl_n))]
    best_alpha = _alpha_state(controller)
    best_ref = start_ref
    best_kl = start_kl
    if best_ref is None:
        with controller.active():
            best_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
    if best_kl is None:
        best_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    if best_ref > cfg.target_refusal:
        return best_ref, best_kl, 0
    items = [-2, -1] + sorted(
        range(controller.num_layers),
        key=lambda L: abs(controller.get_layer_alpha(L)),
        reverse=True,
    )[:4]
    steps = 0
    for step in range(2):
        _log(f"  final trim scan {step + 1}: refusal={best_ref:.3f} kl={best_kl:.3f}")
        scored = []
        _set_alpha_state(controller, best_alpha)
        for item in items:
            start = _alpha_get(controller, item)
            if abs(start) < 1e-6:
                continue
            _alpha_set(controller, item, 0.0)
            kl0 = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
            _alpha_set(controller, item, start)
            drop = best_kl - kl0
            if drop > 1e-3:
                scored.append((drop, item, kl0))
        if not scored:
            break
        accepted = None
        for _drop, item, _kl0 in sorted(scored, reverse=True)[:4]:
            start = _alpha_get(controller, item)
            for scale in (0.0, 0.25, 0.50, 0.75):
                _set_alpha_state(controller, best_alpha)
                _alpha_set(controller, item, start * scale)
                kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
                if kl >= best_kl - 1e-3:
                    continue
                with controller.active():
                    ref, complete = refusal_rate_bounded(
                        bundle, hset, cfg.max_new_tokens, cfg.batch_size,
                        should_stop=lambda floor, _seen, _total: floor > cfg.target_refusal,
                    )
                if not complete and ref > cfg.target_refusal:
                    continue
                if ref <= cfg.target_refusal and (accepted is None or kl < accepted[0]):
                    accepted = (kl, ref, item, scale, _alpha_state(controller))
            if accepted is not None and accepted[0] <= cfg.max_kl:
                break
        if accepted is None:
            break
        best_kl, best_ref, item, scale, best_alpha = accepted
        steps += 1
        _log(f"  final trim {step + 1}: {_alpha_label(item)} x{scale:.2f} refusal={best_ref:.3f} kl={best_kl:.3f}")
        if best_kl <= cfg.max_kl:
            break
    _set_alpha_state(controller, best_alpha)
    return best_ref, best_kl, steps


def _edit_exists(controller, name: str) -> bool:
    return any(e.get("name") == name for e in getattr(controller, "edits", []))


def _edit_R(controller, name: str):
    for e in getattr(controller, "edits", []):
        if e.get("name") == name:
            r = e.get("R")
            return None if r is None else r.detach().cpu().clone()
    return None


def _restore_edit_R(controller, name: str, R):
    if R is not None:
        controller.set_edit_subspace(name, R)
        return
    for e in getattr(controller, "edits", []):
        if e.get("name") == name:
            e["R"] = None
            e["_cast"] = None
            break


def _residual_refusal_repair(bundle, controller, cfg, eval_harmful, eval_harmless, start_ref=None, start_kl=None):
    if bundle.uses_post_norm():
        return None
    hscan = eval_harmful[: max(24, min(len(eval_harmful), cfg.repair_eval_n))]
    lset = eval_harmless[: max(48, min(len(eval_harmless), cfg.repair_kl_n))]
    with controller.active():
        completions = generate(bundle, hscan, cfg.max_new_tokens, cfg.batch_size)
    flags = judge_strict_refusal(bundle, completions, cfg.batch_size, hscan)
    failed = [p for p, f in zip(hscan, flags) if f]
    passed = [p for p, f in zip(hscan, flags) if not f]
    if start_ref is None:
        start_ref = sum(flags) / max(1, len(flags))
    if start_kl is None:
        start_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    if not failed:
        return {"refusal": 0.0, "kl": start_kl, "failures": 0}

    name = "residual_failure"
    if not _edit_exists(controller, name):
        controller.add_edit(name, sign=-1.0, default_alpha=0.0, kind="hidden")
    prev_alpha = _alpha_state(controller)
    prev_R = _edit_R(controller, name)
    contrast = passed[: max(len(failed), 16)] or lset[: max(len(failed), 16)]
    with controller.active():
        fail_acts = collect_activations(bundle, failed, cfg.batch_size)
        base_acts = collect_activations(bundle, contrast, cfg.batch_size)

    active_layers = sorted(
        range(controller.num_layers),
        key=lambda L: abs(controller.get_layer_alpha(L)),
        reverse=True,
    )[: max(4, min(6, controller.num_layers))]
    alphas = (0.08, 0.14, 0.22, 0.32)
    best = None
    for layer in active_layers:
        R, _ = refusal_subspace(
            fail_acts[layer], base_acts[layer],
            rank=1, max_rank=max(1, min(2, cfg.max_rank)),
            seed=cfg.seed + 911 + layer,
            **_direction_kwargs(cfg),
        )
        controller.set_edit_subspace(name, gram_schmidt_remove(R, None))
        for alpha in alphas:
            _set_alpha_state(controller, prev_alpha)
            controller.set_edit_layer_alpha(name, layer, alpha)
            kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
            if kl > cfg.max_kl:
                continue
            with controller.active():
                ref = refusal_rate(bundle, hscan, cfg.max_new_tokens, cfg.batch_size)
            _log(f"  residual repair: L{layer} a={alpha:.2f} refusal={ref:.3f} kl={kl:.3f}")
            if ref >= start_ref - 1e-6:
                continue
            if best is None or (ref, kl) < (best[0], best[1]):
                best = (ref, kl, _alpha_state(controller), _edit_R(controller, name), layer, alpha)
                if ref <= cfg.target_refusal:
                    break
        if best is not None and best[0] <= cfg.target_refusal:
            break

    if best is None:
        _set_alpha_state(controller, prev_alpha)
        _restore_edit_R(controller, name, prev_R)
        return {"refusal": start_ref, "kl": start_kl, "failures": len(failed), "kept": False}
    _set_alpha_state(controller, best[2])
    _restore_edit_R(controller, name, best[3])
    return {
        "refusal": best[0],
        "kl": best[1],
        "failures": len(failed),
        "layer": best[4],
        "alpha": best[5],
        "kept": True,
    }


def _direction_kwargs(cfg):
    return {
        "multi": bool(getattr(cfg, "multi_refusal", True)),
        "clusters": int(getattr(cfg, "multi_refusal_clusters", 6)),
        "min_norm_frac": float(getattr(cfg, "multi_refusal_min_norm", 0.08)),
        "min_separation": float(getattr(cfg, "multi_refusal_min_separation", 0.05)),
        "min_coverage": float(getattr(cfg, "multi_refusal_min_coverage", 0.05)),
        "kl_whiten": float(getattr(cfg, "kl_whiten", 0.0)),
        "kl_whiten_shrink": float(getattr(cfg, "kl_whiten_shrink", 0.1)),
    }


# post-norm models: per-layer reader-side directions + a calibrated global strength.
def _reader_profile(bundle, controller, ah, al, cfg, preserve_lookup, eval_harmful, eval_harmless, log):
    nl = bundle.num_layers
    for l in range(nl):
        Rl, _ = refusal_subspace(
            ah[l], al[l], rank=max(1, min(cfg.max_rank, cfg.refusal_rank)),
            max_rank=cfg.max_rank, seed=cfg.seed,
            orthogonalize=cfg.orthogonalize_direction,
            **_direction_kwargs(cfg),
        )
        controller.set_reader_layer_subspace(l, gram_schmidt_remove(Rl, preserve_lookup(l)))
    if cfg.causal_targeting:
        log("scoring per-layer causal importance (reader) ...")
        # floor 0 concentrates strength on layers that actually carry refusal; this keeps
        # kl down (edge layers near the head spike kl for little refusal benefit).
        causal = causal_layer_scores(
            bundle, controller, eval_harmful[: cfg.opt_eval_n], cfg.batch_size,
            floor=0.0, temperature=cfg.causal_temperature,
        )
    else:
        causal = [1.0] * nl
    hset, lset = eval_harmful[: cfg.opt_eval_n], eval_harmless[: cfg.opt_eval_n]

    def apply(g):
        controller.set_uniform_alpha(0.0)
        for l in range(nl):
            controller.set_layer_alpha(l, g * causal[l])

    controller.enable()
    # rank strengths by real generated refusal (the logit margin is unreliable on post-norm
    # models) and pick the strongest that stays under the kl budget.
    best = None  # (g, refusal, kl, feasible)
    for g in cfg.reader_strengths:
        apply(g)
        with controller.active():
            ref = refusal_rate(bundle, hset, cfg.opt_gen_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
        log(f"  reader strength {g}: refusal={ref:.3f} kl={kl:.3f}")
        feasible = kl <= cfg.reader_max_kl
        if best is None or (feasible and (ref < best[1] - 1e-6 or not best[3])):
            best = (g, ref, kl, feasible)
        if ref <= cfg.target_refusal and feasible:
            break
        if best is not None and best[3] and not feasible:
            break  # kl over budget; higher strengths only raise it further
    apply(best[0])
    with controller.active():  # one real refusal number for the report
        ref = refusal_rate(bundle, eval_harmful[: cfg.opt_eval_n], cfg.opt_gen_tokens, cfg.batch_size)
    return {"strength": best[0], "refusal": ref, "kl": best[2], "causal": causal}


def run(cfg: ApostateConfig, command: Optional[str] = None) -> dict:
    t0 = time.time()
    phase_t = t0
    timings: dict = {}

    def mark(name: str):
        nonlocal phase_t
        now = time.time()
        timings[name] = round(now - phase_t, 1)
        phase_t = now
        _log(f"timing {name}: {timings[name]}s")

    cfg.with_defaults()
    os.makedirs(cfg.output_dir, exist_ok=True)

    _log(f"loading {cfg.model} (4bit={cfg.load_in_4bit}) ...")
    bundle = load_model(cfg)
    tok = bundle.tokenizer
    L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * cfg.direction_layer_frac)))
    nw = len(bundle.layer_writers(bundle.layers()[L_dir]))
    arch = f"MoE ({nw} writers/layer)" if bundle.is_moe() else "dense"
    _log(f"{bundle.num_layers} layers, hidden={bundle.hidden_size}, {arch}, direction layer={L_dir}")
    mark("load_model")

    harmful = _cached_prompts(cfg, "harmful_fit", cfg.harmful_path, cfg.n_harmful + cfg.n_eval, cfg.seed)
    harmless = _cached_prompts(cfg, "harmless_fit", cfg.harmless_path, cfg.n_harmless, cfg.seed)
    fit_harmful = harmful[: cfg.n_harmful]
    if cfg.harmful_test:
        eval_harmful = _cached_prompts(cfg, "harmful_eval", cfg.harmful_test, cfg.n_eval, cfg.seed)
    else:
        tail = harmful[cfg.n_harmful : cfg.n_harmful + cfg.n_eval]
        eval_harmful = tail if len(tail) >= cfg.n_eval else harmful[: cfg.n_eval]
    if cfg.harmless_test:
        eval_harmless = _cached_prompts(cfg, "harmless_eval", cfg.harmless_test, cfg.n_eval, cfg.seed)
    else:
        tail = harmless[cfg.n_harmless - cfg.n_eval : cfg.n_harmless]
        eval_harmless = tail if len(tail) >= cfg.n_eval else harmless[: cfg.n_eval]
    hh = max(1, len(eval_harmful) // 2)
    hl = max(1, len(eval_harmless) // 2)
    test_harmful, eval_harmful = eval_harmful[:hh], eval_harmful[hh:]
    test_harmless, eval_harmless = eval_harmless[:hl], eval_harmless[hl:]
    refusal_eval_prompts = []
    if cfg.refusal_eval_path and cfg.refusal_eval_n > 0:
        refusal_eval_prompts = _cached_prompts(cfg, "refusal_eval", cfg.refusal_eval_path, cfg.refusal_eval_n, cfg.seed)
        eval_harmful = _append_unique(refusal_eval_prompts, eval_harmful)
    if cfg.target_refusal <= 0.0:
        eval_harmful = _append_unique(test_harmful, eval_harmful)
    kl_eval_prompts = []
    if cfg.kl_eval_path and cfg.kl_eval_n > 0:
        kl_eval_prompts = _cached_prompts(cfg, "kl_eval", cfg.kl_eval_path, cfg.kl_eval_n, cfg.seed)
        eval_harmless = _append_unique(kl_eval_prompts, eval_harmless)
    _log(f"prompts: {len(fit_harmful)} harmful (fit), {len(harmless)} harmless, "
         f"val {len(eval_harmful)}/{len(eval_harmless)}, test {len(test_harmful)}/{len(test_harmless)}"
         + (f", refusal calib {len(refusal_eval_prompts)}" if refusal_eval_prompts else "")
         + (f", kl calib {len(kl_eval_prompts)}" if kl_eval_prompts else ""))
    mark("load_prompts")

    controller = ProjectionController(bundle)
    controller.disable()

    if cfg.baseline_eval_n and cfg.baseline_eval_n > 0:
        base_eval = test_harmful[: max(1, min(len(test_harmful), cfg.baseline_eval_n))]
    else:
        base_eval = test_harmful
    base_refusal = refusal_rate(bundle, base_eval, cfg.max_new_tokens, cfg.batch_size)
    _log(f"baseline refusal rate (test): {base_refusal:.3f}")
    mark("baseline_refusal")

    report_extra: dict = {"optimized": cfg.optimize}

    head_only_profile = False
    head_ref_est = None
    reader_est = None  # (refusal, kl) estimate to skip the redundant validation pass
    head_sweep_profile = _head_sweep_enabled(bundle, cfg)
    head_sweep_attrs = None
    ah = al = None
    ple_ah = ple_al = None
    pler_ah = pler_al = None
    plee_ah = plee_al = None
    kv_ah = kv_al = None
    q_ah = q_al = None
    response_h_prompts = response_h_texts = None
    response_l_prompts = response_l_texts = None
    preserve_source = "none"
    preserve_lookup = _preservation_lookup(None, 0)
    ple_preserve_lookup = _preservation_lookup(None, 0)
    pler_preserve_lookup = _preservation_lookup(None, 0)
    kv_preserve_lookup = _kv_preservation_lookup(None, 0)
    q_preserve_lookup = _dict_preservation_lookup(None, 0)
    plee_preserve_basis = None
    preserve_basis = None
    if head_sweep_profile:
        _log("activation fit skipped: head sweep")
        mark("activation_fit")
    else:
        _log("collecting activations (original model) ...")
        if cfg.fit_response_activations:
            ah, al, response_h_prompts, response_h_texts, response_l_prompts, response_l_texts = _response_fit_activations(
                bundle, fit_harmful, harmless, cfg
            )
        else:
            ah = _cached_collect(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful")
            al = _cached_collect(bundle, harmless, cfg.batch_size, cfg, "fit_harmless")
        if getattr(cfg, "gemma_ple", True) and bundle.has_ple():
            _log("collecting ple gate activations ...")
            ple_ah = _cached_collect_ple(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful_ple")
            ple_al = _cached_collect_ple(bundle, harmless, cfg.batch_size, cfg, "fit_harmless_ple")
            _log("collecting ple residual activations ...")
            pler_ah = _cached_collect_ple_projection(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful_ple_residual")
            pler_al = _cached_collect_ple_projection(bundle, harmless, cfg.batch_size, cfg, "fit_harmless_ple_residual")
            _log("collecting ple embed activations ...")
            plee_ah = _cached_collect_ple_embed(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful_ple_embed")
            plee_al = _cached_collect_ple_embed(bundle, harmless, cfg.batch_size, cfg, "fit_harmless_ple_embed")
        if cfg.optimize and bundle.has_shared_kv():
            src = bundle.kv_source_layers()
            if response_h_texts is not None and response_l_texts is not None:
                _log(f"collecting response kv activations layers={src} ...")
                kv_ah = _cached_collect_response_kv(
                    bundle, response_h_prompts, response_h_texts, cfg.batch_size, cfg, "fit_harmful_response_kv"
                )
                kv_al = _cached_collect_response_kv(
                    bundle, response_l_prompts, response_l_texts, cfg.batch_size, cfg, "fit_harmless_response_kv"
                )
            else:
                _log(f"collecting kv activations layers={src} ...")
                kv_ah = _cached_collect_kv(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful_kv")
                kv_al = _cached_collect_kv(bundle, harmless, cfg.batch_size, cfg, "fit_harmless_kv")
        if cfg.optimize and getattr(cfg, "gemma_query", False):
            q_layers = bundle.query_layer_candidates()
            if q_layers:
                if response_h_texts is not None and response_l_texts is not None:
                    _log(f"collecting response query activations layers={q_layers} ...")
                    q_ah = _cached_collect_response_query(
                        bundle, response_h_prompts, response_h_texts, cfg.batch_size, cfg, "fit_harmful_response_query"
                    )
                    q_al = _cached_collect_response_query(
                        bundle, response_l_prompts, response_l_texts, cfg.batch_size, cfg, "fit_harmless_response_query"
                    )
                else:
                    _log(f"collecting query activations layers={q_layers} ...")
                    q_ah = _cached_collect_query(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful_query")
                    q_al = _cached_collect_query(bundle, harmless, cfg.batch_size, cfg, "fit_harmless_query")

        preserve_acts = None
        if cfg.preserve_rank > 0 and cfg.preserve_path:
            preserve = _cached_prompts(cfg, "preserve", cfg.preserve_path, cfg.n_harmless, cfg.seed)
            preserve_acts = _cached_collect(bundle, preserve, cfg.batch_size, cfg, "preserve")
            preserve_source = "custom"
        elif cfg.preserve_rank > 0:
            preserve_acts = al
            preserve_source = "harmless"
        cap_acts = None
        if (getattr(cfg, "capability_preserve", True) and cfg.capability_preserve_rank > 0
                and cfg.capability_preserve_path):
            _log("collecting capability (math/code) activations for preservation ...")
            cap_prompts = _cached_prompts(cfg, "capability_preserve", cfg.capability_preserve_path,
                                          cfg.capability_preserve_n, cfg.seed)
            if cap_prompts:
                cap_acts = _cached_collect(bundle, cap_prompts, cfg.batch_size, cfg, "capability_preserve")
                _log(f"capability preservation: rank={cfg.capability_preserve_rank} prompts={len(cap_prompts)}")
        if cap_acts is not None:
            preserve_lookup = _combined_preservation_lookup(
                [(preserve_acts, cfg.preserve_rank), (cap_acts, cfg.capability_preserve_rank)])
            preserve_source = (preserve_source + "+capability") if preserve_acts is not None else "capability"
        else:
            preserve_lookup = _preservation_lookup(preserve_acts, cfg.preserve_rank)
        preserve_basis = preserve_lookup(L_dir)
        if ple_al is not None:
            ple_preserve_lookup = _preservation_lookup(ple_al, min(4, max(1, cfg.preserve_rank)))
        if pler_al is not None:
            pler_preserve_lookup = _preservation_lookup(pler_al, min(4, max(1, cfg.preserve_rank)))
        if kv_al is not None:
            kv_preserve_lookup = _kv_preservation_lookup(kv_al, min(4, max(1, cfg.preserve_rank)))
        if q_al is not None:
            q_preserve_lookup = _dict_preservation_lookup(q_al, min(4, max(1, cfg.preserve_rank)))
        if plee_al is not None and cfg.preserve_rank > 0:
            plee_preserve_basis = preservation_subspace(plee_al, rank=min(4, max(1, cfg.preserve_rank)))
        if preserve_basis is not None:
            _log(f"preservation subspace rank={preserve_basis.shape[1]} source={preserve_source}")
        mark("activation_fit")

    reader_mode = bundle.uses_post_norm()
    fast = (cfg.profile or "balanced").lower() == "fast"  # profile, not architecture
    if (getattr(cfg, "oblique_ablation", True) and not reader_mode and not head_sweep_profile
            and al is not None):
        # mean-preserving ablation: remove R but along a co-vector orthogonal to the harmless
        # mean, so the edit doesn't shift the harmless residual mean -> the dominant kl term
        # collapses. weight-only, bakes into bias-free qwen2.5. the whole search optimizes it.
        mu = al[L_dir].mean(0)
        controller.enable_oblique(mu, cfg.oblique_strength, cfg.oblique_denom_floor,
                                  writers_only=getattr(cfg, "oblique_writers_only", True))
        _log(f"oblique ablation enabled (strength={cfg.oblique_strength}, "
             f"writers_only={getattr(cfg, 'oblique_writers_only', True)}, |mu|={float(mu.norm()):.2f})")
    if reader_mode:
        # gemma2/3/4-style post-norm: writer edits get renormalized away, so ablate
        # reader-side with per-layer directions, and give kl extra headroom.
        cfg.max_kl = max(cfg.max_kl, cfg.reader_max_kl)
        cfg.kl_target = max(cfg.kl_target, cfg.reader_kl_target)
        _log("post-norm architecture: per-layer reader-side ablation")
        rinfo = _reader_profile(bundle, controller, ah, al, cfg, preserve_lookup,
                                eval_harmful, eval_harmless, _log)
        _log(f"reader profile: strength={rinfo['strength']} "
             f"refusal={rinfo['refusal']:.3f} kl={rinfo['kl']:.3f}")
        report_extra.update({"reader_mode": True, "reader_strength": rinfo["strength"],
                             "best_trial": {"refusal": rinfo["refusal"], "kl": rinfo["kl"]},
                             "n_trials": 0})
        if fast:  # reuse the search estimate instead of a full validation pass
            reader_est = (rinfo["refusal"], rinfo["kl"])
        mark("reader_profile")
        if rinfo["refusal"] > cfg.target_refusal:
            # corrective directions try to push refusal lower; conservative, so it can't hurt
            _log("running reader guard (corrective directions) ...")
            run_reader_guard(bundle, controller, fit_harmful[: cfg.opt_eval_n],
                             harmless[: cfg.opt_eval_n], cfg, preserve_lookup,
                             eval_harmful[: cfg.opt_eval_n], eval_harmless[: cfg.opt_eval_n], _log)
        mark("reader_guard")
    elif head_sweep_profile:
        _log("optimizing head token profile ...")
        best_params, best_attrs, opt_hist = _head_token_sweep(
            bundle, controller, cfg,
            eval_harmful, eval_harmless,
        )
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()}
        _log(f"best head sweep: refusal={best_attrs.get('refusal', best_attrs.get('refusal_proxy'))} "
             f"kl={best_attrs.get('kl')} | {shown}")
        L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * best_params["direction_layer_frac"])))
        head_only_profile = True
        head_ref_est = best_attrs.get("refusal", best_attrs.get("refusal_proxy"))
        head_sweep_attrs = best_attrs
        report_extra.update({
            "best_params": best_params,
            "best_trial": best_attrs,
            "n_trials": 0,
            "head_sweep": True,
            "head_sweep_trials": len(opt_hist),
        })
        mark("head_sweep")
    elif cfg.optimize:
        Rseed, _ = refusal_subspace(
            ah[L_dir], al[L_dir], rank=1, max_rank=cfg.max_rank, seed=cfg.seed,
            **_direction_kwargs(cfg),
        )
        controller.set_subspace(gram_schmidt_remove(Rseed, preserve_basis))
        if cfg.causal_targeting:
            _log("scoring per-layer causal importance (prior) ...")
            causal_shape = causal_layer_scores(
                bundle, controller, eval_harmful[: cfg.opt_eval_n], cfg.batch_size,
                floor=cfg.causal_floor, temperature=cfg.causal_temperature,
            )
        else:
            causal_shape = [1.0] * bundle.num_layers
        mark("causal_scores")
        _log(f"optimizing ablation profile via {'TPE' if _has_optuna() else 'random'} search: {cfg.n_trials} trials ...")
        best_params, best_attrs, opt_hist = optimize_profile(
            bundle, controller, ah, al,
            eval_harmful[: cfg.opt_eval_n], eval_harmless[: cfg.opt_eval_n],
            causal_shape, cfg, preserve_basis, preserve_lookup,
            ple_ah=ple_ah, ple_al=ple_al, ple_preserve_lookup=ple_preserve_lookup,
            pler_ah=pler_ah, pler_al=pler_al, pler_preserve_lookup=pler_preserve_lookup,
            plee_ah=plee_ah, plee_al=plee_al, plee_preserve_basis=plee_preserve_basis,
            kv_ah=kv_ah, kv_al=kv_al, kv_preserve_lookup=kv_preserve_lookup,
            q_ah=q_ah, q_al=q_al, q_preserve_lookup=q_preserve_lookup,
        )
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()}
        _log(f"best trial: refusal_proxy={best_attrs.get('refusal_proxy', best_attrs.get('refusal'))} "
             f"kl={best_attrs.get('kl')} | {shown}")
        L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * best_params["direction_layer_frac"])))
        preserve_basis = preserve_lookup(L_dir)
        head_only_profile = (
            best_params.get("direction_source") == "head_tokens"
            and bool(best_params.get("ablate_head", False))
            and abs(float(best_params.get("strength", 0.0))) <= 0.01
        )
        head_ref_est = best_attrs.get("refusal", best_attrs.get("refusal_proxy"))
        report_extra.update({"best_params": best_params, "best_trial": best_attrs, "n_trials": cfg.n_trials})
        mark("optimize_profile")
    else:
        R, svals = refusal_subspace(
            ah[L_dir], al[L_dir],
            rank=cfg.refusal_rank, variance_threshold=cfg.variance_threshold,
            max_rank=cfg.max_rank, seed=cfg.seed,
            **_direction_kwargs(cfg),
        )
        _log(f"refusal subspace rank={R.shape[1]} (svals={[round(float(s),2) for s in svals]})")
        preserve_basis = preserve_lookup(L_dir)
        controller.set_subspace(gram_schmidt_remove(R, preserve_basis))
        if cfg.causal_targeting:
            _log("scoring per-layer causal importance ...")
            alphas = causal_layer_scores(
                bundle, controller, eval_harmful, cfg.batch_size,
                floor=cfg.causal_floor, temperature=cfg.causal_temperature,
            )
            for L in range(bundle.num_layers):
                controller.set_layer_alpha(L, alphas[L])
            if bundle.can_edit_embed():
                controller.set_embed_alpha(1.0)
            else:
                controller.set_embed_alpha(0.0)
                _log("embed edit disabled: per-layer embeddings")
            top = sorted(range(len(alphas)), key=lambda i: -alphas[i])[:5]
            _log(f"top causal layers: {[(i, round(alphas[i],2)) for i in top]}")
        else:
            controller.set_uniform_alpha(1.0)
        mark("profile_setup")

    initial_sep = 0.0 if head_sweep_profile else separation(ah[L_dir], al[L_dir])
    controller.enable()

    guard_hist = []
    skip_guard = False
    guard_skip_reason = (
        "reader mode" if reader_mode
        else "head token profile" if head_only_profile and cfg.opt_guard
        else None
    )
    if guard_skip_reason is None and ((not cfg.optimize) or cfg.opt_guard) and cfg.opt_early_stop:
        with controller.active():
            ref_quick = refusal_rate(bundle, eval_harmful[:min(24, len(eval_harmful))], cfg.max_new_tokens, cfg.batch_size)
        skip_guard = ref_quick <= cfg.target_refusal
    if guard_skip_reason is None and ((not cfg.optimize) or cfg.opt_guard) and not skip_guard:
        _log("running reconstruction guard ...")
        gcap = max(256, cfg.opt_eval_n)
        guard_hist = run_guard(
            bundle, controller, fit_harmful[:gcap], harmless[:gcap], cfg, L_dir, initial_sep,
            preserve_basis,
            eval_harmful=eval_harmful[: cfg.opt_eval_n], eval_harmless=eval_harmless[: cfg.opt_eval_n],
        )
        for h in guard_hist:
            _log(f"  guard iter {h['iter']}: sep={h['separation']} ratio={h['ratio']} "
                 f"rank={h['rank']} refusal={h.get('refusal')} kl={h.get('kl')}")
    elif guard_skip_reason is not None:
        _log(f"guard: skipped ({guard_skip_reason})")
    elif skip_guard:
        _log(f"guard: skipped (refusal {ref_quick:.3f} <= target {cfg.target_refusal:.3f})")
    mark("guard")

    head_needs_refine = (
        head_only_profile
        and (head_ref_est is None or head_ref_est > cfg.target_refusal + cfg.refine_refusal_slack)
    )
    should_refine = cfg.refine_refusal and (
        head_needs_refine
        or skip_guard
        or (len(guard_hist) > 0 and guard_hist[-1].get("refusal", 1.0) > cfg.target_refusal)
    )
    refine_ref = None
    refine_kl = None
    if head_sweep_profile and head_sweep_attrs is not None:
        refine_ref = head_sweep_attrs.get("refusal", head_sweep_attrs.get("refusal_proxy"))
        refine_kl = head_sweep_attrs.get("kl")
        _log("refine: skipped (head sweep exact)")
    elif should_refine:
        _log("refining to target refusal ...")
        rr, rk = _refine_refusal(bundle, controller, cfg, eval_harmful, eval_harmless)
        refine_ref, refine_kl = rr, rk
        _log(f"refine result: refusal={rr:.3f} kl={rk:.3f}")
    elif cfg.refine_refusal and skip_guard:
        _log("refine: skipped (guard was skipped, refusal already clean)")
    mark("refine_refusal")

    edited_refusal = refine_ref
    kl = refine_kl
    if (edited_refusal is None or kl is None) and reader_est is not None:
        edited_refusal, kl = reader_est  # reuse the sweep estimate; skip a generation pass
    if edited_refusal is None or kl is None:
        with controller.active():
            edited_refusal = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        _log(f"edited refusal rate: {edited_refusal:.3f} | harmless KL: {kl:.3f} nats")
        mark("validation_metrics")
    else:
        _log(f"edited refusal estimate: {edited_refusal:.3f} | harmless KL estimate: {kl:.3f} nats")
        timings["validation_metrics"] = 0.0

    kl_layer_steps = 0
    repair_steps = 0
    residual_repair = []
    if (not head_sweep_profile) and cfg.refine_deescalate and edited_refusal <= cfg.target_refusal + cfg.refine_refusal_slack:
        _log("minimizing kl scale ...")
        edited_refusal, kl = _minimize_kl_scale(bundle, controller, cfg, eval_harmful, eval_harmless)
        refine_ref = None
        refine_kl = None
        _log(f"kl scale result: refusal={edited_refusal:.3f} kl={kl:.3f}")
        if kl > cfg.kl_target:
            _log("trimming kl layers ...")
            edited_refusal, kl, kl_layer_steps = _minimize_kl_layers(
                bundle, controller, cfg, eval_harmful, eval_harmless,
            )
            refine_ref = None
            refine_kl = None
            _log(f"kl layer result: refusal={edited_refusal:.3f} kl={kl:.3f} steps={kl_layer_steps}")

    if (not head_sweep_profile) and cfg.refine_deescalate and (edited_refusal > cfg.target_refusal or kl > cfg.kl_target):
        _log("repairing refusal/kl tradeoff ...")
        edited_refusal, kl, repair_steps = _repair_alphas(
            bundle, controller, cfg, eval_harmful, eval_harmless,
            start_ref=refine_ref,
        )
        refine_ref = None
        refine_kl = None
        _log(f"repair result: refusal={edited_refusal:.3f} kl={kl:.3f} steps={repair_steps}")

    backoff = 0
    if kl > cfg.max_kl:
        edited_refusal, kl, backoff = _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless)
        refine_ref = None
        refine_kl = None
        if cfg.refine_deescalate and edited_refusal > cfg.target_refusal:
            _log("repairing after backoff ...")
            edited_refusal, kl, extra_steps = _repair_alphas(bundle, controller, cfg, eval_harmful, eval_harmless)
            repair_steps += extra_steps
            _log(f"post-backoff repair: refusal={edited_refusal:.3f} kl={kl:.3f} steps={extra_steps}")
            if kl > cfg.max_kl:
                _log("final kl backoff ...")
                edited_refusal, kl, extra_backoff = _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless)
                backoff += extra_backoff
    mark("repair")

    drop_layers: list = []
    skipper = None
    if cfg.prune:
        from .prune import select_prune, LayerSkip
        _log("scoring layer redundancy for pruning ...")
        drop_layers = select_prune(
            bundle, controller,
            eval_harmful[: cfg.opt_eval_n], eval_harmless[: cfg.opt_eval_n], cfg,
        )
        if drop_layers:
            speedup = 100.0 * len(drop_layers) / bundle.num_layers
            _log(f"pruning {len(drop_layers)} / {bundle.num_layers} layers "
                 f"(~{speedup:.0f}% faster): {drop_layers}")
            skipper = LayerSkip(bundle)
            skipper.set(drop_layers)
        else:
            _log("pruning: no layer is free within kl budget")
    mark("prune")

    with controller.active():
        edited_refusal = refusal_rate(bundle, test_harmful, cfg.max_new_tokens, cfg.batch_size)
    if skipper is not None:
        skipper.remove()
    kl = kl_harmless(bundle, controller, test_harmless, cfg.batch_size, positions=cfg.kl_positions)
    if (not head_sweep_profile) and cfg.refine_deescalate and edited_refusal > cfg.target_refusal:
        _log("repairing test leakage ...")
        edited_refusal, kl, extra_steps = _repair_alphas(
            bundle, controller, cfg, test_harmful, test_harmless,
            start_ref=edited_refusal, start_kl=kl,
        )
        repair_steps += extra_steps
        with controller.active():
            edited_refusal = refusal_rate(bundle, test_harmful, cfg.max_new_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, test_harmless, cfg.batch_size, positions=cfg.kl_positions)
        _log(f"test repair result: refusal={edited_refusal:.3f} kl={kl:.3f} steps={extra_steps}")
    if (not head_sweep_profile) and cfg.refine_deescalate and edited_refusal > cfg.target_refusal:
        _log("pushing final refusal scale ...")
        edited_refusal, kl, push_scale = _push_refusal_scale(
            bundle, controller, cfg, test_harmful, test_harmless,
        )
        _log(f"final push result: scale={push_scale:.2f} refusal={edited_refusal:.3f} kl={kl:.3f}")
    _log(f"TEST refusal: {edited_refusal:.3f} | TEST harmless KL: {kl:.3f} nats"
         + (f" | {len(drop_layers)} layers pruned" if drop_layers else ""))
    mark("test_metrics")

    report = {
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "hidden_size": bundle.hidden_size,
        "direction_layer": L_dir,
        "refusal_subspace_rank": int(controller.R.shape[1]),
        "max_refusal_rank": cfg.max_rank,
        "multi_refusal": cfg.multi_refusal,
        "multi_refusal_clusters": cfg.multi_refusal_clusters,
        "multi_refusal_min_norm": cfg.multi_refusal_min_norm,
        "multi_refusal_min_separation": cfg.multi_refusal_min_separation,
        "multi_refusal_min_coverage": cfg.multi_refusal_min_coverage,
        "initial_separation": round(initial_sep, 4),
        "baseline_refusal_rate": round(base_refusal, 4),
        "baseline_eval_n": len(base_eval),
        "edited_refusal_rate": round(edited_refusal, 4),
        "refusal_metric": "classifier + weak guard",
        "harmless_kl_nats": round(kl, 4),
        "kl_backoff_steps": backoff,
        "kl_layer_trim_steps": kl_layer_steps,
        "repair_steps": repair_steps,
        "residual_repair": residual_repair,
        "guard_history": guard_hist,
        "layer_alphas": [round(controller.get_layer_alpha(L), 3) for L in range(bundle.num_layers)],
        "ple_layer_alphas": [round(controller.get_ple_layer_alpha(L), 3) for L in range(bundle.num_layers)]
        if controller.has_ple() else [],
        "ple_embed_alpha": round(controller.get_ple_embed_alpha(), 3),
        "ple_model_projection_alpha": round(controller.get_ple_model_projection_alpha(), 3),
        "embed_alpha": round(controller.get_embed_alpha(), 3),
        "head_alpha": round(controller.get_head_alpha(), 3),
        "head_token_alpha": round(controller.get_head_token_alpha(), 3),
        "preserve_rank": cfg.preserve_rank,
        "preserve_source": preserve_source,
        "pruned_layers": drop_layers,
        "layers_after_prune": bundle.num_layers - len(drop_layers),
        "elapsed_sec": round(time.time() - t0, 1),
        "profile": cfg.profile,
        "target_refusal": cfg.target_refusal,
        "max_kl": cfg.max_kl,
        "kl_target": cfg.kl_target,
        "refusal_eval_path": cfg.refusal_eval_path,
        "refusal_eval_n": cfg.refusal_eval_n,
        "kl_positions": cfg.kl_positions,
        "kl_eval_path": cfg.kl_eval_path,
        "kl_eval_n": cfg.kl_eval_n,
        "oblique_ablation": bool(controller.edits[0].get("U") is not None),
        "oblique_strength": cfg.oblique_strength,
        "kl_whiten": cfg.kl_whiten,
        "opt_capability": cfg.opt_capability,
        "opt_capability_weight": cfg.opt_capability_weight,
        "timings_sec": timings,
        "command": command,
    }
    report.update(report_extra)
    with open(os.path.join(cfg.output_dir, "apostate_config.json"), "w", encoding="utf-8") as f:
        f.write(cfg.to_json())
    _persist_reports(cfg, report, command)

    if cfg.bake:
        export = controller.export()
        _log("freeing 4-bit model and baking edited weights ...")
        controller.remove()
        del bundle.model, bundle
        gc.collect()
        torch.cuda.empty_cache()
        out = bake(cfg, export, tokenizer=tok, drop_layers=drop_layers)
        _log(f"baked edited model -> {out}")
        mark("bake")
        report["baked_to"] = out
        report["elapsed_sec"] = round(time.time() - t0, 1)
        _persist_reports(cfg, report, command)

    _log(f"done in {report['elapsed_sec']}s | refusal {base_refusal:.2f} -> {edited_refusal:.2f}, KL {kl:.3f}")
    return report
