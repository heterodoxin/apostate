"""profile search"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .directions import refusal_subspace, gram_schmidt_remove
import math
from .evaluate import refusal_rate, refusal_logit_margin, kl_harmless
from .data import format_chat
from .search import run_search


def _kl_loss(kl: float, cfg) -> float:
    over_target = max(0.0, kl - cfg.kl_target)
    over_budget = max(0.0, kl - cfg.max_kl)
    return (
        cfg.kl_weight * kl
        + cfg.kl_quad_weight * kl * kl
        + cfg.kl_target_weight * over_target
        + cfg.kl_over_budget_weight * over_budget
    )


def _refusal_loss(refusal: float, cfg) -> float:
    over = max(0.0, refusal - cfg.target_refusal)
    return refusal + cfg.refusal_target_weight * over + cfg.refusal_quad_weight * over * over


def _ref_attr(h: dict) -> float:
    return float(h.get("refusal", h.get("refusal_proxy", 1.0)))


def _low_kl_pick(history: list, cfg):
    if not history:
        return None
    slack = getattr(cfg, "refine_refusal_slack", 0.02)
    feasible = [
        h for h in history
        if _ref_attr(h) <= cfg.target_refusal + slack and float(h.get("kl", 99.0)) <= cfg.max_kl
    ]
    if feasible:
        return min(feasible, key=lambda h: (
            float(h.get("kl", 99.0)), float(h.get("capability_drift", 0.0)), _ref_attr(h), h["value"]
        ))
    under_budget = [h for h in history if float(h.get("kl", 99.0)) <= cfg.max_kl]
    if under_budget:
        return min(under_budget, key=lambda h: (
            _ref_attr(h), float(h.get("capability_drift", 0.0)), float(h.get("kl", 99.0)), h["value"]
        ))
    return min(history, key=lambda h: h["value"])


def _capability_samples(cfg) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    if not cfg.opt_capability:
        return samples
    if cfg.opt_capability_code_n > 0:
        try:
            from .codeeval import load_code_problems
            for p in load_code_problems("openai/openai_humaneval:test", cfg.opt_capability_code_n):
                sol = p.get("canonical_solution") or ""
                prompt = p.get("prompt") or ""
                if prompt and sol:
                    samples.append((prompt, sol))
        except Exception as e:
            print(f"[apostate] capability code skipped: {e}", flush=True)
    if cfg.opt_capability_math_n > 0:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            for i in range(min(cfg.opt_capability_math_n, len(ds))):
                q = ds[i]["question"]
                gold = ds[i]["answer"].split("####")[-1].strip().replace(",", "")
                samples.append((q + "\nThe answer is", " " + gold))
        except Exception as e:
            print(f"[apostate] capability math skipped: {e}", flush=True)
    return samples


@torch.no_grad()
def _target_logprob(bundle: ModelBundle, samples: List[Tuple[str, str]]) -> float:
    if not samples:
        return 0.0
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    vals: List[float] = []
    for prompt, target in samples:
        prompt_text = format_chat(tok, [prompt])[0]
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids
        if not prompt_ids or not target_ids:
            continue
        ids = torch.tensor([prompt_ids + target_ids], device=device)
        logits = model(ids, use_cache=False).logits[:, :-1, :].float()
        labels = ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        start = max(0, len(prompt_ids) - 1)
        sol_logp = tok_logp[:, start:start + len(target_ids)]
        if sol_logp.numel():
            vals.append(float(sol_logp.mean().item()))
    return sum(vals) / max(1, len(vals))


def _apply_profile(
    bundle: ModelBundle,
    controller: ProjectionController,
    ah: torch.Tensor,
    al: torch.Tensor,
    params: Dict,
    causal_shape: List[float],
    cfg,
    preserve_basis: Optional[torch.Tensor],
    preserve_lookup: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
) -> int:
    """apply trial"""
    n = bundle.num_layers
    L_dir = max(0, min(n - 1, int(n * params["direction_layer_frac"])))
    R, _ = refusal_subspace(
        ah[L_dir], al[L_dir],
        rank=int(params["refusal_rank"]), max_rank=cfg.max_rank, seed=cfg.seed,
    )
    basis = preserve_lookup(L_dir) if preserve_lookup is not None else preserve_basis
    R = gram_schmidt_remove(R, basis)
    controller.set_subspace(R)

    if "band_center" in params:
        width = params["band_width"]
        lo = max(0.0, params["band_center"] - width * 0.5)
        hi = min(1.0, params["band_center"] + width * 0.5)
    else:
        lo, hi = sorted((params["band_lo"], params["band_hi"]))
    strength = params["strength"]
    cmix = params["causal_mix"]
    power = params.get("causal_power", 1.0)
    for L in range(n):
        frac = L / max(1, n - 1)
        if lo <= frac <= hi:
            shape = (1.0 - cmix) + cmix * (max(0.0, causal_shape[L]) ** power)
            controller.set_layer_alpha(L, strength * shape)
        else:
            controller.set_layer_alpha(L, 0.0)
    embed_scale = params.get("embed_scale", 1.0)
    controller.set_embed_alpha(strength * embed_scale if params["ablate_embed"] else 0.0)
    return L_dir


def optimize_profile(
    bundle: ModelBundle,
    controller: ProjectionController,
    ah: torch.Tensor,
    al: torch.Tensor,
    eval_harmful: List[str],
    eval_harmless: List[str],
    causal_shape: List[float],
    cfg,
    preserve_basis: Optional[torch.Tensor] = None,
    preserve_lookup: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
) -> Tuple[dict, dict, list]:
    cap_samples = _capability_samples(cfg)
    base_cap = None
    if cap_samples:
        with controller.bypassed():
            base_cap = _target_logprob(bundle, cap_samples)
        print(f"[apostate] capability logprob baseline: {base_cap:.4f}", flush=True)

    space = {
        "direction_layer_frac": ("float", 0.30, 0.82),
        "refusal_rank": ("int", 1, min(3, cfg.max_rank)),
        "strength": ("float", 0.08, 1.15),
        "band_center": ("float", 0.15, 0.90),
        "band_width": ("float", 0.08, 0.65),
        "causal_mix": ("float", 0.0, 1.0),
        "causal_power": ("float", 1.0, 3.0),
        "ablate_embed": ("cat", [False, True]),
        "embed_scale": ("float", 0.0, 0.35),
    }

    def objective(params):
        _apply_profile(bundle, controller, ah, al, params, causal_shape, cfg, preserve_basis, preserve_lookup)
        with controller.active():
            if cfg.opt_objective == "generation":
                proxy = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
            else:
                # margin proxy
                margin = refusal_logit_margin(bundle, eval_harmful, cfg.batch_size)
                proxy = 1.0 / (1.0 + math.exp(-margin))
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        cap_lp = base_cap
        cap_drift = 0.0
        if base_cap is not None:
            with controller.active():
                cap_lp = _target_logprob(bundle, cap_samples)
            cap_drift = max(0.0, base_cap - cap_lp)
        value = _refusal_loss(proxy, cfg) + _kl_loss(kl, cfg) + cfg.opt_capability_weight * cap_drift
        attrs = {
            "refusal_proxy": round(proxy, 4),
            "kl": round(kl, 4),
        }
        if base_cap is not None:
            attrs.update({
                "capability_logprob": round(cap_lp, 4),
                "capability_drift": round(cap_drift, 4),
            })
        return value, attrs

    best_params, best_attrs, best_value, history = run_search(
        objective, space, cfg.n_trials, cfg.seed,
        early_stop=cfg.opt_early_stop, early_stop_margin=cfg.opt_early_stop_margin,
        adaptive=cfg.adaptive_trials
    )

    # rerank candidates
    if cfg.opt_objective != "generation" and history:
        topk = sorted(history, key=lambda h: h["value"])[: max(1, cfg.opt_rerank_k)]
        best = None
        for h in topk:
            _apply_profile(bundle, controller, ah, al, h["params"], causal_shape, cfg, preserve_basis, preserve_lookup)
            with controller.active():
                ref = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
                cap_lp = _target_logprob(bundle, cap_samples) if base_cap is not None else None
            kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
            cap_drift = max(0.0, base_cap - cap_lp) if base_cap is not None else 0.0
            v = _refusal_loss(ref, cfg) + _kl_loss(kl, cfg) + cfg.opt_capability_weight * cap_drift
            if best is None or v < best[0]:
                attrs = {"refusal": round(ref, 4), "kl": round(kl, 4)}
                if base_cap is not None:
                    attrs.update({
                        "capability_logprob": round(cap_lp, 4),
                        "capability_drift": round(cap_drift, 4),
                    })
                best = (v, h["params"], attrs)
        best_params, best_attrs = best[1], best[2]
    elif history:
        low_kl = _low_kl_pick(history, cfg)
        best_params = low_kl["params"]
        best_attrs = {
            k: low_kl[k]
            for k in ("refusal_proxy", "refusal", "kl", "capability_logprob", "capability_drift")
            if k in low_kl
        }

    _apply_profile(bundle, controller, ah, al, best_params, causal_shape, cfg, preserve_basis, preserve_lookup)
    return best_params, best_attrs, history
