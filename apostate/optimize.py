"""profile search"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .directions import refusal_subspace, gram_schmidt_remove
import math
from .evaluate import refusal_rate, refusal_rate_bounded, refusal_logit_margin, kl_harmless
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
    ref = float(h.get("refusal", h.get("refusal_proxy", 1.0)))
    if h.get("refusal_complete") is False:
        return min(1.0, ref + 0.15)
    return ref


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
        best_under = min(under_budget, key=lambda h: (
            _ref_attr(h), float(h.get("capability_drift", 0.0)), float(h.get("kl", 99.0)), h["value"]
        ))
        repair_cap = cfg.max_kl * 3.0
        needed_gain = max(0.08, _ref_attr(best_under) * 0.25)
        repairable = [
            h for h in history
            if float(h.get("kl", 99.0)) <= repair_cap
            and _ref_attr(h) + needed_gain < _ref_attr(best_under)
            and float(h.get("capability_drift", 0.0)) <= max(0.05, float(best_under.get("capability_drift", 0.0)) + 0.05)
        ]
        if repairable:
            return min(repairable, key=lambda h: (
                _ref_attr(h), max(0.0, float(h.get("kl", 99.0)) - cfg.max_kl),
                float(h.get("capability_drift", 0.0)), float(h.get("kl", 99.0)), h["value"]
            ))
        return best_under
    repairable = [h for h in history if float(h.get("kl", 99.0)) <= cfg.max_kl * 3.0]
    if repairable:
        return min(repairable, key=lambda h: (
            _ref_attr(h), max(0.0, float(h.get("kl", 99.0)) - cfg.max_kl),
            float(h.get("capability_drift", 0.0)), h["value"]
        ))
    return min(history, key=lambda h: h["value"])


def _candidate_pool(history: list, cfg) -> list:
    seen = set()
    pool = []

    def add(rows):
        for h in rows:
            params = h.get("params", {})
            sig = tuple((k, params[k]) for k in sorted(params))
            if sig in seen:
                continue
            seen.add(sig)
            pool.append(h)

    k = max(1, cfg.opt_rerank_k)
    add(sorted(history, key=lambda h: h["value"])[:k])
    add(sorted(history, key=lambda h: (_ref_attr(h), float(h.get("kl", 99.0)), h["value"]))[: max(3, k)])
    add(sorted(
        [h for h in history if float(h.get("kl", 99.0)) <= cfg.max_kl * 3.0],
        key=lambda h: (_ref_attr(h), max(0.0, float(h.get("kl", 99.0)) - cfg.max_kl), h["value"]),
    )[: max(3, k)])
    return pool


def _anchor_profiles(bundle: ModelBundle, space: dict) -> list:
    if bundle.can_edit_embed():
        return []
    rank_hi = int(space["refusal_rank"][2])
    strength_hi = float(space["strength"][2])
    rows = []
    for direction_sign in (1.0, -1.0):
        for direction_layer_frac, rank, band_center, band_width, strength, causal_mix, causal_power in (
            (0.58, 1, 0.58, 0.78, 1.15, 0.25, 1.50),
            (0.62, min(2, rank_hi), 0.62, 0.82, 1.35, 0.20, 1.25),
            (0.70, rank_hi, 0.76, 0.72, strength_hi, 0.45, 2.00),
        ):
            rows.append({
                "direction_layer_frac": direction_layer_frac,
                "refusal_rank": rank,
                "strength": min(strength, strength_hi),
                "band_center": band_center,
                "band_width": band_width,
                "causal_mix": causal_mix,
                "causal_power": causal_power,
                "ablate_embed": False,
                "direction_sign": direction_sign,
            })
    return rows


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


def _capability_batches(bundle: ModelBundle, samples: List[Tuple[str, str]], batch_size: int):
    tok = bundle.tokenizer
    model = bundle.model
    device = next(model.parameters()).device
    cache = getattr(bundle, "_cap_cache", None)
    if cache is None:
        cache = {}
        setattr(bundle, "_cap_cache", cache)
    key = (tuple(samples), batch_size, str(device))
    if key in cache:
        return cache[key]

    pad = tok.pad_token_id
    if pad is None:
        pad = tok.eos_token_id or 0
    rows = []
    for prompt, target in samples:
        prompt_text = format_chat(tok, [prompt])[0]
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids
        if not prompt_ids or not target_ids:
            continue
        ids = prompt_ids + target_ids
        start = max(0, len(prompt_ids) - 1)
        end = start + len(target_ids)
        rows.append((ids, start, end))

    batches = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        max_len = max(len(ids) for ids, _start, _end in chunk)
        input_ids = torch.full((len(chunk), max_len), pad, dtype=torch.long, device=device)
        mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        target = torch.zeros((len(chunk), max_len - 1), dtype=torch.bool, device=device)
        for row, (ids, start, end) in enumerate(chunk):
            n = len(ids)
            input_ids[row, :n] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[row, :n] = 1
            target[row, start:end] = True
        batches.append((input_ids, mask, target))

    cache[key] = batches
    return batches


@torch.inference_mode()
def _target_logprob(bundle: ModelBundle, samples: List[Tuple[str, str]], batch_size: int = 8) -> float:
    if not samples:
        return 0.0
    model = bundle.model
    vals: List[float] = []
    for input_ids, mask, target_mask in _capability_batches(bundle, samples, batch_size):
        logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        for row in range(input_ids.shape[0]):
            sol_logp = tok_logp[row][target_mask[row]]
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
    strength = params["strength"] * params.get("direction_sign", 1.0)
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
    controller.set_embed_alpha(strength * embed_scale if params.get("ablate_embed", False) else 0.0)
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
            base_cap = _target_logprob(bundle, cap_samples, cfg.batch_size)
        print(f"[apostate] capability logprob baseline: {base_cap:.4f}", flush=True)

    strength_hi = 1.70 if not bundle.can_edit_embed() else 1.25
    width_hi = 0.82 if not bundle.can_edit_embed() else 0.65
    space = {
        "direction_layer_frac": ("float", 0.30, 0.82),
        "refusal_rank": ("int", 1, min(3, cfg.max_rank)),
        "strength": ("float", 0.08, strength_hi),
        "band_center": ("float", 0.15, 0.90),
        "band_width": ("float", 0.08, width_hi),
        "causal_mix": ("float", 0.0, 1.0),
        "causal_power": ("float", 1.0, 3.0),
        "direction_sign": ("cat", [1.0, -1.0]),
    }
    if bundle.can_edit_embed():
        space["ablate_embed"] = ("cat", [False, True])
        space["embed_scale"] = ("float", 0.0, 0.35)
    else:
        space["ablate_embed"] = ("cat", [False])
        print("[apostate] embed edit disabled: per-layer embeddings", flush=True)

    best_seen = [float("inf")]

    def objective(params):
        _apply_profile(bundle, controller, ah, al, params, causal_shape, cfg, preserve_basis, preserve_lookup)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        kl_part = _kl_loss(kl, cfg)
        if kl_part >= best_seen[0] - 1e-4:
            return kl_part, {"refusal_proxy": 1.0, "refusal_complete": False, "kl": round(kl, 4)}

        with controller.active():
            if cfg.opt_objective == "generation":
                proxy, complete = refusal_rate_bounded(
                    bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size,
                    should_stop=lambda floor, _seen, _total: (
                        _refusal_loss(floor, cfg) + kl_part >= best_seen[0] - 1e-4
                    ),
                )
            else:
                margin = refusal_logit_margin(bundle, eval_harmful, cfg.batch_size)
                proxy = 1.0 / (1.0 + math.exp(-margin))
                complete = True
        cap_lp = base_cap
        cap_drift = 0.0
        if base_cap is not None:
            with controller.active():
                cap_lp = _target_logprob(bundle, cap_samples, cfg.batch_size)
            cap_drift = max(0.0, base_cap - cap_lp)
        value = _refusal_loss(proxy, cfg) + kl_part + cfg.opt_capability_weight * cap_drift
        if value < best_seen[0]:
            best_seen[0] = value
        attrs = {
            "refusal_proxy": round(proxy, 4),
            "refusal_complete": complete,
            "kl": round(kl, 4),
        }
        if base_cap is not None:
            attrs.update({
                "capability_logprob": round(cap_lp, 4),
                "capability_drift": round(cap_drift, 4),
            })
        return value, attrs

    anchor_history = []
    anchors = _anchor_profiles(bundle, space)
    for idx, params in enumerate(anchors, 1):
        print(f"\n[Seed {idx}/{len(anchors)}]")
        print(f"  Parameters: {params}")
        value, attrs = objective(params)
        print(f"  Metrics: {attrs}")
        print(f"  Loss: {value:.6f}")
        anchor_history.append({"params": params, "value": value, **attrs})

    best_params, best_attrs, best_value, history = run_search(
        objective, space, cfg.n_trials, cfg.seed,
        early_stop=cfg.opt_early_stop, early_stop_margin=cfg.opt_early_stop_margin,
        adaptive=cfg.adaptive_trials
    )
    history = anchor_history + history

    # exact rerank
    if history:
        exact = []
        pool = _candidate_pool(history, cfg)
        for idx, h in enumerate(pool, 1):
            _apply_profile(bundle, controller, ah, al, h["params"], causal_shape, cfg, preserve_basis, preserve_lookup)
            with controller.active():
                ref = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
                cap_lp = _target_logprob(bundle, cap_samples, cfg.batch_size) if base_cap is not None else None
            kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
            cap_drift = max(0.0, base_cap - cap_lp) if base_cap is not None else 0.0
            v = _refusal_loss(ref, cfg) + _kl_loss(kl, cfg) + cfg.opt_capability_weight * cap_drift
            print(f"[apostate] exact rerank {idx}/{len(pool)}: refusal={ref:.3f} kl={kl:.3f}", flush=True)
            item = {
                "params": h["params"],
                "value": v,
                "refusal": round(ref, 4),
                "refusal_complete": True,
                "kl": round(kl, 4),
            }
            if base_cap is not None:
                item.update({
                    "capability_logprob": round(cap_lp, 4),
                    "capability_drift": round(cap_drift, 4),
                })
            exact.append(item)
        low_kl = _low_kl_pick(exact or history, cfg)
        best_params = low_kl["params"]
        best_attrs = {
            k: low_kl[k]
            for k in ("refusal_proxy", "refusal", "kl", "capability_logprob", "capability_drift")
            if k in low_kl
        }

    _apply_profile(bundle, controller, ah, al, best_params, causal_shape, cfg, preserve_basis, preserve_lookup)
    return best_params, best_attrs, history
