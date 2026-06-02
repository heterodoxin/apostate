from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .directions import refusal_subspace, gram_schmidt_remove
import math
from .evaluate import (
    _COMPLY_STARTS,
    _REFUSAL_STARTS,
    _first_token_ids,
    strict_refusal_rate as refusal_rate,
    strict_refusal_rate_bounded as refusal_rate_bounded,
    refusal_logit_margin,
    kl_harmless,
)
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


_HEAD_SUBSPACE_CACHE: dict = {}


def _head_token_subspace(bundle: ModelBundle) -> torch.Tensor:
    key = id(bundle.model)
    cached = _HEAD_SUBSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    head = bundle.lm_head()
    if head is None:
        raise AttributeError("no lm head")
    tok = bundle.tokenizer
    device = next(bundle.model.parameters()).device
    refusal_ids = torch.tensor(_first_token_ids(tok, _REFUSAL_STARTS), dtype=torch.long)
    comply_ids = torch.tensor(_first_token_ids(tok, _COMPLY_STARTS), dtype=torch.long)
    W = head.weight.detach().float().cpu()
    direction = W[refusal_ids].mean(0) - W[comply_ids].mean(0)
    direction = direction / (direction.norm() + 1e-8)
    cached = direction.unsqueeze(1).to(device=device)
    _HEAD_SUBSPACE_CACHE[key] = cached
    return cached


def _anchor_profiles(bundle: ModelBundle, space: dict) -> list:
    if bundle.can_edit_embed():
        return []
    rank_hi = int(space["refusal_rank"][2])
    strength_hi = float(space["strength"][2])
    head_ok = "ablate_head" in space
    direction_opts = space.get("direction_source", ("cat", []))[1]
    head_tokens_ok = "head_tokens" in direction_opts
    rows = []
    if "head_alpha" in space and head_tokens_ok:
        for alpha in (3.5, 4.0, 4.5, 4.65, 5.0):
            rows.append({
                "direction_source": "head_tokens",
                "direction_layer_frac": 0.58,
                "refusal_rank": 1,
                "strength": 0.0,
                "band_center": 0.58,
                "band_width": 0.78,
                "causal_mix": 0.0,
                "causal_power": 1.0,
                "ablate_embed": False,
                "direction_sign": 1.0,
                "ablate_head": True,
                "head_scale": 0.0,
                "head_alpha": alpha,
            })
    for direction_sign in (1.0, -1.0):
        for direction_layer_frac, rank, band_center, band_width, strength, causal_mix, causal_power, head, head_scale in (
            (0.58, 1, 0.58, 0.78, 1.15, 0.25, 1.50, False, 0.0),
            (0.58, 1, 0.58, 0.78, 0.85, 0.25, 1.50, True, 0.55),
            (0.62, min(2, rank_hi), 0.62, 0.82, 1.35, 0.20, 1.25, False, 0.0),
            (0.70, rank_hi, 0.76, 0.72, strength_hi, 0.45, 2.00, False, 0.0),
        ):
            rows.append({
                "direction_source": "activations",
                "direction_layer_frac": direction_layer_frac,
                "refusal_rank": rank,
                "strength": min(strength, strength_hi),
                "band_center": band_center,
                "band_width": band_width,
                "causal_mix": causal_mix,
                "causal_power": causal_power,
                "ablate_embed": False,
                "direction_sign": direction_sign,
                "ablate_head": head if head_ok else False,
                "head_scale": head_scale if head_ok else 0.0,
            })
    if "embed_scale" in space and not bundle.can_edit_embed():
        for direction_sign in (1.0, -1.0):
            for strength, embed_scale, head_scale in (
                (0.85, 0.06, 0.20),
                (1.15, 0.08, 0.15),
                (1.35, 0.10, 0.00),
            ):
                rows.append({
                    "direction_source": "activations",
                    "direction_layer_frac": 0.58,
                    "refusal_rank": 1,
                    "strength": strength,
                    "band_center": 0.58,
                    "band_width": 0.78,
                    "causal_mix": 0.25,
                    "causal_power": 1.5,
                    "ablate_embed": True,
                    "embed_scale": embed_scale,
                    "head_token_alpha": 2.5 if head_scale > 0.0 else 0.0,
                    "direction_sign": direction_sign,
                    "ablate_head": head_ok and head_scale > 0.0,
                    "head_scale": head_scale if head_ok else 0.0,
                })
    if "head_token_alpha" in space and not bundle.can_edit_embed():
        for strength, head_alpha in (
            (0.50, 2.5),
            (0.65, 3.5),
            (0.80, 3.5),
        ):
            rows.append({
                "direction_source": "activations",
                "direction_layer_frac": 0.58,
                "refusal_rank": min(2, rank_hi),
                "strength": min(strength, strength_hi),
                "band_center": 0.58,
                "band_width": 0.12,
                "causal_mix": 0.0,
                "causal_power": 1.0,
                "ablate_embed": False,
                "embed_scale": 0.0,
                "head_token_alpha": head_alpha,
                "direction_sign": 1.0,
                "ablate_head": False,
                "head_scale": 0.0,
            })
    if "ple_ablate" in space:
        for direction_sign in (1.0, -1.0):
            for ple_layer_frac, ple_strength, ple_band_center, ple_band_width, head, head_scale in (
                (0.18, 0.65, 0.18, 0.45, False, 0.0),
                (0.58, 0.75, 0.58, 0.55, False, 0.0),
                (0.82, 0.90, 0.78, 0.42, False, 0.0),
                (0.58, 0.55, 0.58, 0.55, True, 0.15),
            ):
                rows.append({
                    "direction_source": "activations",
                    "direction_layer_frac": ple_layer_frac,
                    "refusal_rank": 1,
                    "strength": 0.0,
                    "band_center": ple_band_center,
                    "band_width": ple_band_width,
                    "causal_mix": 0.25,
                    "causal_power": 1.5,
                    "ablate_embed": False,
                    "direction_sign": direction_sign,
                    "ablate_head": head if head_ok else False,
                    "head_scale": head_scale if head_ok else 0.0,
                    "ple_ablate": True,
                    "ple_layer_frac": ple_layer_frac,
                    "ple_rank": 1,
                    "ple_strength": ple_strength,
                    "ple_band_center": ple_band_center,
                    "ple_band_width": ple_band_width,
                })
    if "ple_embed_ablate" in space:
        for direction_sign in (1.0, -1.0):
            for alpha in (0.04, 0.08, 0.12):
                rows.append({
                    "direction_source": "activations",
                    "direction_layer_frac": 0.58,
                    "refusal_rank": 1,
                    "strength": 0.0,
                    "band_center": 0.58,
                    "band_width": 0.55,
                    "causal_mix": 0.25,
                    "causal_power": 1.5,
                    "ablate_embed": False,
                    "direction_sign": direction_sign,
                    "ablate_head": False,
                    "head_scale": 0.0,
                    "ple_embed_ablate": True,
                    "ple_embed_alpha": alpha,
                    "ple_model_projection_ablate": False,
                    "ple_model_projection_alpha": 0.0,
                })
    if bundle.has_ple():
        def priority(row):
            if float(row.get("head_token_alpha", 0.0)) > 0.0:
                return (0, -float(row.get("head_token_alpha", 0.0)))
            if row.get("direction_source") == "head_tokens":
                return (1, abs(float(row.get("head_alpha", 0.0)) - 4.65))
            if row.get("ablate_embed"):
                return (2, float(row.get("strength", 99.0)))
            return (3, float(row.get("strength", 99.0)))

        rows = sorted(rows, key=priority)[:8]
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
    ple_ah: Optional[torch.Tensor] = None,
    ple_al: Optional[torch.Tensor] = None,
    ple_preserve_lookup: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    plee_ah: Optional[torch.Tensor] = None,
    plee_al: Optional[torch.Tensor] = None,
    plee_preserve_basis: Optional[torch.Tensor] = None,
) -> int:
    n = bundle.num_layers
    L_dir = max(0, min(n - 1, int(n * params["direction_layer_frac"])))
    if params.get("direction_source") == "head_tokens":
        R = _head_token_subspace(bundle)
    else:
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
    if params.get("ablate_head", False):
        if "head_alpha" in params:
            controller.set_head_alpha(params["head_alpha"] * params.get("direction_sign", 1.0))
        else:
            head_scale = params.get("head_scale", 1.0)
            controller.set_head_alpha(strength * head_scale)
    else:
        controller.set_head_alpha(0.0)
    if hasattr(controller, "set_head_token_alpha"):
        controller.set_head_token_alpha(0.0)
        if float(params.get("head_token_alpha", 0.0)) != 0.0:
            controller.set_head_token_subspace(_head_token_subspace(bundle))
            controller.set_head_token_alpha(float(params.get("head_token_alpha", 0.0)))

    controller.clear_ple()
    if params.get("ple_ablate", False) and ple_ah is not None and ple_al is not None and controller.has_ple():
        L_ple = max(0, min(n - 1, int(n * params.get("ple_layer_frac", params["direction_layer_frac"]))))
        ple_rank = int(params.get("ple_rank", 1))
        Rple, _ = refusal_subspace(
            ple_ah[L_ple], ple_al[L_ple],
            rank=ple_rank, max_rank=max(1, min(int(getattr(cfg, "ple_max_rank", 2)), ple_rank)),
            seed=cfg.seed,
        )
        basis = ple_preserve_lookup(L_ple) if ple_preserve_lookup is not None else None
        Rple = gram_schmidt_remove(Rple, basis)
        controller.set_ple_subspace(Rple)
        width = params.get("ple_band_width", params.get("band_width", 0.5))
        center = params.get("ple_band_center", params.get("band_center", 0.5))
        lo = max(0.0, center - width * 0.5)
        hi = min(1.0, center + width * 0.5)
        strength = params.get("ple_strength", 0.0) * params.get("direction_sign", 1.0)
        cmix = params.get("causal_mix", 0.0)
        power = params.get("causal_power", 1.0)
        for L in range(n):
            frac = L / max(1, n - 1)
            if lo <= frac <= hi:
                shape = (1.0 - cmix) + cmix * (max(0.0, causal_shape[L]) ** power)
                controller.set_ple_layer_alpha(L, strength * shape)
    if params.get("ple_embed_ablate", False) and plee_ah is not None and plee_al is not None:
        Rplee, _ = refusal_subspace(
            plee_ah, plee_al,
            rank=1, max_rank=1, seed=cfg.seed,
        )
        Rplee = gram_schmidt_remove(Rplee, plee_preserve_basis)
        controller.set_ple_embed_subspace(Rplee)
        controller.set_ple_embed_alpha(params.get("ple_embed_alpha", 0.0) * params.get("direction_sign", 1.0))
    if params.get("ple_model_projection_ablate", False) and plee_ah is not None and plee_al is not None:
        Rplep, _ = refusal_subspace(
            plee_ah, plee_al,
            rank=1, max_rank=1, seed=cfg.seed,
        )
        Rplep = gram_schmidt_remove(Rplep, plee_preserve_basis)
        controller.set_ple_model_projection_subspace(Rplep)
        controller.set_ple_model_projection_alpha(
            params.get("ple_model_projection_alpha", 0.0) * params.get("direction_sign", 1.0)
        )
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
    ple_ah: Optional[torch.Tensor] = None,
    ple_al: Optional[torch.Tensor] = None,
    ple_preserve_lookup: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
    plee_ah: Optional[torch.Tensor] = None,
    plee_al: Optional[torch.Tensor] = None,
    plee_preserve_basis: Optional[torch.Tensor] = None,
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
        "direction_source": ("cat", ["activations"]),
        "direction_layer_frac": ("float", 0.30, 0.82),
        "refusal_rank": ("int", 1, min(3, cfg.max_rank)),
        "strength": ("float", 0.0, strength_hi),
        "band_center": ("float", 0.15, 0.90),
        "band_width": ("float", 0.08, width_hi),
        "causal_mix": ("float", 0.0, 1.0),
        "causal_power": ("float", 1.0, 3.0),
        "direction_sign": ("cat", [1.0, -1.0]),
    }
    embed_ok = bundle.can_edit_embed() or bundle.has_ple()
    if embed_ok:
        space["ablate_embed"] = ("cat", [False, True])
        space["embed_scale"] = ("float", 0.0, 0.35 if bundle.can_edit_embed() else 0.14)
    else:
        space["ablate_embed"] = ("cat", [False])
        print("[apostate] embed edit disabled: per-layer embeddings", flush=True)
    if bundle.final_norm() is not None and bundle.lm_head() is not None:
        space["ablate_head"] = ("cat", [False, True])
        space["head_scale"] = ("float", 0.0, 0.75 if not bundle.can_edit_embed() else 0.35)
        space["head_alpha"] = ("float", 0.0, 6.0 if not bundle.can_edit_embed() else 2.0)
        if not bundle.can_edit_embed() and getattr(cfg, "head_sweep", True):
            space["direction_source"] = ("cat", ["activations", "head_tokens"])
            space["head_token_alpha"] = ("float", 0.0, 4.5)
    else:
        space["ablate_head"] = ("cat", [False])
    if (
        getattr(cfg, "gemma_ple", True)
        and ple_ah is not None
        and ple_al is not None
        and controller.has_ple()
    ):
        space["ple_ablate"] = ("cat", [False, True])
        space["ple_layer_frac"] = ("float", 0.0, 0.95)
        space["ple_rank"] = ("int", 1, max(1, min(2, int(getattr(cfg, "ple_max_rank", 2)))))
        space["ple_strength"] = ("float", 0.0, 1.25)
        space["ple_band_center"] = ("float", 0.0, 0.95)
        space["ple_band_width"] = ("float", 0.08, 0.90)
    if (
        getattr(cfg, "gemma_ple", True)
        and plee_ah is not None
        and plee_al is not None
    ):
        space["ple_embed_ablate"] = ("cat", [False, True])
        space["ple_embed_alpha"] = ("float", 0.0, 0.18)
        space["ple_model_projection_ablate"] = ("cat", [False, True])
        space["ple_model_projection_alpha"] = ("float", 0.0, 0.16)

    best_seen = [float("inf")]

    def objective(params):
        _apply_profile(
            bundle, controller, ah, al, params, causal_shape, cfg, preserve_basis, preserve_lookup,
            ple_ah=ple_ah, ple_al=ple_al, ple_preserve_lookup=ple_preserve_lookup,
            plee_ah=plee_ah, plee_al=plee_al, plee_preserve_basis=plee_preserve_basis,
        )
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

    if history:
        exact = []
        pool = _candidate_pool(history, cfg)
        for idx, h in enumerate(pool, 1):
            _apply_profile(
                bundle, controller, ah, al, h["params"], causal_shape, cfg, preserve_basis, preserve_lookup,
                ple_ah=ple_ah, ple_al=ple_al, ple_preserve_lookup=ple_preserve_lookup,
                plee_ah=plee_ah, plee_al=plee_al, plee_preserve_basis=plee_preserve_basis,
            )
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

    _apply_profile(
        bundle, controller, ah, al, best_params, causal_shape, cfg, preserve_basis, preserve_lookup,
        ple_ah=ple_ah, ple_al=ple_al, ple_preserve_lookup=ple_preserve_lookup,
        plee_ah=plee_ah, plee_al=plee_al, plee_preserve_basis=plee_preserve_basis,
    )
    return best_params, best_attrs, history
