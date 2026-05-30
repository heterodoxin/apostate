"""profile search."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .directions import refusal_subspace, gram_schmidt_remove
import math
from .evaluate import refusal_rate, refusal_logit_margin, kl_harmless
from .search import run_search


def _apply_profile(
    bundle: ModelBundle,
    controller: ProjectionController,
    ah: torch.Tensor,
    al: torch.Tensor,
    params: Dict,
    causal_shape: List[float],
    cfg,
    preserve_basis: Optional[torch.Tensor],
) -> int:
    """Configure the controller from a trial's params. Returns the direction layer."""
    n = bundle.num_layers
    L_dir = max(0, min(n - 1, int(n * params["direction_layer_frac"])))
    R, _ = refusal_subspace(
        ah[L_dir], al[L_dir],
        rank=int(params["refusal_rank"]), max_rank=cfg.max_rank, seed=cfg.seed,
    )
    R = gram_schmidt_remove(R, preserve_basis)
    controller.set_subspace(R)

    lo, hi = sorted((params["band_lo"], params["band_hi"]))
    strength = params["strength"]
    cmix = params["causal_mix"]
    for L in range(n):
        frac = L / max(1, n - 1)
        if lo <= frac <= hi:
            shape = (1.0 - cmix) + cmix * causal_shape[L]
            controller.set_layer_alpha(L, strength * shape)
        else:
            controller.set_layer_alpha(L, 0.0)
    controller.set_embed_alpha(strength if params["ablate_embed"] else 0.0)
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
) -> Tuple[dict, dict, list]:
    space = {
        "direction_layer_frac": ("float", 0.35, 0.80),
        "refusal_rank": ("int", 1, min(3, cfg.max_rank)),
        "strength": ("float", 0.40, 1.80),
        "band_lo": ("float", 0.00, 0.35),
        "band_hi": ("float", 0.65, 1.00),
        "causal_mix": ("float", 0.0, 1.0),
        "ablate_embed": ("cat", [True, False]),
    }

    def objective(params):
        _apply_profile(bundle, controller, ah, al, params, causal_shape, cfg, preserve_basis)
        with controller.active():
            if cfg.opt_objective == "generation":
                proxy = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
            else:
                # one forward pass; squash the refusal-logit margin into [0,1] so it
                # composes with KL (the guard + refine then verify real refusal).
                margin = refusal_logit_margin(bundle, eval_harmful, cfg.batch_size)
                proxy = 1.0 / (1.0 + math.exp(-margin))
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size)
        penalty = max(0.0, kl - cfg.max_kl) * 2.0   # respect the KL budget
        value = proxy + cfg.kl_weight * kl + penalty
        return value, {"refusal_proxy": round(proxy, 4), "kl": round(kl, 4)}

    best_params, best_attrs, best_value, history = run_search(
        objective, space, cfg.n_trials, cfg.seed,
        early_stop=cfg.opt_early_stop, early_stop_margin=cfg.opt_early_stop_margin,
        adaptive=cfg.adaptive_trials
    )

    # top-K candidates by REAL generation refusal so the final pick matches the
    if cfg.opt_objective != "generation" and history:
        topk = sorted(history, key=lambda h: h["value"])[: max(1, cfg.opt_rerank_k)]
        best = None
        for h in topk:
            _apply_profile(bundle, controller, ah, al, h["params"], causal_shape, cfg, preserve_basis)
            with controller.active():
                ref = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
            kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size)
            v = ref + cfg.kl_weight * kl + max(0.0, kl - cfg.max_kl) * 2.0
            if best is None or v < best[0]:
                best = (v, h["params"], {"refusal": round(ref, 4), "kl": round(kl, 4)})
        best_params, best_attrs = best[1], best[2]

    _apply_profile(bundle, controller, ah, al, best_params, causal_shape, cfg, preserve_basis)
    return best_params, best_attrs, history
