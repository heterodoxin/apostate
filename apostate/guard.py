# reconstruction guard: re-measure leakage and fold corrective directions back in.

from __future__ import annotations

from typing import List, Optional
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .activations import collect_layer_activations, collect_activations
from .directions import refusal_subspace, gram_schmidt_remove, augment_subspace, separation
from .evaluate import strict_refusal_rate as refusal_rate, kl_harmless, refusal_logit_margin


def run_guard(
    bundle: ModelBundle,
    controller: ProjectionController,
    harmful: List[str],
    harmless: List[str],
    cfg,
    direction_layer: int,
    initial_sep: float,
    preserve_basis: Optional[torch.Tensor] = None,
    eval_harmful: Optional[List[str]] = None,
    eval_harmless: Optional[List[str]] = None,
) -> List[dict]:
    history: List[dict] = []
    eval_harmful = eval_harmful or harmful[: cfg.opt_eval_n]
    eval_harmless = eval_harmless or harmless[: cfg.opt_eval_n]

    for it in range(cfg.guard_max_iters):
        with controller.active():
            ah = collect_layer_activations(bundle, harmful, direction_layer, cfg.batch_size)
            al = collect_layer_activations(bundle, harmless, direction_layer, cfg.batch_size)
            ref = refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        sep = separation(ah, al)
        ratio = sep / (initial_sep + 1e-8)
        rank = 0 if controller.R is None else controller.R.shape[1]
        history.append({
            "iter": it, "separation": round(sep, 4), "ratio": round(ratio, 4),
            "rank": rank, "refusal": round(ref, 4), "kl": round(kl, 4),
        })

        if ratio <= cfg.guard_leakage_eps or ref <= cfg.target_refusal:
            break

        prev_R = controller.R.detach().cpu().clone()
        prev_alpha = dict(controller.alpha)

        new_basis, _ = refusal_subspace(
            ah, al,
            rank=cfg.refusal_rank, variance_threshold=cfg.variance_threshold,
            max_rank=cfg.max_rank, seed=cfg.seed,
        )
        new_basis = gram_schmidt_remove(new_basis, preserve_basis)
        merged = augment_subspace(prev_R, new_basis, cfg.max_rank)
        controller.set_subspace(merged)
        for L in range(bundle.num_layers):
            a = controller.get_layer_alpha(L)
            if abs(a) < 1.0:
                sign = -1.0 if a < 0.0 else 1.0
                controller.set_layer_alpha(L, sign * min(1.0, abs(a) + cfg.guard_alpha_step))

        new_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        if new_kl > cfg.max_kl:
            controller.set_subspace(prev_R)
            controller.alpha = prev_alpha
            history[-1]["reverted"] = True
            break

    return history


def run_reader_guard(
    bundle: ModelBundle,
    controller: ProjectionController,
    harmful: List[str],
    harmless: List[str],
    cfg,
    preserve_lookup,
    eval_harmful: List[str],
    eval_harmless: List[str],
    log,
) -> List[dict]:
    # per-layer analog of run_guard for post-norm models: on the edited model,
    # find the residual refusal that survived and add a corrective direction per layer.
    nl = bundle.num_layers
    cap = max(cfg.max_rank, cfg.reader_guard_rank)
    history: List[dict] = []

    def score():
        with controller.active():
            return refusal_rate(bundle, eval_harmful, cfg.opt_gen_tokens, cfg.batch_size)

    m = score()
    kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)

    for it in range(cfg.guard_max_iters):
        log(f"  reader guard {it}: refusal={m:.3f} kl={kl:.3f}")
        history.append({"iter": it, "refusal": round(m, 4), "kl": round(kl, 4)})
        if m <= cfg.target_refusal:
            break

        # snapshot so a step that doesn't help can be undone
        prev_state = controller.alpha_state()
        prev_R = [controller.get_reader_layer_subspace(l) for l in range(nl)]
        with controller.active():
            ah = collect_activations(bundle, harmful, cfg.batch_size)
            al = collect_activations(bundle, harmless, cfg.batch_size)
        for l in range(nl):
            if controller.get_layer_alpha(l) == 0.0:
                continue
            d = ah[l].mean(0) - al[l].mean(0)
            if float(d.norm()) < 1e-6:
                continue
            new = gram_schmidt_remove((d / d.norm()).unsqueeze(1), preserve_lookup(l))
            merged = augment_subspace(prev_R[l], new, cap)
            controller.set_reader_layer_subspace(l, merged)

        new_m = score()
        new_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        # only keep a corrective step that actually lowers refusal, within budget
        if new_m < m - 1e-3 and new_kl <= cfg.reader_max_kl:
            m, kl = new_m, new_kl
        else:
            controller.set_alpha_state(prev_state)
            for l in range(nl):
                if prev_R[l] is not None:
                    controller.set_reader_layer_subspace(l, prev_R[l])
            log(f"  reader guard: no improvement ({new_m:.3f} kl {new_kl:.3f}), reverted")
            break
    return history
