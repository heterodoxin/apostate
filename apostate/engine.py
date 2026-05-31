"""ablation pipeline."""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Optional

import torch

from .config import ApostateConfig
from .model import load_model
from .data import resolve_prompts
from .activations import collect_activations
from .directions import refusal_subspace, preservation_subspace, gram_schmidt_remove, separation
from .projectors import ProjectionController
from .causal import causal_layer_scores
from .guard import run_guard
from .evaluate import refusal_rate, kl_harmless
from .optimize import optimize_profile
from .search import _has_optuna
from .bake import bake


def _log(msg: str):
    print(f"[apostate] {msg}", flush=True)


def _refine_refusal(bundle, controller, cfg, eval_harmful, eval_harmless):
    """escalate edit to clear residual refusals."""
    base = dict(controller.alpha)
    es = eval_harmful[: max(48, cfg.opt_eval_n)]
    el = eval_harmless[: cfg.opt_eval_n]
    with controller.active():
        ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
    kl = kl_harmless(bundle, controller, el, cfg.batch_size)
    if ref <= cfg.target_refusal:
        if not cfg.refine_deescalate:
            return ref, kl
        # shrink alpha; claw back kl
        best = (ref, kl, dict(base))
        for s in (0.9, 0.8, 0.7):
            for mid, a in base.items():
                controller.alpha[mid] = a * s
            with controller.active():
                new_ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
            if new_ref > cfg.target_refusal:
                break
            new_kl = kl_harmless(bundle, controller, el, cfg.batch_size)
            best = (new_ref, new_kl, dict(controller.alpha))
            _log(f"  refine(down): scale={s:.2f} refusal={new_ref:.3f} kl={new_kl:.3f} (kept)")
        controller.alpha = best[2]
        return best[0], best[1]
    best = (ref, kl, dict(base))
    s = 1.0
    step = (cfg.refine_max_scale - 1.0) / max(1, cfg.refine_steps)
    for _ in range(cfg.refine_steps):
        s += step
        for mid, a in base.items():
            controller.alpha[mid] = min(cfg.refine_max_scale, a * s)
        new_kl = kl_harmless(bundle, controller, el, cfg.batch_size)
        if new_kl > cfg.max_kl:
            break
        with controller.active():
            new_ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
        if new_ref < best[0] - 1e-6:                  # keep
            best = (new_ref, new_kl, dict(controller.alpha))
            _log(f"  refine: scale={s:.2f} refusal={new_ref:.3f} kl={new_kl:.3f} (kept)")
            if new_ref <= cfg.target_refusal:
                break
        else:
            break
    controller.alpha = best[2]
    return best[0], best[1]


def run(cfg: ApostateConfig) -> dict:
    t0 = time.time()
    cfg.with_defaults()
    os.makedirs(cfg.output_dir, exist_ok=True)

    _log(f"loading {cfg.model} (4bit={cfg.load_in_4bit}) ...")
    bundle = load_model(cfg)
    tok = bundle.tokenizer
    L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * cfg.direction_layer_frac)))
    nw = len(bundle.layer_writers(bundle.layers()[L_dir]))
    arch = f"MoE ({nw} writers/layer)" if bundle.is_moe() else "dense"
    _log(f"{bundle.num_layers} layers, hidden={bundle.hidden_size}, {arch}, direction layer={L_dir}")

    harmful = resolve_prompts(cfg.harmful_path, cfg.n_harmful + cfg.n_eval, cfg.seed)
    harmless = resolve_prompts(cfg.harmless_path, cfg.n_harmless, cfg.seed)
    fit_harmful = harmful[: cfg.n_harmful]
    if cfg.harmful_test:
        eval_harmful = resolve_prompts(cfg.harmful_test, cfg.n_eval, cfg.seed)
    else:
        tail = harmful[cfg.n_harmful : cfg.n_harmful + cfg.n_eval]
        eval_harmful = tail if len(tail) >= cfg.n_eval else harmful[: cfg.n_eval]
    if cfg.harmless_test:
        eval_harmless = resolve_prompts(cfg.harmless_test, cfg.n_eval, cfg.seed)
    else:
        tail = harmless[cfg.n_harmless - cfg.n_eval : cfg.n_harmless]
        eval_harmless = tail if len(tail) >= cfg.n_eval else harmless[: cfg.n_eval]
    # split val (tuning) / test (honest report) so hyperparams never see test
    hh = max(1, len(eval_harmful) // 2)
    hl = max(1, len(eval_harmless) // 2)
    test_harmful, eval_harmful = eval_harmful[:hh], eval_harmful[hh:]
    test_harmless, eval_harmless = eval_harmless[:hl], eval_harmless[hl:]
    _log(f"prompts: {len(fit_harmful)} harmful (fit), {len(harmless)} harmless, "
         f"val {len(eval_harmful)}/{len(eval_harmless)}, test {len(test_harmful)}/{len(test_harmless)}")

    controller = ProjectionController(bundle)
    controller.disable()

    base_refusal = refusal_rate(bundle, test_harmful, cfg.max_new_tokens, cfg.batch_size)
    _log(f"baseline refusal rate (test): {base_refusal:.3f}")

    _log("collecting activations (original model) ...")
    ah = collect_activations(bundle, fit_harmful, cfg.batch_size)
    al = collect_activations(bundle, harmless, cfg.batch_size)

    preserve_basis = None
    if cfg.preserve_path:
        preserve = resolve_prompts(cfg.preserve_path, cfg.n_harmless, cfg.seed)
        ap = collect_activations(bundle, preserve, cfg.batch_size)
        preserve_basis = preservation_subspace(ap[L_dir], rank=cfg.preserve_rank)
        _log(f"preservation subspace rank={preserve_basis.shape[1]}")

    report_extra: dict = {"optimized": cfg.optimize}

    if cfg.optimize:
        Rseed, _ = refusal_subspace(ah[L_dir], al[L_dir], rank=1, max_rank=cfg.max_rank, seed=cfg.seed)
        controller.set_subspace(gram_schmidt_remove(Rseed, preserve_basis))
        if cfg.causal_targeting:
            _log("scoring per-layer causal importance (prior) ...")
            causal_shape = causal_layer_scores(
                bundle, controller, eval_harmful[: cfg.opt_eval_n], cfg.batch_size,
                floor=cfg.causal_floor, temperature=cfg.causal_temperature,
            )
        else:
            causal_shape = [1.0] * bundle.num_layers
        _log(f"optimizing ablation profile via {'TPE' if _has_optuna() else 'random'} search: {cfg.n_trials} trials ...")
        best_params, best_attrs, opt_hist = optimize_profile(
            bundle, controller, ah, al,
            eval_harmful[: cfg.opt_eval_n], eval_harmless,
            causal_shape, cfg, preserve_basis,
        )
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()}
        _log(f"best trial: refusal_proxy={best_attrs.get('refusal_proxy', best_attrs.get('refusal'))} "
             f"kl={best_attrs.get('kl')} | {shown}")
        L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * best_params["direction_layer_frac"])))
        report_extra.update({"best_params": best_params, "best_trial": best_attrs, "n_trials": cfg.n_trials})
    else:
        R, svals = refusal_subspace(
            ah[L_dir], al[L_dir],
            rank=cfg.refusal_rank, variance_threshold=cfg.variance_threshold,
            max_rank=cfg.max_rank, seed=cfg.seed,
        )
        _log(f"refusal subspace rank={R.shape[1]} (svals={[round(float(s),2) for s in svals]})")
        controller.set_subspace(gram_schmidt_remove(R, preserve_basis))
        if cfg.causal_targeting:
            _log("scoring per-layer causal importance ...")
            alphas = causal_layer_scores(
                bundle, controller, eval_harmful, cfg.batch_size,
                floor=cfg.causal_floor, temperature=cfg.causal_temperature,
            )
            for L in range(bundle.num_layers):
                controller.set_layer_alpha(L, alphas[L])
            controller.set_embed_alpha(1.0)
            top = sorted(range(len(alphas)), key=lambda i: -alphas[i])[:5]
            _log(f"top causal layers: {[(i, round(alphas[i],2)) for i in top]}")
        else:
            controller.set_uniform_alpha(1.0)

    initial_sep = separation(ah[L_dir], al[L_dir])
    controller.enable()

    guard_hist = []
    skip_guard = False
    if ((not cfg.optimize) or cfg.opt_guard) and cfg.opt_early_stop:
        with controller.active():
            ref_quick = refusal_rate(bundle, eval_harmful[:min(24, len(eval_harmful))], cfg.max_new_tokens, cfg.batch_size)
        skip_guard = ref_quick <= cfg.target_refusal
    if ((not cfg.optimize) or cfg.opt_guard) and not skip_guard:
        _log("running reconstruction guard ...")
        gcap = max(256, cfg.opt_eval_n)   # guard activation subset (separation estimate)
        guard_hist = run_guard(
            bundle, controller, fit_harmful[:gcap], harmless[:gcap], cfg, L_dir, initial_sep,
            preserve_basis,
            eval_harmful=eval_harmful[: cfg.opt_eval_n], eval_harmless=eval_harmless[: cfg.opt_eval_n],
        )
        for h in guard_hist:
            _log(f"  guard iter {h['iter']}: sep={h['separation']} ratio={h['ratio']} "
                 f"rank={h['rank']} refusal={h.get('refusal')} kl={h.get('kl')}")
    elif skip_guard:
        _log(f"guard: skipped (refusal {ref_quick:.3f} <= target {cfg.target_refusal:.3f})")

    should_refine = cfg.refine_refusal and (skip_guard or (len(guard_hist) > 0 and guard_hist[-1].get("refusal", 1.0) > cfg.target_refusal))
    if should_refine:
        _log("refining to target refusal ...")
        rr, rk = _refine_refusal(bundle, controller, cfg, eval_harmful, eval_harmless)
        _log(f"refine result: refusal={rr:.3f} kl={rk:.3f}")
    elif cfg.refine_refusal and skip_guard:
        _log("refine: skipped (guard was skipped, refusal already clean)")

    with controller.active():
        edited_refusal = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
    kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size)
    _log(f"edited refusal rate: {edited_refusal:.3f} | harmless KL: {kl:.3f} nats")

    backoff = 0
    if kl > cfg.max_kl:
        base_alpha = dict(controller.alpha)

        def _apply_scale(s: float):
            for mid, a in base_alpha.items():
                controller.alpha[mid] = a * s
        lo, hi = 0.0, 1.0
        for backoff in range(1, 7):
            mid = 0.5 * (lo + hi)
            _apply_scale(mid)
            kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size)
            if kl > cfg.max_kl:
                hi = mid
            else:
                lo = mid
            _log(f"  backoff {backoff}: scale={mid:.3f} KL={kl:.3f}")
        _apply_scale(lo)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size)
        with controller.active():
            edited_refusal = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        _log(f"  backoff final: scale={lo:.3f} KL={kl:.3f} refusal={edited_refusal:.3f}")

    # prune redundant layers
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
            skipper.set(drop_layers)   # eval the pruned model
        else:
            _log("pruning: no layer is free within kl budget")

    # honest report: disjoint test set, never used for tuning (reflects pruned model)
    with controller.active():
        edited_refusal = refusal_rate(bundle, test_harmful, cfg.max_new_tokens, cfg.batch_size)
    if skipper is not None:
        skipper.remove()
    kl = kl_harmless(bundle, controller, test_harmless, cfg.batch_size)
    _log(f"TEST refusal: {edited_refusal:.3f} | TEST harmless KL: {kl:.3f} nats"
         + (f" | {len(drop_layers)} layers pruned" if drop_layers else ""))

    report = {
        "model": cfg.model,
        "num_layers": bundle.num_layers,
        "hidden_size": bundle.hidden_size,
        "direction_layer": L_dir,
        "refusal_subspace_rank": int(controller.R.shape[1]),
        "initial_separation": round(initial_sep, 4),
        "baseline_refusal_rate": round(base_refusal, 4),
        "edited_refusal_rate": round(edited_refusal, 4),
        "harmless_kl_nats": round(kl, 4),
        "kl_backoff_steps": backoff,
        "guard_history": guard_hist,
        "layer_alphas": [round(controller.get_layer_alpha(L), 3) for L in range(bundle.num_layers)],
        "pruned_layers": drop_layers,
        "layers_after_prune": bundle.num_layers - len(drop_layers),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    report.update(report_extra)
    with open(os.path.join(cfg.output_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        f.write(cfg.to_json())

    if cfg.bake:
        export = controller.export()
        _log("freeing 4-bit model and baking edited weights ...")
        controller.remove()
        del bundle.model, bundle
        gc.collect()
        torch.cuda.empty_cache()
        out = bake(cfg, export, tokenizer=tok, drop_layers=drop_layers)
        _log(f"baked edited model -> {out}")
        report["baked_to"] = out

    _log(f"done in {report['elapsed_sec']}s | refusal {base_refusal:.2f} -> {edited_refusal:.2f}, KL {kl:.3f}")
    return report
