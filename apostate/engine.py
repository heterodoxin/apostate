"""ablation pipeline"""

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
from .activations import collect_activations
from .directions import refusal_subspace, preservation_subspace, gram_schmidt_remove, separation
from .projectors import ProjectionController
from .causal import causal_layer_scores
from .guard import run_guard
from .evaluate import refusal_logit_margin, refusal_rate, kl_harmless
from .optimize import optimize_profile
from .search import _has_optuna
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


def _cached_collect(bundle, instructions, batch_size: int, cfg: ApostateConfig, name: str):
    if not cfg.cache_activations:
        return collect_activations(bundle, instructions, batch_size)
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
    acts = collect_activations(bundle, instructions, batch_size)
    try:
        torch.save({"meta": meta, "activations": acts}, path)
        _log(f"activation cache saved: {name}")
    except Exception as e:
        _log(f"activation cache save failed for {name}: {e}")
    return acts


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


def _refine_refusal(bundle, controller, cfg, eval_harmful, eval_harmless):
    """refine refusal"""
    base = dict(controller.alpha)
    es = eval_harmful[: max(48, cfg.opt_eval_n)]
    el = eval_harmless[: cfg.opt_eval_n]
    with controller.active():
        ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
    kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
    if ref <= cfg.target_refusal:
        if not cfg.refine_deescalate:
            return ref, kl
        # shrink alpha
        best = (ref, kl, dict(base))
        for s in (0.9, 0.8, 0.7):
            for mid, a in base.items():
                controller.alpha[mid] = a * s
            with controller.active():
                new_ref = refusal_rate(bundle, es, cfg.max_new_tokens, cfg.batch_size)
            if new_ref > cfg.target_refusal:
                break
            new_kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
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
        new_kl = kl_harmless(bundle, controller, el, cfg.batch_size, positions=cfg.kl_positions)
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


def _minimize_kl_scale(bundle, controller, cfg, eval_harmful, eval_harmless):
    base_alpha = dict(controller.alpha)
    best_alpha = dict(base_alpha)
    best_ref = None
    best_kl = None
    lo, hi = 0.0, 1.0
    target = cfg.target_refusal + cfg.refine_refusal_slack

    def _apply(s: float):
        for mid, a in base_alpha.items():
            controller.alpha[mid] = a * s

    for _ in range(cfg.refine_kl_steps):
        mid = 0.5 * (lo + hi)
        _apply(mid)
        with controller.active():
            ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
        if ref <= target:
            hi = mid
            best_alpha = dict(controller.alpha)
            best_ref, best_kl = ref, kl
        else:
            lo = mid

    controller.alpha = best_alpha
    if best_ref is None:
        _apply(1.0)
        with controller.active():
            best_ref = refusal_rate(bundle, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
        best_kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
    return best_ref, best_kl


def _alpha_get(controller, item: int) -> float:
    if item < 0:
        return controller.get_embed_alpha()
    return controller.get_layer_alpha(item)


def _alpha_set(controller, item: int, value: float):
    if item < 0:
        controller.set_embed_alpha(value)
    else:
        controller.set_layer_alpha(item, value)


def _alpha_label(item: int) -> str:
    return "embed" if item < 0 else f"L{item}"


def _minimize_kl_layers(bundle, controller, cfg, eval_harmful, eval_harmless):
    target = cfg.target_refusal + cfg.refine_refusal_slack
    hset = eval_harmful[: max(24, cfg.opt_eval_n)]
    lset = eval_harmless[: max(48, cfg.opt_eval_n)]
    best_alpha = dict(controller.alpha)
    with controller.active():
        best_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
    best_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    if best_ref > target or best_kl <= cfg.kl_target:
        return best_ref, best_kl, 0

    items = [-1] + list(range(controller.num_layers))
    kept = 0
    scales = (0.75, 0.50, 0.25, 0.0)
    for step in range(cfg.refine_kl_layer_steps):
        controller.alpha = dict(best_alpha)
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
            controller.alpha = dict(best_alpha)
            start = _alpha_get(controller, item)
            for scale in scales:
                controller.alpha = dict(best_alpha)
                _alpha_set(controller, item, start * scale)
                trial_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
                if trial_kl >= best_kl - 1e-4:
                    continue
                with controller.active():
                    trial_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
                if trial_ref <= target and (accepted is None or trial_kl < accepted[0]):
                    accepted = (trial_kl, trial_ref, item, scale, dict(controller.alpha))
            if accepted is not None and accepted[0] <= cfg.kl_target:
                break

        if accepted is None:
            break
        best_kl, best_ref, item, scale, best_alpha = accepted
        kept += 1
        _log(f"  kl trim {step + 1}: {_alpha_label(item)} x{scale:.2f} refusal={best_ref:.3f} kl={best_kl:.3f}")
        if best_kl <= cfg.kl_target:
            break

    controller.alpha = best_alpha
    return best_ref, best_kl, kept


def _repair_loss(ref: float, kl: float, cfg) -> float:
    ref_over = max(0.0, ref - cfg.target_refusal)
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


def _repair_scales(best_ref: float, best_kl: float, cfg):
    target = cfg.target_refusal + cfg.refine_refusal_slack
    if best_kl > cfg.kl_target and best_ref <= target:
        return (0.0, 0.25, 0.50, 0.75, 0.90)
    if best_ref > cfg.target_refusal and best_kl <= cfg.kl_target:
        return (1.10, 1.25, 1.45)
    return (0.0, 0.50, 0.75, 0.90, 1.10, 1.25)


def _repair_priority(start: float, scale: float, best_ref: float, best_kl: float, cfg) -> float:
    shrink = max(0.0, 1.0 - scale)
    grow = max(0.0, scale - 1.0)
    kl_need = max(0.0, best_kl - cfg.kl_target) + 2.0 * max(0.0, best_kl - cfg.max_kl)
    ref_need = max(0.0, best_ref - cfg.target_refusal)
    return abs(start) * (1.0 + shrink + grow + 8.0 * kl_need * shrink + 8.0 * ref_need * grow)


def _repair_alphas(bundle, controller, cfg, eval_harmful, eval_harmless):
    hset = eval_harmful[: max(24, cfg.repair_eval_n)]
    lset = eval_harmless[: max(48, cfg.repair_kl_n)]
    hprobe = hset[: max(4, min(len(hset), cfg.repair_probe_ref_n))]
    lprobe = lset[: max(8, min(len(lset), cfg.repair_probe_kl_n))]
    probe_positions = max(4, min(cfg.kl_positions, cfg.repair_probe_positions))
    with controller.active():
        best_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
    best_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
    best_alpha = dict(controller.alpha)
    best_score = _repair_loss(best_ref, best_kl, cfg)
    steps = 0

    for step in range(cfg.repair_steps):
        controller.alpha = dict(best_alpha)
        active = [
            item for item in ([-1] + list(range(controller.num_layers)))
            if abs(_alpha_get(controller, item)) > 1e-6
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
        candidates.sort(reverse=True)
        candidates = candidates[: max(1, cfg.repair_probe_candidates)]
        _log(
            f"  repair {step + 1}/{cfg.repair_steps}: "
            f"probe={len(candidates)} exact={cfg.repair_rerank_k} "
            f"refusal={best_ref:.3f} kl={best_kl:.3f}"
        )

        cheap = []
        for _priority, item, scale, value in candidates:
            controller.alpha = dict(best_alpha)
            _alpha_set(controller, item, value)
            trial_kl = kl_harmless(bundle, controller, lprobe, cfg.batch_size, positions=probe_positions)
            with controller.active():
                proxy_ref = _margin_refusal(bundle, hprobe, cfg.batch_size)
            proxy_score = _repair_loss(proxy_ref, trial_kl, cfg)
            cheap.append((proxy_score, trial_kl, item, scale, dict(controller.alpha)))

        accepted = None
        cheap.sort(key=lambda x: (x[0], x[1]))
        for _proxy_score, _probe_kl, item, scale, alpha in cheap[: cfg.repair_rerank_k]:
            controller.alpha = dict(alpha)
            with controller.active():
                trial_ref = refusal_rate(bundle, hset, cfg.max_new_tokens, cfg.batch_size)
            trial_kl = kl_harmless(bundle, controller, lset, cfg.batch_size, positions=cfg.kl_positions)
            trial_score = _repair_loss(trial_ref, trial_kl, cfg)
            if trial_score < best_score - 1e-4:
                if accepted is None or trial_score < accepted[0]:
                    accepted = (trial_score, trial_ref, trial_kl, item, scale, dict(controller.alpha))

        if accepted is None:
            _log(f"  repair {step + 1}: no better exact candidate")
            break
        best_score, best_ref, best_kl, item, scale, best_alpha = accepted
        steps += 1
        _log(f"  repair {step + 1}: {_alpha_label(item)} x{scale:.2f} refusal={best_ref:.3f} kl={best_kl:.3f}")
        if best_ref <= cfg.target_refusal and best_kl <= cfg.kl_target:
            break

    controller.alpha = best_alpha
    return best_ref, best_kl, steps


def _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless):
    base_alpha = dict(controller.alpha)

    def apply_scale(s: float):
        for mid, a in base_alpha.items():
            controller.alpha[mid] = a * s

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


def run(cfg: ApostateConfig, command: Optional[str] = None) -> dict:
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
    # split eval
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
    ah = _cached_collect(bundle, fit_harmful, cfg.batch_size, cfg, "fit_harmful")
    al = _cached_collect(bundle, harmless, cfg.batch_size, cfg, "fit_harmless")

    preserve_acts = None
    preserve_source = "none"
    if cfg.preserve_rank > 0 and cfg.preserve_path:
        preserve = resolve_prompts(cfg.preserve_path, cfg.n_harmless, cfg.seed)
        preserve_acts = _cached_collect(bundle, preserve, cfg.batch_size, cfg, "preserve")
        preserve_source = "custom"
    elif cfg.preserve_rank > 0:
        preserve_acts = al
        preserve_source = "harmless"
    preserve_lookup = _preservation_lookup(preserve_acts, cfg.preserve_rank)
    preserve_basis = preserve_lookup(L_dir)
    if preserve_basis is not None:
        _log(f"preservation subspace rank={preserve_basis.shape[1]} source={preserve_source}")

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
            causal_shape, cfg, preserve_basis, preserve_lookup,
        )
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_params.items()}
        _log(f"best trial: refusal_proxy={best_attrs.get('refusal_proxy', best_attrs.get('refusal'))} "
             f"kl={best_attrs.get('kl')} | {shown}")
        L_dir = max(0, min(bundle.num_layers - 1, int(bundle.num_layers * best_params["direction_layer_frac"])))
        preserve_basis = preserve_lookup(L_dir)
        report_extra.update({"best_params": best_params, "best_trial": best_attrs, "n_trials": cfg.n_trials})
    else:
        R, svals = refusal_subspace(
            ah[L_dir], al[L_dir],
            rank=cfg.refusal_rank, variance_threshold=cfg.variance_threshold,
            max_rank=cfg.max_rank, seed=cfg.seed,
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
        gcap = max(256, cfg.opt_eval_n)   # guard subset
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
    kl = kl_harmless(bundle, controller, eval_harmless, cfg.batch_size, positions=cfg.kl_positions)
    _log(f"edited refusal rate: {edited_refusal:.3f} | harmless KL: {kl:.3f} nats")

    kl_layer_steps = 0
    repair_steps = 0
    if cfg.refine_deescalate and edited_refusal <= cfg.target_refusal + cfg.refine_refusal_slack:
        _log("minimizing kl scale ...")
        edited_refusal, kl = _minimize_kl_scale(bundle, controller, cfg, eval_harmful, eval_harmless)
        _log(f"kl scale result: refusal={edited_refusal:.3f} kl={kl:.3f}")
        if kl > cfg.kl_target:
            _log("trimming kl layers ...")
            edited_refusal, kl, kl_layer_steps = _minimize_kl_layers(
                bundle, controller, cfg, eval_harmful, eval_harmless,
            )
            _log(f"kl layer result: refusal={edited_refusal:.3f} kl={kl:.3f} steps={kl_layer_steps}")

    if cfg.refine_deescalate and (edited_refusal > cfg.target_refusal or kl > cfg.kl_target):
        _log("repairing refusal/kl tradeoff ...")
        edited_refusal, kl, repair_steps = _repair_alphas(bundle, controller, cfg, eval_harmful, eval_harmless)
        _log(f"repair result: refusal={edited_refusal:.3f} kl={kl:.3f} steps={repair_steps}")

    backoff = 0
    if kl > cfg.max_kl:
        edited_refusal, kl, backoff = _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless)
        if cfg.refine_deescalate and edited_refusal > cfg.target_refusal:
            _log("repairing after backoff ...")
            edited_refusal, kl, extra_steps = _repair_alphas(bundle, controller, cfg, eval_harmful, eval_harmless)
            repair_steps += extra_steps
            _log(f"post-backoff repair: refusal={edited_refusal:.3f} kl={kl:.3f} steps={extra_steps}")
            if kl > cfg.max_kl:
                _log("final kl backoff ...")
                edited_refusal, kl, extra_backoff = _backoff_to_kl(bundle, controller, cfg, eval_harmful, eval_harmless)
                backoff += extra_backoff

    # prune layers
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
            skipper.set(drop_layers)   # eval prune
        else:
            _log("pruning: no layer is free within kl budget")

    # test report
    with controller.active():
        edited_refusal = refusal_rate(bundle, test_harmful, cfg.max_new_tokens, cfg.batch_size)
    if skipper is not None:
        skipper.remove()
    kl = kl_harmless(bundle, controller, test_harmless, cfg.batch_size, positions=cfg.kl_positions)
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
        "kl_layer_trim_steps": kl_layer_steps,
        "repair_steps": repair_steps,
        "guard_history": guard_hist,
        "layer_alphas": [round(controller.get_layer_alpha(L), 3) for L in range(bundle.num_layers)],
        "embed_alpha": round(controller.get_embed_alpha(), 3),
        "preserve_rank": cfg.preserve_rank,
        "preserve_source": preserve_source,
        "pruned_layers": drop_layers,
        "layers_after_prune": bundle.num_layers - len(drop_layers),
        "elapsed_sec": round(time.time() - t0, 1),
        "profile": cfg.profile,
        "target_refusal": cfg.target_refusal,
        "max_kl": cfg.max_kl,
        "kl_target": cfg.kl_target,
        "kl_positions": cfg.kl_positions,
        "opt_capability": cfg.opt_capability,
        "opt_capability_weight": cfg.opt_capability_weight,
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
        report["baked_to"] = out
        _persist_reports(cfg, report, command)

    _log(f"done in {report['elapsed_sec']}s | refusal {base_refusal:.2f} -> {edited_refusal:.2f}, KL {kl:.3f}")
    return report
