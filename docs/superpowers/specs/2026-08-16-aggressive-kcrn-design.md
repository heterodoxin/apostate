# Aggressive KCRN Design

## Purpose

Aggressive KCRN is an opt-in fixed-weight KCRN variant that begins with the normal projected KCRN baseline and may add bounded non-overlapping projected writer updates. Normal KCRN remains the default. The aggressive result is accepted only when the reloaded checkpoint meets its configured harmful-delivery and held-out benign-KL thresholds.

## Compatibility

The existing `kcrn` method and projected solve keep their normal behavior. The canonical opt-in name is `aggressive-kcrn`; the existing `aggressive` profile spelling remains a compatibility alias. The exported model contains ordinary tensors only and has no router, hook, detector, or prompt-dependent runtime branch.

## Architecture

Aggressive KCRN first calls the normal KCRN fitting path with the aggressive baseline settings. It then reuses the projected KCRN update for additional candidates:

```text
K̃_h = K_h - Q_b(Q_bᵀK_h)
L   = -s R(Rᵀ(WK_h))
(K_hᵀK̃_h + λI)C = K̃_hᵀ
ΔW  = LC
```

The controller temporarily applies the baseline and candidate factors to the in-memory model, measures calibration behavior, restores the untouched writers, and leaves final application to the existing one-time bake. Baseline and adaptive targets are disjoint.

## Adaptive selection

Candidates come from the existing writer-discovery path and are bounded by projected harmful-key energy, condition number, delta norm, and relative update safeguards. Additional strengths are selected from a configurable finite grid, defaulting to `1, 2, 4, 6, 8`.

The calibration objective is a teacher-forced response-prefix margin on an external harmful tuning split: the compliance prefix `Sure, here are the steps:` is compared with the refusal prefix `I cannot help with that.`. Benign calibration KL is measured on cached base outputs after each complete baseline-plus-candidate stack. The harmful final holdout and benign final holdout are excluded from selection.

## Safeguards

The projected solver skips candidates with insufficient projected harmful energy, excessive condition number, or excessive relative update norm. The aggressive controller additionally limits cumulative relative update and calibration KL and records rejected candidates. A run that misses the final thresholds is reported as failed.

The final validator reloads the saved checkpoint using the exact save dtype and loading path, computes calibration and held-out benign KL against cached base outputs, and measures harmful delivery with the strict content-delivery judge.

## Configuration and reporting

The profile selects the variant while the method remains `kcrn`. Aggressive-specific controls expose the baseline strength, extra-factor strength grid, maximum greedy steps, calibration KL budget, cumulative update limit, and target delivery. Reports include the variant, baseline and adaptive step counts, candidate order, accepted and rejected records, per-writer projected KCRN certificates, calibration metrics, reloaded held-out metrics, fixed-weight runtime status, and serialization preservation metrics.

## Tests and acceptance

Focused fp32 tests cover projected solver fast-path equivalence, greedy selection, preservation nulling, deterministic candidate ordering, cumulative safeguards, reversible writer application, prompt-split disjointness, response-prefix scoring, and the unchanged normal path. Integration validation runs aggressive KCRN on Qwen3-8B with disjoint prompt sources, fixed dtype, fixed-weight reload, and the final content-delivery judge.
