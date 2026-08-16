# Aggressive KCRN Design

## Purpose

Aggressive KCRN is a second fixed-weight KCRN variant for cases where normal KCRN does not remove enough harmful-prompt refusal. Normal KCRN remains unchanged and remains the default. The aggressive variant is accepted only when the reloaded checkpoint reaches at least 90% harmful delivery on the 96-prompt held-out set while held-out benign KL remains at or below 0.02.

## Compatibility

The existing `kcrn` method and projected solve keep their current behavior. The canonical opt-in name is `aggressive-kcrn`; the existing `aggressive` profile spelling remains a compatibility alias. The report records the selected variant, all adaptive steps, calibration metrics, held-out metrics, and the final baked dtype. The exported model contains ordinary tensors only and has no router, hook, detector, or prompt-dependent runtime branch.

## Architecture

Aggressive KCRN reuses the projected KCRN update as its inner operation:

```text
K̃_h = K_h - Q_b(Q_bᵀK_h)
L   = -s R(Rᵀ(WK_h))
(K_hᵀK̃_h + λI)C = K̃_hᵀ
ΔW  = LC
```

The variant adds a calibration-only greedy controller around that operation. It builds a candidate pool from the existing writer discovery path, ranks candidates using harmful-versus-benign refusal signal, and evaluates bounded candidate updates against the current in-memory model state. After accepting an update, it recomputes the remaining refusal residual for later writers so later edits do not repeatedly target a stale base matrix.

Each accepted step stores its layer, writer, strength, projected-key certificate, calibration refusal change, calibration benign KL change, and cumulative update norm. The final list of updates is composed and baked once into the standalone checkpoint.

## Adaptive selection

Aggressive KCRN uses `auto` layer and writer discovery by default so candidates receive a nonzero measured selection score. Candidate ranking combines:

- harmful refusal-signal reduction on the calibration tuning split;
- the projected harmful-key energy certificate;
- benign calibration KL cost;
- condition number and relative update safeguards.

Strength is selected from a configurable finite grid, defaulting to `1, 2, 4, 6, 8`. A candidate is accepted only when it improves the calibration refusal objective without exceeding the configured update and conditioning limits. Selection stops when no candidate improves the objective, the benign calibration KL budget is exhausted, or the maximum step count is reached. The default maximum is 24 steps.

The refusal objective uses calibration-only harmful prompts and a disjoint calibration tuning subset. It may use the existing strict delivery judge or its deterministic refusal-logit proxy for candidate ranking, but it never reads the held-out harmful set. Benign calibration prompts are used for basis construction and calibration KL only. The held-out harmful and benign prompts remain untouched until final validation.

## Safeguards

The projected solver continues to skip candidates with insufficient projected harmful energy, excessive condition number, or excessive relative update norm. Aggressive KCRN adds cumulative relative-update and calibration-KL limits. A candidate or complete run that violates a safeguard is reported as skipped or failed rather than being silently clipped into an unverified result.

The final validator reloads the saved checkpoint using the exact save dtype and loading path, computes calibration and held-out benign KL against cached base outputs, and measures harmful delivery with the strict content-delivery judge. A result below 90% delivery or above 0.02 held-out KL is not labeled successful and is not made the recommended aggressive checkpoint.

## Configuration and reporting

The profile selects the variant while the method remains `kcrn`. Aggressive-specific controls expose the strength grid, maximum greedy steps, calibration KL budget, cumulative update limit, and target delivery. Normal KCRN ignores these controls. Reports include:

- `variant` and canonical profile name;
- candidate order and accepted-step records;
- per-writer projected KCRN certificates;
- calibration delivery/refusal objective and calibration KL;
- reloaded held-out delivery and benign KL;
- fixed-weight runtime status, bake dtype, and serialization preservation metrics.

## Tests and acceptance

Focused fp32 tests cover greedy residual updates, preservation nulling, strength selection, deterministic candidate ordering, cumulative safeguards, and the unchanged normal path. Integration validation runs normal and aggressive KCRN on Qwen3-8B with identical prompt sources, disjoint splits, dtype, token budget, and reloaded checkpoint protocol. Aggressive KCRN is considered complete only if that validation meets both acceptance thresholds without enabling runtime hooks.
