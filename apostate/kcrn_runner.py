"""Production runner for fixed-weight KCRN checkpoint creation."""

from __future__ import annotations

import gc
import json
import math
import random
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

import torch

from .bake import bake
from .aggressive_kcrn import (
    AggressiveEvaluation,
    AggressiveTrial,
    apply_kcrn_factor,
    greedy_select,
    merge_trial_stacks,
    parse_strength_grid,
    restore_writer,
    snapshot_writer,
    shortlist_trials,
)
from .config import ApostateConfig
from .data import format_chat, resolve_prompts
from .directions import refusal_subspace
from .evaluate import (
    _COMPLY_STARTS,
    _REFUSAL_STARTS,
    _first_token_ids,
    _logits_kwarg,
    generate,
    judge_strict_refusal,
    response_prefix_margin,
)
from .activations import collect_activations, collect_response_activations
from .kcrn import (
    KCRNDegeneracyError,
    collect_writer_inputs,
    key_basis,
    key_conditional_nulling,
    key_conditional_nulling_projected,
    select_writer_targets,
    writer_matrix,
)
from .model import load_model


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def guard_model_and_output(model: str, output_dir: str, force: bool = False) -> None:
    """Reject output paths that would overwrite the input or existing data."""

    model_path = Path(model).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if model_path == output_path:
        raise ValueError("output directory must be different from the base model directory")
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory {output_path}; pass --kcrn-force"
        )


def split_prompt_sets(
    prompts: Iterable[str],
    calibration_n: int,
    holdout_n: int,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Shuffle unique prompts once and return disjoint calibration and holdout sets."""

    if calibration_n < 1 or holdout_n < 1:
        raise ValueError("calibration_n and holdout_n must be positive")
    unique = list(dict.fromkeys(prompt for prompt in prompts if isinstance(prompt, str) and prompt.strip()))
    needed = int(calibration_n) + int(holdout_n)
    if len(unique) < needed:
        raise ValueError(f"need {needed} unique prompts, got {len(unique)}")
    random.Random(seed).shuffle(unique)
    return unique[:calibration_n], unique[calibration_n:needed]


def _resolve_disjoint_prompt_sets(
    calibration_path: str,
    holdout_path: str,
    calibration_n: int,
    holdout_n: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Resolve calibration and external holdout prompts without text overlap."""

    if calibration_path == holdout_path:
        pool = resolve_prompts(calibration_path, calibration_n + holdout_n, seed)
        return split_prompt_sets(pool, calibration_n, holdout_n, seed)
    calibration = resolve_prompts(calibration_path, calibration_n, seed)
    holdout_pool = resolve_prompts(holdout_path, holdout_n + len(calibration), seed + 1)
    calibration_set = set(calibration)
    holdout = [prompt for prompt in holdout_pool if prompt not in calibration_set][:holdout_n]
    if len(calibration) < calibration_n or len(holdout) < holdout_n:
        raise ValueError(
            "calibration and holdout prompt sources do not contain enough unique, disjoint prompts"
        )
    return calibration, holdout


def _resolve_aggressive_prompt_sets(
    calibration_path: str,
    holdout_path: str,
    calibration_n: int,
    tuning_n: int,
    holdout_n: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Resolve fit, external tuning, and final harmful sets without text overlap."""

    if calibration_n < 1 or tuning_n < 1 or holdout_n < 1:
        raise ValueError("aggressive harmful calibration, tuning, and holdout counts must be positive")
    calibration = resolve_prompts(calibration_path, calibration_n, seed)
    needed_external = int(tuning_n) + int(holdout_n)
    external_pool = resolve_prompts(
        holdout_path,
        needed_external + len(calibration),
        seed + 1,
    )
    calibration_set = set(calibration)
    external = [prompt for prompt in external_pool if prompt not in calibration_set]
    if len(calibration) < calibration_n or len(external) < needed_external:
        raise ValueError(
            "aggressive harmful sources do not contain enough unique disjoint fit, tuning, and holdout prompts"
        )
    tuning = external[: int(tuning_n)]
    holdout = external[int(tuning_n) : needed_external]
    return calibration, tuning, holdout


def parse_index_spec(spec: str | None, size: int, name: str) -> list[int]:
    """Parse an index list containing integers, ranges, or the word all."""

    text = (spec or "all").strip().lower()
    if text in ("", "all"):
        return list(range(size))
    if text == "auto":
        return []
    values: set[int] = set()
    try:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                first, last = part.split("-", 1)
                lo, hi = int(first), int(last)
                if hi < lo:
                    raise ValueError
                values.update(range(lo, hi + 1))
            else:
                values.add(int(part))
    except ValueError as exc:
        raise ValueError(f"invalid {name} selection: {spec!r}") from exc
    invalid = sorted(index for index in values if index < 0 or index >= size)
    if invalid:
        raise ValueError(f"{name} selection out of range for size {size}: {invalid}")
    return sorted(values)


def _pointwise_kl_from_logits(
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    vocab_chunk_size: int = 8192,
) -> torch.Tensor:
    if base_logits.shape != edited_logits.shape or base_logits.ndim != 3:
        raise ValueError("logits must have identical [batch, positions, vocabulary] shapes")
    if int(vocab_chunk_size) < 1:
        raise ValueError("vocab_chunk_size must be positive")
    base_norm = torch.logsumexp(base_logits.float(), dim=-1)
    edited_norm = torch.logsumexp(edited_logits.float(), dim=-1)
    pointwise = torch.zeros(base_logits.shape[:2], dtype=torch.float32, device=base_logits.device)
    for start in range(0, base_logits.shape[-1], int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), base_logits.shape[-1])
        base_lp = base_logits[..., start:stop].float() - base_norm.unsqueeze(-1)
        edited_lp = edited_logits[..., start:stop].float() - edited_norm.unsqueeze(-1)
        pointwise += (base_lp.exp() * (base_lp - edited_lp)).sum(dim=-1)
    return pointwise


def masked_kl_from_logits(
    base_logits: torch.Tensor,
    edited_logits: torch.Tensor,
    mask: torch.Tensor,
    vocab_chunk_size: int = 8192,
) -> float:
    """Return float32 KL(base||edited) over the true token positions."""

    if base_logits.shape != edited_logits.shape:
        raise ValueError("base and edited logits must have identical shapes")
    if base_logits.ndim != 3 or mask.shape != base_logits.shape[:2]:
        raise ValueError("logits must be [batch, positions, vocabulary] and mask must match positions")
    pointwise = _pointwise_kl_from_logits(base_logits, edited_logits, vocab_chunk_size)
    weights = mask.to(dtype=pointwise.dtype, device=pointwise.device)
    total = (pointwise * weights).sum()
    count = weights.sum().clamp_min(1.0)
    return max(0.0, float((total / count).item()))


def target_priority(
    selection_score: float,
    delta_norm: float,
    condition: float,
    max_delta_norm: float,
    max_condition: float,
) -> Optional[float]:
    """Score a target by refusal signal per bounded factor norm."""

    if not all(math.isfinite(float(value)) for value in (selection_score, delta_norm, condition)):
        return None
    if delta_norm > float(max_delta_norm) or condition > float(max_condition):
        return None
    return float(selection_score) / (1.0 + max(0.0, float(delta_norm)))


def _writer_label(layer, module) -> str:
    names = [name for name, child in layer.named_modules() if child is module]
    return names[0] if names else type(module).__name__


def _compatible_targets(bundle, layers: Iterable[int], writer_spec: str) -> list[dict]:
    spec = (writer_spec or "all").strip().lower()
    requested = None if spec in ("", "all", "mlp") else set(parse_index_spec(spec, 128, "writer"))
    targets = []
    for layer_index in layers:
        layer = bundle.layers()[layer_index]
        for writer_index, writer in enumerate(bundle.layer_writers(layer)):
            label = _writer_label(layer, writer)
            if requested is not None and writer_index not in requested:
                continue
            if spec == "mlp" and not any(
                token in label.lower()
                for token in ("down_proj", "c_proj", "w2", "fc_out", "dense_4h_to_h", "wo")
            ):
                continue
            try:
                matrix, _transpose = writer_matrix(writer)
            except (TypeError, ValueError, RuntimeError):
                continue
            targets.append({
                "layer": int(layer_index),
                "writer_index": int(writer_index),
                "writer": label,
                "selection_score": 0.0,
                "matrix_shape": list(matrix.shape),
            })
    return targets


def _write_model_card(output: Path, report: dict) -> None:
    """Write a checkpoint README describing the KCRN edit and the numbers this run measured."""

    delivery = report.get("harmful_delivery")
    kl = report.get("heldout_benign_kl")
    rows = [
        ("base model", f"`{report.get('model')}`"),
        ("method", "Apostate KCRN (fixed-weight key-conditional refusal nulling)"),
        ("edited writers", str(len(report.get("targets") or []))),
        ("strength", str(report.get("kcrn_strength"))),
        ("checkpoint dtype", str(report.get("save_dtype"))),
    ]
    if delivery is not None:
        rows.append(("held-out harmful delivery", f"{delivery * 100:.1f}% of {report.get('eval_n')} prompts"))
    if kl is not None:
        rows.append(("held-out benign full-position KL", f"{kl:.6f}"))
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    text = f"""# {output.name}

Uncensored build of `{report.get('model')}`, produced with
[Apostate](https://github.com/heterodoxin/apostate) using the fixed-weight KCRN path.

KCRN removes the refusal direction from the residual writers that carry it, conditioned on the
harmful key subspace where the refusal decision is made, while pinning a benign key basis to
zero change. The result is a plain checkpoint: no runtime hook, adapter, finetune, or router.

| field | value |
|---|---|
{table}

Delivery is scored on held-out prompts the edit was never fitted on, with a strict judge. KL is
raw float32 `KL(base || edited)` over every non-padding prompt position, measured after reloading
this checkpoint. Both numbers are protocol-specific and only comparable against runs using the
same splits, token budget, judge, dtype, and loading path; `kcrn_report.json` records that protocol.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output.name}", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("{output.name}")
```

## Warning

This model is uncensored and will answer harmful and dangerous requests. You are responsible
for how you use it.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _load_saved_edits(path: str) -> tuple[list[dict], dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        edits = saved.get("edits")
        metadata = saved.get("metadata", {})
    else:
        edits, metadata = saved, {}
    if not isinstance(edits, list) or not edits:
        raise ValueError(f"KCRN edits file contains no edits: {path}")
    for index, edit in enumerate(edits):
        if edit.get("kind") != "kcrn":
            raise ValueError(f"edit {index} is not a KCRN edit")
        if not isinstance(edit.get("left"), torch.Tensor) or not isinstance(edit.get("right"), torch.Tensor):
            raise ValueError(f"edit {index} lacks tensor factors")
        if "layer" not in edit or "writer_index" not in edit:
            raise ValueError(f"edit {index} lacks layer and writer indices")
    return edits, metadata if isinstance(metadata, dict) else {}


def _refusal_basis(cfg: ApostateConfig, harmful: torch.Tensor, benign: torch.Tensor) -> torch.Tensor:
    """Build the output refusal basis used by every writer in one layer."""

    rank = max(1, int(cfg.kcrn_refusal_rank))
    harmful = harmful.float()
    benign = benign.float()
    if harmful.ndim != 2 or benign.ndim != 2 or harmful.shape[1] != benign.shape[1]:
        raise ValueError("KCRN refusal activations must be [samples, hidden] with matching widths")
    if rank == 1:
        return (harmful.mean(0) - benign.mean(0)).unsqueeze(1).contiguous()
    basis, _weights = refusal_subspace(
        harmful,
        benign,
        rank=rank,
        max_rank=rank,
        seed=int(cfg.seed),
        multi=bool(cfg.kcrn_refusal_multi),
        clusters=(int(cfg.kcrn_refusal_clusters) or None),
        min_norm_frac=float(cfg.kcrn_refusal_min_norm_frac),
        min_separation=float(cfg.kcrn_refusal_min_separation),
        min_coverage=float(cfg.kcrn_refusal_min_coverage),
    )
    if basis.ndim != 2 or basis.shape[1] == 0:
        raise ValueError("KCRN refusal basis construction returned no directions")
    return basis.contiguous()


def _head_refusal_basis(cfg: ApostateConfig, bundle) -> torch.Tensor:
    """Build an output basis from refusal-token and comply-token LM-head rows."""

    head = bundle.lm_head()
    if head is None or not isinstance(getattr(head, "weight", None), torch.Tensor):
        raise ValueError("KCRN head refusal source requires an accessible LM head")
    refusal_ids = _first_token_ids(bundle.tokenizer, _REFUSAL_STARTS)
    comply_ids = _first_token_ids(bundle.tokenizer, _COMPLY_STARTS)
    if not refusal_ids or not comply_ids:
        raise ValueError("KCRN head refusal source found no refusal or comply token ids")
    weights = head.weight.detach().float()
    refusal_rows = weights[refusal_ids]
    comply_rows = weights[comply_ids]
    mean_difference = refusal_rows.mean(0) - comply_rows.mean(0)
    rank = max(1, int(cfg.kcrn_refusal_rank))
    if rank == 1:
        return mean_difference.unsqueeze(1).contiguous()
    contrasts = torch.cat(
        (
            refusal_rows - comply_rows.mean(0, keepdim=True),
            mean_difference.unsqueeze(0),
        ),
        dim=0,
    )
    basis = torch.linalg.svd(contrasts.T, full_matrices=False).U[:, :rank]
    if basis.shape[1] == 0:
        raise ValueError("KCRN head refusal source returned no directions")
    return basis.contiguous()


def _fit_edits(
    cfg: ApostateConfig,
    bundle,
    fit_harmful: list[str],
    fit_benign: list[str],
) -> tuple[list[dict], list[dict], list[dict], dict[tuple[int, int], torch.Tensor]]:
    harmful_activations = collect_activations(bundle, fit_harmful, cfg.batch_size)
    benign_activations = collect_activations(bundle, fit_benign, cfg.batch_size)
    layer_spec = (cfg.kcrn_layers or "auto").strip().lower()
    selected_layers = None if layer_spec == "auto" else parse_index_spec(
        layer_spec, bundle.num_layers, "layer"
    )
    writer_spec = (cfg.kcrn_writers or "auto").strip().lower()
    writer_indices = None
    mlp_only = False
    if writer_spec not in ("", "auto", "all"):
        mlp_only = writer_spec == "mlp"
        if not mlp_only:
            writer_indices = set(parse_index_spec(writer_spec, 128, "writer"))
    if selected_layers is None or writer_spec == "auto":
        pilot_harmful = collect_writer_inputs(
            bundle,
            fit_harmful[: min(len(fit_harmful), max(1, int(cfg.kcrn_pilot_n)))],
            layers=selected_layers,
            batch_size=cfg.batch_size,
            all_positions=False,
        )
        pilot_benign = collect_writer_inputs(
            bundle,
            fit_benign[: min(len(fit_benign), max(1, int(cfg.kcrn_pilot_n)))],
            layers=selected_layers,
            batch_size=cfg.batch_size,
            all_positions=False,
        )
        targets = select_writer_targets(
            bundle,
            harmful_activations,
            benign_activations,
            pilot_harmful,
            pilot_benign,
            max_targets=(
                max(1, int(cfg.kcrn_target_writers)) * 8
                if int(cfg.kcrn_target_writers) > 0
                else 10000
            ),
            writer_indices=writer_indices,
            mlp_only=mlp_only,
        )
    else:
        targets = _compatible_targets(bundle, selected_layers, writer_spec)
    if not targets:
        raise RuntimeError("KCRN found no compatible residual writer targets")

    target_layers = sorted({int(target["layer"]) for target in targets})
    all_positions = bool(cfg.kcrn_all_positions or cfg.kcrn_preserve_rank > 64)
    full_harmful = collect_writer_inputs(
        bundle,
        fit_harmful,
        layers=target_layers,
        batch_size=cfg.batch_size,
        all_positions=all_positions,
        max_samples=(int(cfg.kcrn_max_key_samples) if int(cfg.kcrn_max_key_samples) > 0 else None),
        seed=cfg.seed,
    )
    full_benign = collect_writer_inputs(
        bundle,
        fit_benign,
        layers=target_layers,
        batch_size=cfg.batch_size,
        all_positions=all_positions,
        max_samples=(int(cfg.kcrn_max_key_samples) if int(cfg.kcrn_max_key_samples) > 0 else None),
        seed=cfg.seed + 1,
    )
    harmful_rank = int(cfg.kcrn_harmful_rank or cfg.kcrn_key_rank)
    benign_rank = int(cfg.kcrn_preserve_rank or cfg.kcrn_key_rank)
    refusal_source = (cfg.kcrn_refusal_source or "activation").strip().lower()
    if refusal_source in {"head", "head_tokens", "output"}:
        head_basis = _head_refusal_basis(cfg, bundle)
        refusal_bases = {layer_index: head_basis for layer_index in target_layers}
    elif refusal_source in {"activation", "activations", "residual"}:
        refusal_bases = {
            layer_index: _refusal_basis(
                cfg,
                harmful_activations[layer_index],
                benign_activations[layer_index],
            )
            for layer_index in target_layers
        }
    elif refusal_source in {"response", "responses"}:
        response_n = min(
            len(fit_harmful),
            len(fit_benign),
            max(1, int(cfg.kcrn_refusal_response_n)),
        )
        response_harmful = fit_harmful[:response_n]
        response_benign = fit_benign[:response_n]
        harmful_responses = generate(
            bundle,
            response_harmful,
            max(1, int(cfg.kcrn_refusal_response_tokens)),
            cfg.batch_size,
        )
        benign_responses = generate(
            bundle,
            response_benign,
            max(1, int(cfg.kcrn_refusal_response_tokens)),
            cfg.batch_size,
        )
        response_harmful_activations = collect_response_activations(
            bundle,
            response_harmful,
            harmful_responses,
            cfg.batch_size,
        )
        response_benign_activations = collect_response_activations(
            bundle,
            response_benign,
            benign_responses,
            cfg.batch_size,
        )
        refusal_bases = {
            layer_index: _refusal_basis(
                cfg,
                response_harmful_activations[layer_index],
                response_benign_activations[layer_index],
            )
            for layer_index in target_layers
        }
    else:
        raise ValueError(f"unknown KCRN refusal source: {cfg.kcrn_refusal_source}")
    candidates = []
    certificates = []
    for target in targets:
        layer_index = int(target["layer"])
        writer_index = int(target["writer_index"])
        layer = bundle.layers()[layer_index]
        writer = bundle.layer_writers(layer)[writer_index]
        matrix, _transpose = writer_matrix(writer)
        matrix = matrix.to(dtype=torch.float64)
        refusal = refusal_bases[layer_index]
        if matrix.shape[0] != refusal.shape[0]:
            raise ValueError(f"KCRN refusal width does not match writer at layer {layer_index}")
        refusal = refusal.to(device=matrix.device, dtype=matrix.dtype)
        harmful_keys = key_basis(
            full_harmful[layer_index][writer_index].to(device=matrix.device, dtype=matrix.dtype),
            harmful_rank,
            tolerance=cfg.kcrn_basis_tolerance,
        )
        basis_mode = cfg.kcrn_benign_basis_mode or "legacy"
        benign_keys = key_basis(
            full_benign[layer_index][writer_index].to(device=matrix.device, dtype=matrix.dtype),
            benign_rank,
            mode=basis_mode,
            tolerance=cfg.kcrn_basis_tolerance,
            explained_variance=cfg.kcrn_benign_explained_variance,
        ).to(matrix.device)
        solver = (cfg.kcrn_solver or "").strip().lower()
        try:
            if solver == "projected":
                update = key_conditional_nulling_projected(
                    matrix,
                    refusal,
                    harmful_keys,
                    benign_keys,
                    strength=cfg.kcrn_strength,
                    ridge=cfg.kcrn_ridge,
                    min_projected_energy=cfg.kcrn_min_projected_energy,
                    max_condition=cfg.kcrn_max_condition,
                    max_relative_update=cfg.kcrn_max_relative_update,
                    svd_tolerance=cfg.kcrn_basis_tolerance,
                    explained_variance=cfg.kcrn_benign_explained_variance,
                )
            elif solver == "original":
                update = key_conditional_nulling(
                    matrix,
                    refusal,
                    harmful_keys,
                    benign_keys,
                    strength=cfg.kcrn_strength,
                    ridge=cfg.kcrn_ridge,
                )
            else:
                raise ValueError(f"unknown kcrn solver: {cfg.kcrn_solver}")
        except KCRNDegeneracyError as exc:
            certificate = {
                "layer": layer_index,
                "writer_index": writer_index,
                "writer": _writer_label(layer, writer),
                "solver": solver,
                "status": "skipped_degenerate",
                "skip_reason": str(exc),
                "selection_score": float(target.get("selection_score", 0.0)),
                **exc.diagnostics,
            }
            certificates.append(certificate)
            continue
        except RuntimeError as exc:
            certificate = {
                "layer": layer_index,
                "writer_index": writer_index,
                "writer": _writer_label(layer, writer),
                "solver": solver,
                "status": "skipped_solver_error",
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "selection_score": float(target.get("selection_score", 0.0)),
            }
            certificates.append(certificate)
            continue
        certificate = dict(update.diagnostics)
        certificate.update({
            "layer": layer_index,
            "writer_index": writer_index,
            "writer": _writer_label(layer, writer),
            "selection_score": float(target.get("selection_score", 0.0)),
            "harmful_basis_rank": int(harmful_keys.shape[1]),
            "benign_basis_rank": int(benign_keys.shape[1]),
            "refusal_basis_rank": int(refusal.shape[1]),
        })
        priority = target_priority(
            certificate["selection_score"],
            certificate["delta_fro"],
            certificate["condition"],
            cfg.kcrn_max_delta_norm,
            cfg.kcrn_max_condition,
        )
        certificate["priority"] = priority
        if priority is None:
            continue
        candidates.append({
            "priority": priority,
            "solver": solver,
            "target": target,
            "certificate": certificate,
            "preserve_basis": benign_keys.detach().cpu(),
            "edit": {
            "kind": "kcrn",
            "layer": layer_index,
            "writer_index": writer_index,
            "left": update.left.detach().cpu(),
            "right": update.right.detach().cpu(),
            },
        })
    if not candidates:
        raise RuntimeError("KCRN found no stable writer updates within the configured bounds")
    candidates.sort(key=lambda item: item["priority"], reverse=True)
    target_limit = int(cfg.kcrn_target_writers)
    selected = candidates if target_limit <= 0 else candidates[:target_limit]
    preserve_bases = {
        (int(item["edit"]["layer"]), int(item["edit"]["writer_index"])): item["preserve_basis"]
        for item in selected
        if item["solver"] == "projected"
    }
    return (
        [item["edit"] for item in selected],
        [item["target"] for item in selected],
        [item["certificate"] for item in selected],
        preserve_bases,
    )


def _aggressive_refusal_bases(
    cfg: ApostateConfig,
    bundle,
    fit_harmful: list[str],
    fit_benign: list[str],
    target_layers: list[int],
    harmful_activations: torch.Tensor,
    benign_activations: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Build refusal bases for aggressive candidates with the normal KCRN sources."""

    refusal_source = (cfg.kcrn_refusal_source or "activation").strip().lower()
    if refusal_source in {"head", "head_tokens", "output"}:
        head_basis = _head_refusal_basis(cfg, bundle)
        return {layer_index: head_basis for layer_index in target_layers}
    if refusal_source in {"activation", "activations", "residual"}:
        return {
            layer_index: _refusal_basis(
                cfg,
                harmful_activations[layer_index],
                benign_activations[layer_index],
            )
            for layer_index in target_layers
        }
    if refusal_source not in {"response", "responses"}:
        raise ValueError(f"unknown KCRN refusal source: {cfg.kcrn_refusal_source}")
    response_n = min(
        len(fit_harmful),
        len(fit_benign),
        max(1, int(cfg.kcrn_refusal_response_n)),
    )
    response_harmful = fit_harmful[:response_n]
    response_benign = fit_benign[:response_n]
    harmful_responses = generate(
        bundle,
        response_harmful,
        max(1, int(cfg.kcrn_refusal_response_tokens)),
        cfg.batch_size,
    )
    benign_responses = generate(
        bundle,
        response_benign,
        max(1, int(cfg.kcrn_refusal_response_tokens)),
        cfg.batch_size,
    )
    harmful_response_activations = collect_response_activations(
        bundle,
        response_harmful,
        harmful_responses,
        cfg.batch_size,
    )
    benign_response_activations = collect_response_activations(
        bundle,
        response_benign,
        benign_responses,
        cfg.batch_size,
    )
    return {
        layer_index: _refusal_basis(
            cfg,
            harmful_response_activations[layer_index],
            benign_response_activations[layer_index],
        )
        for layer_index in target_layers
    }


def _fit_aggressive_edits(
    cfg: ApostateConfig,
    bundle,
    fit_harmful: list[str],
    fit_benign: list[str],
    tune_harmful: list[str],
    calibration_benign: list[str],
    calibration_cache: list[Path],
) -> tuple[list[dict], list[dict], list[dict], dict[tuple[int, int], torch.Tensor], dict]:
    """Select projected KCRN factors with a calibration-only greedy controller."""

    baseline_edits, baseline_targets, baseline_certificates, baseline_preserve_bases = _fit_edits(
        cfg,
        bundle,
        fit_harmful,
        fit_benign,
    )
    baseline_keys = {
        (int(edit["layer"]), int(edit["writer_index"]))
        for edit in baseline_edits
    }
    baseline_certificate_by_key = {
        (int(certificate["layer"]), int(certificate["writer_index"])): certificate
        for certificate in baseline_certificates
        if "layer" in certificate and "writer_index" in certificate
    }
    baseline_trials = []
    for edit, target in zip(baseline_edits, baseline_targets):
        key = (int(edit["layer"]), int(edit["writer_index"]))
        certificate = dict(baseline_certificate_by_key.get(key, {}))
        certificate["status"] = "baseline"
        certificate["aggressive_baseline"] = True
        baseline_trials.append(
            AggressiveTrial(
                key,
                float(cfg.kcrn_strength),
                edit,
                certificate,
                float(certificate.get("priority", target.get("selection_score", 0.0))),
            )
        )
    baseline_trials = tuple(baseline_trials)

    harmful_activations = collect_activations(bundle, fit_harmful, cfg.batch_size)
    benign_activations = collect_activations(bundle, fit_benign, cfg.batch_size)
    layer_spec = (cfg.kcrn_layers or "auto").strip().lower()
    selected_layers = None if layer_spec == "auto" else parse_index_spec(
        layer_spec, bundle.num_layers, "layer"
    )
    writer_spec = (cfg.kcrn_writers or "auto").strip().lower()
    writer_indices = None
    mlp_only = False
    if writer_spec not in ("", "auto", "all"):
        mlp_only = writer_spec == "mlp"
        if not mlp_only:
            writer_indices = set(parse_index_spec(writer_spec, 128, "writer"))
    candidate_limit = int(cfg.kcrn_aggressive_candidate_limit)
    if int(cfg.kcrn_target_writers) > 0:
        candidate_limit = min(candidate_limit, int(cfg.kcrn_target_writers))
    candidate_limit = max(1, candidate_limit)
    if selected_layers is None or writer_spec == "auto":
        pilot_harmful = collect_writer_inputs(
            bundle,
            fit_harmful[: min(len(fit_harmful), max(1, int(cfg.kcrn_pilot_n)))],
            layers=selected_layers,
            batch_size=cfg.batch_size,
            all_positions=False,
        )
        pilot_benign = collect_writer_inputs(
            bundle,
            fit_benign[: min(len(fit_benign), max(1, int(cfg.kcrn_pilot_n)))],
            layers=selected_layers,
            batch_size=cfg.batch_size,
            all_positions=False,
        )
        targets = select_writer_targets(
            bundle,
            harmful_activations,
            benign_activations,
            pilot_harmful,
            pilot_benign,
            max_targets=max(candidate_limit * 4, candidate_limit),
            writer_indices=writer_indices,
            mlp_only=mlp_only,
        )
    else:
        targets = _compatible_targets(bundle, selected_layers, writer_spec)
    if not targets:
        raise RuntimeError("aggressive KCRN found no compatible residual writer targets")

    target_layers = sorted({int(target["layer"]) for target in targets})
    all_positions = bool(cfg.kcrn_all_positions or cfg.kcrn_preserve_rank > 64)
    max_samples = int(cfg.kcrn_max_key_samples)
    full_harmful = collect_writer_inputs(
        bundle,
        fit_harmful,
        layers=target_layers,
        batch_size=cfg.batch_size,
        all_positions=all_positions,
        max_samples=max_samples if max_samples > 0 else None,
        seed=cfg.seed,
    )
    full_benign = collect_writer_inputs(
        bundle,
        fit_benign,
        layers=target_layers,
        batch_size=cfg.batch_size,
        all_positions=all_positions,
        max_samples=max_samples if max_samples > 0 else None,
        seed=cfg.seed + 1,
    )
    refusal_bases = _aggressive_refusal_bases(
        cfg,
        bundle,
        fit_harmful,
        fit_benign,
        target_layers,
        harmful_activations,
        benign_activations,
    )
    harmful_rank = int(cfg.kcrn_harmful_rank or cfg.kcrn_key_rank)
    benign_rank = int(cfg.kcrn_preserve_rank or cfg.kcrn_key_rank)
    candidates = []
    skipped_certificates = []
    for target in targets:
        layer_index = int(target["layer"])
        writer_index = int(target["writer_index"])
        layer = bundle.layers()[layer_index]
        writer = bundle.layer_writers(layer)[writer_index]
        try:
            matrix, _transpose = writer_matrix(writer)
            matrix = matrix.to(dtype=torch.float64)
            refusal = refusal_bases[layer_index].to(device=matrix.device, dtype=matrix.dtype)
            if matrix.shape[0] != refusal.shape[0]:
                raise ValueError(f"KCRN refusal width does not match writer at layer {layer_index}")
            harmful_keys = key_basis(
                full_harmful[layer_index][writer_index].to(device=matrix.device, dtype=matrix.dtype),
                harmful_rank,
                tolerance=cfg.kcrn_basis_tolerance,
            )
            benign_keys = key_basis(
                full_benign[layer_index][writer_index].to(device=matrix.device, dtype=matrix.dtype),
                benign_rank,
                mode=cfg.kcrn_benign_basis_mode or "raw",
                tolerance=cfg.kcrn_basis_tolerance,
                explained_variance=cfg.kcrn_benign_explained_variance,
            ).to(matrix.device)
            refusal_signal = float(torch.linalg.norm(refusal.T @ matrix @ harmful_keys).item())
            selection_score = float(target.get("selection_score", 0.0))
            static_priority = max(selection_score, refusal_signal)
            candidates.append({
                "key": (layer_index, writer_index),
                "target": target,
                "writer": writer,
                "refusal": refusal.detach().cpu(),
                "harmful_keys": harmful_keys.detach().cpu(),
                "benign_keys": benign_keys.detach().cpu(),
                "preserve_basis": benign_keys.detach().cpu(),
                "static_priority": static_priority,
            })
        except (KCRNDegeneracyError, RuntimeError, TypeError, ValueError) as exc:
            skipped_certificates.append({
                "layer": layer_index,
                "writer_index": writer_index,
                "writer": _writer_label(layer, writer),
                "status": "skipped_candidate_setup",
                "skip_reason": f"{type(exc).__name__}: {exc}",
                "selection_score": float(target.get("selection_score", 0.0)),
            })
    candidates.sort(
        key=lambda item: (
            -float(item["static_priority"]),
            int(item["key"][0]),
            int(item["key"][1]),
        )
    )
    candidates = [item for item in candidates if item["key"] not in baseline_keys]
    candidates = candidates[:candidate_limit]
    if not candidates and not baseline_trials:
        raise RuntimeError("aggressive KCRN found no stable candidate writer updates")
    candidate_by_key = {item["key"]: item for item in candidates}
    strengths = parse_strength_grid(cfg.kcrn_aggressive_strengths)
    scoring_harmful = tune_harmful[: min(
        len(tune_harmful),
        int(cfg.kcrn_aggressive_scoring_harmful_n),
    )]
    search_calibration_benign = calibration_benign[: min(
        len(calibration_benign),
        int(cfg.kcrn_aggressive_calibration_n),
    )]

    def _writer_for_key(key):
        layer = bundle.layers()[key[0]]
        return bundle.layer_writers(layer)[key[1]]

    def build_trials(_accepted, remaining):
        trials = []
        for key in sorted(remaining):
            candidate = candidate_by_key[key]
            writer = _writer_for_key(key)
            matrix, _transpose = writer_matrix(writer)
            matrix = matrix.to(dtype=torch.float64)
            refusal = candidate["refusal"].to(device=matrix.device, dtype=matrix.dtype)
            harmful_keys = candidate["harmful_keys"].to(device=matrix.device, dtype=matrix.dtype)
            benign_keys = candidate["benign_keys"].to(device=matrix.device, dtype=matrix.dtype)
            for strength in strengths:
                try:
                    update = key_conditional_nulling_projected(
                        matrix,
                        refusal,
                        harmful_keys,
                        benign_keys,
                        strength=strength,
                        ridge=cfg.kcrn_ridge,
                        min_projected_energy=cfg.kcrn_min_projected_energy,
                        max_condition=cfg.kcrn_max_condition,
                        max_relative_update=cfg.kcrn_max_relative_update,
                        svd_tolerance=cfg.kcrn_basis_tolerance,
                        explained_variance=cfg.kcrn_benign_explained_variance,
                        preserve_basis_orthonormal=True,
                        diagnostics_spectrum=False,
                    )
                    certificate = dict(update.diagnostics)
                    certificate.update({
                        "layer": key[0],
                        "writer_index": key[1],
                        "writer": candidate["target"].get("writer", type(writer).__name__),
                        "selection_score": float(candidate["target"].get("selection_score", 0.0)),
                        "static_priority": float(candidate["static_priority"]),
                        "harmful_basis_rank": int(harmful_keys.shape[1]),
                        "benign_basis_rank": int(benign_keys.shape[1]),
                        "refusal_basis_rank": int(refusal.shape[1]),
                    })
                    if float(certificate["delta_fro"]) > float(cfg.kcrn_max_delta_norm):
                        certificate.update({
                            "status": "skipped_delta_norm",
                            "skip_reason": (
                                f"delta norm {certificate['delta_fro']:.6g} exceeds "
                                f"maximum {float(cfg.kcrn_max_delta_norm):.6g}"
                            ),
                        })
                        trials.append(AggressiveTrial(key, strength, {}, certificate, candidate["static_priority"]))
                        continue
                    certificate["status"] = "candidate"
                    trials.append(AggressiveTrial(
                        key,
                        strength,
                        {
                            "kind": "kcrn",
                            "layer": key[0],
                            "writer_index": key[1],
                            "left": update.left.detach().cpu(),
                            "right": update.right.detach().cpu(),
                        },
                        certificate,
                        candidate["static_priority"],
                    ))
                except KCRNDegeneracyError as exc:
                    certificate = {
                        "layer": key[0],
                        "writer_index": key[1],
                        "writer": candidate["target"].get("writer", type(writer).__name__),
                        "status": "skipped_degenerate",
                        "skip_reason": str(exc),
                        "selection_score": float(candidate["target"].get("selection_score", 0.0)),
                        **exc.diagnostics,
                    }
                    trials.append(AggressiveTrial(key, strength, {}, certificate, candidate["static_priority"]))
                except (RuntimeError, TypeError, ValueError) as exc:
                    certificate = {
                        "layer": key[0],
                        "writer_index": key[1],
                        "writer": candidate["target"].get("writer", type(writer).__name__),
                        "status": "skipped_solver_error",
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                        "selection_score": float(candidate["target"].get("selection_score", 0.0)),
                    }
                    trials.append(AggressiveTrial(key, strength, {}, certificate, candidate["static_priority"]))
        trials.sort(
            key=lambda trial: (
                1 if str(trial.certificate.get("status", "")).startswith("skipped") else 0,
                float(trial.certificate.get("refusal_residual", math.inf)),
                -float(trial.priority),
                int(trial.target_key[0]),
                int(trial.target_key[1]),
                float(trial.strength),
            )
        )
        return shortlist_trials(
            trials,
            strengths,
            max(1, int(cfg.kcrn_aggressive_probe_candidates)),
        )

    def evaluate(stack) -> AggressiveEvaluation:
        full_stack = merge_trial_stacks(baseline_trials, stack)
        snapshots = []
        try:
            for trial in full_stack:
                snapshots.append(snapshot_writer(_writer_for_key(trial.target_key)))
            for trial in full_stack:
                apply_kcrn_factor(
                    _writer_for_key(trial.target_key),
                    trial.edit["left"],
                    trial.edit["right"],
                )
            prefix_margin = response_prefix_margin(
                bundle,
                scoring_harmful,
                cfg.batch_size,
            )
            calibration_kl = _full_position_kl(
                bundle,
                search_calibration_benign,
                cfg.batch_size,
                calibration_cache,
                int(cfg.kcrn_kl_max_length),
                positions=int(cfg.kcrn_aggressive_calibration_positions),
            )
            return AggressiveEvaluation(
                refusal_margin=-prefix_margin,
                calibration_kl=calibration_kl,
            )
        finally:
            for snapshot in reversed(snapshots):
                restore_writer(snapshot)

    initial = evaluate(())
    selection = greedy_select(
        [item["key"] for item in candidates],
        strengths,
        initial,
        build_trials,
        evaluate,
        max_steps=min(
            int(cfg.kcrn_aggressive_max_steps),
            int(cfg.kcrn_target_writers) if int(cfg.kcrn_target_writers) > 0 else int(cfg.kcrn_aggressive_max_steps),
        ),
        calibration_kl_budget=cfg.kcrn_aggressive_calibration_kl_budget,
        cumulative_relative_update_budget=cfg.kcrn_aggressive_max_cumulative_relative_update,
        min_margin_improvement=cfg.kcrn_aggressive_min_margin_improvement,
    )
    if not selection.accepted and not baseline_trials:
        raise RuntimeError("aggressive KCRN accepted no calibration-improving writer updates")
    adaptive_trials = tuple(selection.accepted)
    all_trials = merge_trial_stacks(baseline_trials, adaptive_trials)
    edits = [trial.edit for trial in all_trials]
    selected_targets = list(baseline_targets) + [
        candidate_by_key[trial.target_key]["target"] for trial in adaptive_trials
    ]
    certificates = list(skipped_certificates)
    for certificate in baseline_certificates:
        baseline_certificate = dict(certificate)
        baseline_certificate.update({
            "status": "baseline",
            "aggressive_baseline": True,
        })
        certificates.append(baseline_certificate)
    for step_index, trial in enumerate(selection.accepted):
        certificate = dict(trial.certificate)
        record = next(
            item
            for item in reversed(selection.records)
            if item.get("status") == "accepted"
            and item.get("layer") == trial.target_key[0]
            and item.get("writer_index") == trial.target_key[1]
            and item.get("strength") == float(trial.strength)
        )
        certificate.update({
            "aggressive_step": step_index,
            "calibration_prefix_margin": -record["refusal_margin"],
            "calibration_kl": record["calibration_kl"],
            "cumulative_relative_update": record["cumulative_relative_update"],
            "prefix_margin_improvement": record["margin_improvement"],
        })
        certificates.append(certificate)
    preserve_bases = dict(baseline_preserve_bases)
    preserve_bases.update({
        trial.target_key: candidate_by_key[trial.target_key]["preserve_basis"]
        for trial in adaptive_trials
    })
    detailed_certificates = []
    for trial in selection.accepted:
        candidate = candidate_by_key[trial.target_key]
        writer = _writer_for_key(trial.target_key)
        matrix, _transpose = writer_matrix(writer)
        matrix = matrix.to(dtype=torch.float64)
        update = key_conditional_nulling_projected(
            matrix,
            candidate["refusal"].to(device=matrix.device, dtype=matrix.dtype),
            candidate["harmful_keys"].to(device=matrix.device, dtype=matrix.dtype),
            candidate["benign_keys"].to(device=matrix.device, dtype=matrix.dtype),
            strength=trial.strength,
            ridge=cfg.kcrn_ridge,
            min_projected_energy=cfg.kcrn_min_projected_energy,
            max_condition=cfg.kcrn_max_condition,
            max_relative_update=cfg.kcrn_max_relative_update,
            svd_tolerance=cfg.kcrn_basis_tolerance,
            explained_variance=cfg.kcrn_benign_explained_variance,
            preserve_basis_orthonormal=True,
            diagnostics_spectrum=True,
        )
        detailed = dict(update.diagnostics)
        detailed.update(trial.certificate)
        detailed_certificates.append(detailed)
    baseline_relative_update = sum(
        float(certificate.get("relative_update_norm", 0.0))
        for certificate in baseline_certificates
        if math.isfinite(float(certificate.get("relative_update_norm", 0.0)))
    )
    controller = {
        "candidate_order": [
            {
                "layer": item["key"][0],
                "writer_index": item["key"][1],
                "selection_score": float(item["target"].get("selection_score", 0.0)),
                "static_priority": float(item["static_priority"]),
            }
            for item in candidates
        ],
        "records": list(selection.records),
        "baseline_steps": len(baseline_trials),
        "adaptive_accepted_steps": len(selection.accepted),
        "accepted_steps": len(all_trials),
        "baseline_strength": float(cfg.kcrn_strength),
        "calibration_prefix_margin_before": float(-initial.refusal_margin),
        "calibration_prefix_margin_after": float(-selection.final_evaluation.refusal_margin),
        "calibration_kl_after": float(selection.final_evaluation.calibration_kl),
        "baseline_cumulative_relative_update": float(baseline_relative_update),
        "cumulative_relative_update": float(
            baseline_relative_update + selection.cumulative_relative_update
        ),
        "tuning_harmful_n": len(tune_harmful),
        "scoring_harmful_n": len(scoring_harmful),
        "key_fit_harmful_n": len(fit_harmful),
        "calibration_benign_n": len(search_calibration_benign),
    }
    certificates.extend(detailed_certificates)
    return edits, selected_targets, certificates, preserve_bases, controller


def _release_bundle(bundle) -> None:
    bundle.model = None
    bundle.tokenizer = None
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def _validation_head(model):
    getter = getattr(model, "get_output_embeddings", None)
    head = getter() if callable(getter) else None
    if head is None:
        head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if head is None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    return head


def _validation_logit_scale(model) -> float:
    value = getattr(getattr(model, "config", None), "logits_scaling", 1.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 / value if math.isfinite(value) and value != 0.0 else 1.0


def _validation_base(model):
    getter = getattr(model, "get_base_model", None)
    base = getter() if callable(getter) else None
    return None if base is model else base


def _validation_hidden(bundle, encoded):
    base = _validation_base(bundle.model)
    head = _validation_head(bundle.model)
    if base is None or head is None:
        return None, None, 1.0
    try:
        output = base(**encoded, use_cache=False, return_dict=True)
    except TypeError:
        output = base(**encoded, use_cache=False)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)):
        hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        return None, None, 1.0
    return hidden, head, _validation_logit_scale(bundle.model)


def _head_logits(hidden: torch.Tensor, head, start: int, stop: int) -> torch.Tensor:
    weight = head.weight[start:stop]
    bias = getattr(head, "bias", None)
    if isinstance(bias, torch.Tensor):
        bias = bias[start:stop]
    return torch.nn.functional.linear(hidden, weight, bias)


def _chunked_pointwise_kl(
    base_hidden: torch.Tensor,
    edited_hidden: torch.Tensor,
    head,
    vocab_chunk_size: int = 8192,
    logit_scale: float = 1.0,
) -> torch.Tensor:
    if base_hidden.shape != edited_hidden.shape or base_hidden.ndim != 3:
        raise ValueError("validation hidden states must have identical [batch, positions, hidden] shapes")
    if int(vocab_chunk_size) < 1:
        raise ValueError("vocab_chunk_size must be positive")
    vocab_size = int(head.weight.shape[0])
    base_norm = torch.full(
        base_hidden.shape[:2],
        -torch.inf,
        dtype=torch.float32,
        device=base_hidden.device,
    )
    edited_norm = base_norm.clone()
    for start in range(0, vocab_size, int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), vocab_size)
        base_chunk = _head_logits(base_hidden, head, start, stop) * logit_scale
        edited_chunk = _head_logits(edited_hidden, head, start, stop) * logit_scale
        base_norm = torch.logaddexp(
            base_norm,
            torch.logsumexp(base_chunk.float(), dim=-1),
        )
        edited_norm = torch.logaddexp(
            edited_norm,
            torch.logsumexp(edited_chunk.float(), dim=-1),
        )
    pointwise = torch.zeros_like(base_norm)
    for start in range(0, vocab_size, int(vocab_chunk_size)):
        stop = min(start + int(vocab_chunk_size), vocab_size)
        base_chunk = (_head_logits(base_hidden, head, start, stop) * logit_scale).float()
        edited_chunk = (_head_logits(edited_hidden, head, start, stop) * logit_scale).float()
        base_lp = base_chunk - base_norm.unsqueeze(-1)
        edited_lp = edited_chunk - edited_norm.unsqueeze(-1)
        pointwise += (base_lp.exp() * (base_lp - edited_lp)).sum(dim=-1)
    return pointwise


@torch.inference_mode()
def _cache_full_position_logits(
    bundle,
    instructions: list[str],
    batch_size: int,
    cache_dir: Path,
    max_length: int,
    prefix: str = "base",
) -> list[Path]:
    """Cache final hidden states or logits for reload-based validation."""

    device = next(bundle.model.parameters()).device
    prompts = format_chat(bundle.tokenizer, instructions)
    paths = []
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        encoded = bundle.tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden, _head, _scale = _validation_hidden(bundle, encoded)
        if hidden is None:
            entry = {"logits": bundle.model(**encoded, use_cache=False).logits.detach().cpu()}
        else:
            entry = {"hidden_states": hidden.detach().cpu()}
        entry["attention_mask"] = encoded["attention_mask"].cpu()
        path = cache_dir / f"{prefix}-{batch_index:05d}.pt"
        torch.save(entry, path)
        paths.append(path)
        del encoded, hidden, entry
    return paths


@torch.inference_mode()
def _full_position_kl(
    bundle,
    instructions: list[str],
    batch_size: int,
    base_cache: list[Path],
    max_length: int,
    positions: Optional[int] = None,
) -> float:
    """Return float32 KL(base||edited) over every non-padding prompt position."""

    device = next(bundle.model.parameters()).device
    prompts = format_chat(bundle.tokenizer, instructions)
    total = 0.0
    count = 0.0
    for batch_index, start in enumerate(range(0, len(prompts), batch_size)):
        encoded = bundle.tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        cached = torch.load(base_cache[batch_index], map_location=device, weights_only=False)
        mask = cached["attention_mask"].to(device=device, dtype=torch.bool)
        if positions is not None:
            if int(positions) < 1:
                raise ValueError("KL positions must be positive")
            keep = min(int(positions), int(mask.shape[1]))
            mask = mask[:, -keep:]
        if "hidden_states" in cached:
            edited_hidden, head, logit_scale = _validation_hidden(bundle, encoded)
            if edited_hidden is None or head is None:
                raise RuntimeError("edited model cannot expose final hidden states and output embeddings")
            base_hidden = cached["hidden_states"].to(device)
            if positions is not None:
                base_hidden = base_hidden[:, -keep:, :]
                edited_hidden = edited_hidden[:, -keep:, :]
            pointwise = _chunked_pointwise_kl(
                base_hidden,
                edited_hidden,
                head,
                logit_scale=logit_scale,
            )
        else:
            if positions is not None:
                logits_kwargs = {}
                is_diffusion = getattr(bundle, "is_block_diffusion", lambda: False)()
                if not is_diffusion:
                    kw = _logits_kwarg(bundle)
                    if kw:
                        logits_kwargs[kw] = keep
                edited_logits = bundle.model(**encoded, use_cache=False, **logits_kwargs).logits
                base_logits = cached["logits"][:, -keep:, :].to(device)
                edited_logits = edited_logits[:, -keep:, :]
            else:
                edited_logits = bundle.model(**encoded, use_cache=False).logits
                base_logits = cached["logits"].to(device)
            pointwise = _pointwise_kl_from_logits(base_logits, edited_logits)
        total += float((pointwise * mask).sum().item())
        count += float(mask.sum().item())
        del encoded, cached, mask, pointwise
    return max(0.0, total / max(count, 1.0))


def run(cfg: ApostateConfig, command: Optional[str] = None) -> dict:
    """Fit KCRN factors, bake them into weights, and write a validation report."""

    del command
    started = time.time()
    cfg.method = "kcrn"
    cfg.compute_dtype = cfg.kcrn_compute_dtype
    cfg.save_dtype = cfg.kcrn_save_dtype
    if cfg.compute_dtype != cfg.save_dtype:
        raise ValueError("KCRN requires matching compute and save dtypes for raw KL validation")
    cfg.load_in_4bit = False
    if cfg.batch_size == 24:
        cfg.batch_size = 4
    cfg.with_defaults()
    cfg.load_in_4bit = False
    guard_model_and_output(cfg.model, cfg.output_dir, bool(cfg.kcrn_force))
    fit_n = int(cfg.kcrn_harmful_fit_n or cfg.kcrn_fit_n or cfg.n_harmful)
    benign_fit_n = int(cfg.kcrn_benign_fit_n or fit_n)
    eval_n = int(cfg.kcrn_eval_n or cfg.n_eval)
    if fit_n < 1 or benign_fit_n < 1 or eval_n < 1:
        raise ValueError("KCRN fit and evaluation counts must be positive")
    aggressive = (cfg.profile or "").strip().lower() == "aggressive-kcrn"
    tune_n = max(1, int(cfg.kcrn_aggressive_tune_n))
    tune_harmful = []
    if aggressive and not cfg.kcrn_edits:
        fit_harmful, tune_harmful, eval_harmful = _resolve_aggressive_prompt_sets(
            cfg.harmful_path,
            cfg.harmful_test or cfg.harmful_path,
            fit_n,
            tune_n,
            eval_n,
            cfg.seed,
        )
    else:
        fit_harmful, eval_harmful = _resolve_disjoint_prompt_sets(
            cfg.harmful_path,
            cfg.harmful_test or cfg.harmful_path,
            fit_n,
            eval_n,
            cfg.seed,
        )
    fit_benign, eval_benign = _resolve_disjoint_prompt_sets(
        cfg.harmless_path,
        cfg.kl_eval_path or cfg.harmless_path,
        benign_fit_n,
        eval_n,
        cfg.seed + 1,
    )
    calibration_eval_n = min(
        benign_fit_n,
        int(cfg.kcrn_calibration_eval_n or benign_fit_n),
    )
    if calibration_eval_n < 1:
        raise ValueError("kcrn_calibration_eval_n must be positive")
    calibration_benign = fit_benign[:calibration_eval_n]
    key_fit_harmful = fit_harmful
    if aggressive and not cfg.kcrn_edits:
        if len(fit_harmful) < 1 or len(tune_harmful) < 1:
            raise ValueError("aggressive-kcrn needs non-empty harmful fit and tuning prompts")
    print(
        f"[kcrn] loading {cfg.model} with {cfg.compute_dtype} on {cfg.device}",
        flush=True,
    )
    bundle = load_model(cfg)
    tokenizer = bundle.tokenizer
    num_layers = int(bundle.num_layers)
    hidden_size = int(bundle.hidden_size)
    targets = []
    certificates = []
    source_metadata = {}
    preserve_bases = {}
    aggressive_controller = {}
    delivery = None
    response_lengths = None
    cache_parent = Path(cfg.output_dir).expanduser().resolve().parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".apostate-kcrn-kl-",
        dir=str(cache_parent),
    ) as temp_name:
        if aggressive and not cfg.kcrn_edits:
            base_cache = _cache_full_position_logits(
                bundle,
                calibration_benign,
                cfg.batch_size,
                Path(temp_name),
                int(cfg.kcrn_kl_max_length),
                prefix="calibration",
            )
        if cfg.kcrn_edits:
            edits, source_metadata = _load_saved_edits(cfg.kcrn_edits)
            targets = [
                {"layer": int(edit["layer"]), "writer_index": int(edit["writer_index"])}
                for edit in edits
            ]
        elif aggressive:
            print(
                f"[kcrn] aggressive search using {len(key_fit_harmful)} harmful key-fit, "
                f"{len(tune_harmful)} harmful tuning, and {benign_fit_n} benign prompts",
                flush=True,
            )
            (
                edits,
                targets,
                certificates,
                preserve_bases,
                aggressive_controller,
            ) = _fit_aggressive_edits(
                cfg,
                bundle,
                key_fit_harmful,
                fit_benign,
                tune_harmful,
                calibration_benign,
                base_cache,
            )
        else:
            print(
                f"[kcrn] fitting {fit_n} harmful and {benign_fit_n} benign prompts "
                f"with {eval_n} disjoint held-out prompts",
                flush=True,
            )
            edits, targets, certificates, preserve_bases = _fit_edits(
                cfg, bundle, fit_harmful, fit_benign
            )
        if not (aggressive and not cfg.kcrn_edits):
            base_cache = _cache_full_position_logits(
                bundle,
                calibration_benign,
                cfg.batch_size,
                Path(temp_name),
                int(cfg.kcrn_kl_max_length),
                prefix="calibration",
            )
        heldout_cache = _cache_full_position_logits(
            bundle,
            eval_benign,
            cfg.batch_size,
            Path(temp_name),
            int(cfg.kcrn_kl_max_length),
            prefix="heldout",
        )
        print("[kcrn] baking fixed factors into a standalone checkpoint", flush=True)
        post_bake_metrics = {}
        bake(
            cfg,
            {"edits": edits},
            tokenizer=tokenizer,
            model=bundle.model,
            preserve_bases=preserve_bases,
            post_bake_metrics=post_bake_metrics,
        )
        _release_bundle(bundle)
        eval_cfg = ApostateConfig(
            method="kcrn",
            model=cfg.output_dir,
            output_dir=cfg.output_dir,
            device=cfg.device,
            load_in_4bit=False,
            cpu_offload_gb=cfg.cpu_offload_gb,
            compute_dtype=cfg.save_dtype,
            save_dtype=cfg.save_dtype,
            batch_size=cfg.batch_size,
        )
        edited = load_model(eval_cfg)
        calibration_kl = _full_position_kl(
            edited,
            calibration_benign,
            cfg.batch_size,
            base_cache,
            int(cfg.kcrn_kl_max_length),
        )
        heldout_kl = _full_position_kl(
            edited,
            eval_benign,
            cfg.batch_size,
            heldout_cache,
            int(cfg.kcrn_kl_max_length),
        )
        print(f"[kcrn] reloaded calibration KL: {calibration_kl:.8f}", flush=True)
        print(f"[kcrn] reloaded held-out KL: {heldout_kl:.8f}", flush=True)
        if cfg.kcrn_eval_generation:
            outputs = generate(edited, eval_harmful, cfg.max_new_tokens, cfg.batch_size)
            refused = judge_strict_refusal(edited, outputs, cfg.batch_size, eval_harmful)
            delivery = 1.0 - sum(refused) / max(1, len(refused))
            lengths = sorted(len(output.strip()) for output in outputs)
            response_lengths = {
                "min": min(lengths, default=0),
                "median": lengths[len(lengths) // 2] if lengths else 0,
                "max": max(lengths, default=0),
            }
        _release_bundle(edited)
    aggressive_acceptance = None
    if aggressive:
        aggressive_acceptance = {
            "target_delivery": float(cfg.kcrn_aggressive_target_delivery),
            "target_heldout_benign_kl": float(cfg.kcrn_aggressive_heldout_kl_max),
            "measured_delivery": delivery,
            "measured_heldout_benign_kl": heldout_kl,
            "success": bool(
                delivery is not None
                and float(delivery) >= float(cfg.kcrn_aggressive_target_delivery)
                and float(heldout_kl) <= float(cfg.kcrn_aggressive_heldout_kl_max)
            ),
        }
    report = _finite_json({
        "method": "kcrn",
        "profile": cfg.profile,
        "variant": "aggressive-kcrn" if aggressive else "kcrn",
        "deployment": "fixed_weight_checkpoint",
        "runtime_hooks": False,
        "model": cfg.model,
        "output_dir": str(Path(cfg.output_dir).expanduser().resolve()),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "compute_dtype": cfg.compute_dtype,
        "save_dtype": cfg.save_dtype,
        "fit_n": fit_n,
        "harmful_key_fit_n": len(key_fit_harmful),
        "harmful_tuning_n": len(tune_harmful),
        "benign_fit_n": benign_fit_n,
        "eval_n": eval_n,
        "calibration_n": calibration_eval_n,
        "calibration_eval_n": calibration_eval_n,
        "heldout_n": eval_n,
        "kl_protocol": "float32 KL(base||edited) over every non-padding prompt position",
        "validation": "reloaded fixed-weight checkpoint against cached native-dtype base hidden states and chunked output projection",
        "validation_reloaded": True,
        "kl_max_length": int(cfg.kcrn_kl_max_length),
        "kcrn_solver": cfg.kcrn_solver if not cfg.kcrn_edits else "precomputed",
        "kcrn_strength": cfg.kcrn_strength if not cfg.kcrn_edits else source_metadata.get("strength"),
        "kcrn_ridge": cfg.kcrn_ridge if not cfg.kcrn_edits else source_metadata.get("ridge"),
        "kcrn_benign_basis_mode": cfg.kcrn_benign_basis_mode,
        "kcrn_benign_explained_variance": cfg.kcrn_benign_explained_variance,
        "kcrn_refusal_rank": cfg.kcrn_refusal_rank,
        "kcrn_refusal_source": cfg.kcrn_refusal_source,
        "kcrn_refusal_response_n": cfg.kcrn_refusal_response_n,
        "kcrn_refusal_response_tokens": cfg.kcrn_refusal_response_tokens,
        "kcrn_refusal_multi": cfg.kcrn_refusal_multi,
        "kcrn_refusal_clusters": cfg.kcrn_refusal_clusters,
        "kcrn_basis_tolerance": cfg.kcrn_basis_tolerance,
        "kcrn_max_key_samples": cfg.kcrn_max_key_samples,
        "kcrn_min_projected_energy": cfg.kcrn_min_projected_energy,
        "kcrn_max_relative_update": cfg.kcrn_max_relative_update,
        "kcrn_max_condition": cfg.kcrn_max_condition,
        "kcrn_aggressive_strengths": cfg.kcrn_aggressive_strengths,
        "kcrn_aggressive_max_steps": cfg.kcrn_aggressive_max_steps,
        "kcrn_aggressive_candidate_limit": cfg.kcrn_aggressive_candidate_limit,
        "kcrn_aggressive_probe_candidates": cfg.kcrn_aggressive_probe_candidates,
        "kcrn_aggressive_tune_n": cfg.kcrn_aggressive_tune_n,
        "kcrn_aggressive_scoring_harmful_n": cfg.kcrn_aggressive_scoring_harmful_n,
        "kcrn_aggressive_calibration_n": cfg.kcrn_aggressive_calibration_n,
        "kcrn_aggressive_calibration_positions": cfg.kcrn_aggressive_calibration_positions,
        "kcrn_aggressive_calibration_kl_budget": cfg.kcrn_aggressive_calibration_kl_budget,
        "kcrn_aggressive_heldout_kl_max": cfg.kcrn_aggressive_heldout_kl_max,
        "kcrn_aggressive_max_cumulative_relative_update": cfg.kcrn_aggressive_max_cumulative_relative_update,
        "kcrn_aggressive_min_margin_improvement": cfg.kcrn_aggressive_min_margin_improvement,
        "kcrn_aggressive_target_delivery": cfg.kcrn_aggressive_target_delivery,
        "aggressive_controller": aggressive_controller,
        "aggressive_acceptance": aggressive_acceptance,
        "generation_token_budget": int(cfg.max_new_tokens),
        "harmful_fit_source": cfg.harmful_path,
        "harmful_holdout_source": cfg.harmful_test,
        "benign_fit_source": cfg.harmless_path,
        "benign_holdout_source": cfg.kl_eval_path,
        "targets": targets,
        "certificates": certificates,
        "factor_storage_bytes": int(sum(
            edit["left"].numel() * edit["left"].element_size()
            + edit["right"].numel() * edit["right"].element_size()
            for edit in edits
        )),
        "calibration_kl": calibration_kl,
        "heldout_benign_kl": heldout_kl,
        "harmless_full_position_kl": heldout_kl,
        "post_bake_preservation": post_bake_metrics,
        "harmful_delivery": delivery,
        "harmful_response_lengths": response_lengths,
        "elapsed_sec": round(time.time() - started, 1),
    })
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "kcrn_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "apostate_config.json").write_text(cfg.to_json() + "\n", encoding="utf-8")
    _write_model_card(output, report)
    print(json.dumps(report, indent=2), flush=True)
    bundle.model = None
    bundle.tokenizer = None
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
