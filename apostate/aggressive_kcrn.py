"""Calibration-controlled selection for the fixed-weight aggressive KCRN variant."""

from __future__ import annotations

from dataclasses import dataclass
import math
from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import torch

from .bake import _edit_kcrn_writer


TargetKey = tuple[int, int]


@dataclass(frozen=True)
class AggressiveEvaluation:
    """Calibration measurements for one complete candidate stack."""

    refusal_margin: float
    calibration_kl: float
    harmful_delivery: float | None = None


@dataclass(frozen=True)
class AggressiveTrial:
    """One projected KCRN factor proposed for a residual writer."""

    target_key: TargetKey
    strength: float
    edit: dict
    certificate: dict
    priority: float = 0.0


@dataclass(frozen=True)
class AggressiveSelection:
    """Accepted trials and the measurements collected while selecting them."""

    accepted: tuple[AggressiveTrial, ...]
    records: tuple[dict, ...]
    final_evaluation: AggressiveEvaluation
    cumulative_relative_update: float


@dataclass(frozen=True)
class WriterSnapshot:
    """Exact storage for one temporarily edited residual writer."""

    module: torch.nn.Module
    attribute: str
    value: torch.Tensor


def snapshot_writer(module: torch.nn.Module) -> WriterSnapshot:
    """Copy one supported writer so a calibration trial can be reverted exactly."""

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return WriterSnapshot(module, "weight", weight.detach().clone())
    packed = getattr(module, "down_proj", None)
    if isinstance(packed, torch.Tensor) and packed.ndim == 3:
        return WriterSnapshot(module, "down_proj", packed.detach().clone())
    raise TypeError(f"unsupported aggressive KCRN writer type: {type(module).__name__}")


def restore_writer(snapshot: WriterSnapshot) -> None:
    """Restore one writer without changing its parameter identity or dtype."""

    parameter = getattr(snapshot.module, snapshot.attribute)
    if not isinstance(parameter, torch.Tensor) or parameter.shape != snapshot.value.shape:
        raise TypeError("writer changed shape while an aggressive KCRN trial was active")
    parameter.data.copy_(snapshot.value)


def apply_kcrn_factor(
    module: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict:
    """Apply one factorized update using the production bake arithmetic."""

    return _edit_kcrn_writer(module, left, right)


@contextmanager
def temporary_kcrn_factor(
    module: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
):
    """Apply one trial factor and restore the exact writer storage afterward."""

    snapshot = snapshot_writer(module)
    try:
        metrics = apply_kcrn_factor(module, left, right)
        yield metrics
    finally:
        restore_writer(snapshot)


def parse_strength_grid(spec: str | Sequence[float]) -> tuple[float, ...]:
    """Parse a finite, positive, deterministic strength grid."""

    if isinstance(spec, str):
        values = [part.strip() for part in spec.split(",") if part.strip()]
    else:
        values = list(spec)
    if not values:
        raise ValueError("aggressive KCRN strength grid must contain at least one value")
    parsed = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid aggressive KCRN strength: {value!r}") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"aggressive KCRN strengths must be finite and positive: {value!r}")
        parsed.append(number)
    return tuple(sorted(set(parsed)))


def shortlist_trials(
    trials: Sequence[AggressiveTrial],
    strengths: str | Sequence[float],
    limit: int,
) -> list[AggressiveTrial]:
    """Keep deterministic coverage for every configured strength before filling the probe budget."""

    if int(limit) < 1:
        raise ValueError("aggressive KCRN trial shortlist limit must be positive")
    grid = parse_strength_grid(strengths)
    grouped = {strength: [] for strength in grid}
    for trial in trials:
        strength = float(trial.strength)
        if strength in grouped:
            grouped[strength].append(trial)
    selected = []
    used = set()
    for strength in grid:
        for trial in grouped[strength]:
            marker = (trial.target_key, float(trial.strength))
            if marker in used:
                continue
            selected.append(trial)
            used.add(marker)
            break
        if len(selected) >= int(limit):
            return selected[: int(limit)]
    for trial in trials:
        marker = (trial.target_key, float(trial.strength))
        if marker in used:
            continue
        selected.append(trial)
        used.add(marker)
        if len(selected) >= int(limit):
            break
    return selected[: int(limit)]


def merge_trial_stacks(
    baseline: Sequence[AggressiveTrial],
    adaptive: Sequence[AggressiveTrial],
) -> tuple[AggressiveTrial, ...]:
    """Combine fixed baseline and adaptive factors without repeated writers."""

    merged = []
    seen = set()
    for trial in tuple(baseline) + tuple(adaptive):
        key = _target_key(trial.target_key)
        if key in seen:
            raise ValueError(f"aggressive KCRN trial stack repeats target {key}")
        seen.add(key)
        merged.append(trial)
    return tuple(merged)


def _target_key(value) -> TargetKey:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"aggressive KCRN target keys must be (layer, writer), got {value!r}")
    layer, writer = int(value[0]), int(value[1])
    if layer < 0 or writer < 0:
        raise ValueError(f"aggressive KCRN target keys must be non-negative, got {value!r}")
    return layer, writer


def _finite_measurements(measurement: AggressiveEvaluation) -> bool:
    values = (
        math.isfinite(float(value))
        for value in (measurement.refusal_margin, measurement.calibration_kl)
    )
    if not all(values):
        return False
    if measurement.harmful_delivery is None:
        return True
    delivery = float(measurement.harmful_delivery)
    return math.isfinite(delivery) and 0.0 <= delivery <= 1.0


def _relative_update(trial: AggressiveTrial) -> float | None:
    value = trial.certificate.get("relative_update_norm")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _record(
    trial: AggressiveTrial,
    status: str,
    reason: str | None = None,
    measurement: AggressiveEvaluation | None = None,
    cumulative_relative_update: float | None = None,
    improvement: float | None = None,
) -> dict:
    record = {
        "layer": int(trial.target_key[0]),
        "writer_index": int(trial.target_key[1]),
        "strength": float(trial.strength),
        "priority": float(trial.priority),
        "status": status,
    }
    if reason is not None:
        record["reason"] = reason
    if measurement is not None:
        record["refusal_margin"] = float(measurement.refusal_margin)
        record["calibration_kl"] = float(measurement.calibration_kl)
    if cumulative_relative_update is not None:
        record["cumulative_relative_update"] = float(cumulative_relative_update)
    if improvement is not None:
        if measurement is not None and measurement.harmful_delivery is not None:
            record["delivery_improvement"] = float(improvement)
        else:
            record["margin_improvement"] = float(improvement)
    if measurement is not None and measurement.harmful_delivery is not None:
        record["harmful_delivery"] = float(measurement.harmful_delivery)
    return record


def greedy_select(
    candidate_keys: Sequence[TargetKey],
    strengths: str | Sequence[float],
    initial_evaluation: AggressiveEvaluation,
    build_trials: Callable[
        [tuple[AggressiveTrial, ...], tuple[TargetKey, ...]],
        Iterable[AggressiveTrial],
    ],
    evaluate: Callable[[tuple[AggressiveTrial, ...]], AggressiveEvaluation],
    *,
    max_steps: int,
    calibration_kl_budget: float,
    cumulative_relative_update_budget: float,
    min_margin_improvement: float,
) -> AggressiveSelection:
    """Greedily select fixed-weight trials using calibration-only measurements."""

    grid = parse_strength_grid(strengths)
    keys = tuple(sorted({_target_key(key) for key in candidate_keys}))
    if not _finite_measurements(initial_evaluation):
        raise ValueError("initial aggressive KCRN calibration measurements must be finite")
    limits = (
        float(calibration_kl_budget),
        float(cumulative_relative_update_budget),
        float(min_margin_improvement),
    )
    if not math.isfinite(limits[0]) or limits[0] < 0.0:
        raise ValueError("aggressive KCRN calibration KL budget must be finite and non-negative")
    if not math.isfinite(limits[1]) or limits[1] < 0.0:
        raise ValueError(
            "aggressive KCRN cumulative relative-update budget must be finite and non-negative"
        )
    if not math.isfinite(limits[2]) or limits[2] < 0.0:
        raise ValueError(
            "aggressive KCRN minimum margin improvement must be finite and non-negative"
        )
    if int(max_steps) < 0:
        raise ValueError("aggressive KCRN maximum greedy steps must be non-negative")

    accepted: list[AggressiveTrial] = []
    records: list[dict] = []
    remaining = set(keys)
    current = initial_evaluation
    cumulative = 0.0

    for _step in range(int(max_steps)):
        if not remaining:
            break
        trials = list(build_trials(tuple(accepted), tuple(sorted(remaining))))
        feasible = []
        for trial in trials:
            key = _target_key(trial.target_key)
            if key not in remaining or key != trial.target_key:
                records.append(_record(trial, "rejected", "invalid_or_selected_target"))
                continue
            if float(trial.strength) not in grid:
                records.append(_record(trial, "rejected", "strength_not_in_grid"))
                continue
            status = str(trial.certificate.get("status", "")).strip().lower()
            if status.startswith("skipped") or status in {"error", "invalid"}:
                records.append(
                    _record(
                        trial,
                        "rejected",
                        str(trial.certificate.get("skip_reason") or status),
                    )
                )
                continue
            relative_update = _relative_update(trial)
            if relative_update is None:
                records.append(_record(trial, "rejected", "invalid_relative_update"))
                continue
            trial_cumulative = cumulative + relative_update
            if trial_cumulative > limits[1]:
                records.append(
                    _record(
                        trial,
                        "rejected",
                        "cumulative_relative_update",
                        cumulative_relative_update=trial_cumulative,
                    )
                )
                continue
            try:
                measurement = evaluate(tuple(accepted) + (trial,))
            except Exception as exc:
                records.append(
                    _record(
                        trial,
                        "rejected",
                        f"evaluation_error:{type(exc).__name__}",
                        cumulative_relative_update=trial_cumulative,
                    )
                )
                continue
            if not _finite_measurements(measurement):
                records.append(
                    _record(
                        trial,
                        "rejected",
                        "nonfinite_evaluation",
                        measurement=measurement,
                        cumulative_relative_update=trial_cumulative,
                    )
                )
                continue
            if measurement.calibration_kl > limits[0]:
                records.append(
                    _record(
                        trial,
                        "rejected",
                        "calibration_kl",
                        measurement=measurement,
                        cumulative_relative_update=trial_cumulative,
                    )
                )
                continue
            if current.harmful_delivery is not None and measurement.harmful_delivery is not None:
                improvement = float(measurement.harmful_delivery - current.harmful_delivery)
            else:
                improvement = float(current.refusal_margin - measurement.refusal_margin)
            if improvement <= limits[2]:
                records.append(
                    _record(
                        trial,
                        "rejected",
                        "minimum_margin_improvement",
                        measurement=measurement,
                        cumulative_relative_update=trial_cumulative,
                        improvement=improvement,
                    )
                )
                continue
            feasible.append((trial, measurement, trial_cumulative, improvement))

        if not feasible:
            break
        chosen = max(
            feasible,
            key=lambda item: (
                item[3],
                -float(item[1].calibration_kl),
                float(item[0].priority),
                tuple(-value for value in item[0].target_key),
                -float(item[0].strength),
            ),
        )
        trial, measurement, cumulative, improvement = chosen
        accepted.append(trial)
        remaining.remove(trial.target_key)
        current = measurement
        records.append(
            _record(
                trial,
                "accepted",
                measurement=measurement,
                cumulative_relative_update=cumulative,
                improvement=improvement,
            )
        )

    return AggressiveSelection(
        accepted=tuple(accepted),
        records=tuple(records),
        final_evaluation=current,
        cumulative_relative_update=float(cumulative),
    )
