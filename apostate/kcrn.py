"""Fixed-weight Key-Conditional Refusal Nulling utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .model import _is_conv1d


class KCRNDegeneracyError(ValueError):
    """Raised when projected KCRN cannot separate harmful keys from benign keys."""

    def __init__(self, message: str, diagnostics: Optional[dict] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(slots=True)
class KCRNUpdate:
    """Store a low-rank weight update and its solver certificate."""

    left: torch.Tensor
    right: torch.Tensor
    diagnostics: dict

    @property
    def delta(self) -> torch.Tensor:
        return self.left @ self.right


def _matrix(value, name: str, device=None) -> torch.Tensor:
    source_dtype = getattr(value, "dtype", None)
    dtype = source_dtype if source_dtype in (torch.float32, torch.float64) else torch.float32
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {tuple(tensor.shape)}")
    return tensor


def orthonormal_basis(
    columns: torch.Tensor,
    rank: Optional[int] = None,
    tolerance: float = 1e-7,
    explained_variance: float = 1.0,
) -> torch.Tensor:
    """Return a numerically stable orthonormal basis for the supplied column span."""

    columns = _matrix(columns, "columns")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not math.isfinite(float(explained_variance)) or not 0 < explained_variance <= 1:
        raise ValueError("explained_variance must be in (0, 1]")
    if columns.shape[1] == 0:
        return columns.new_empty((columns.shape[0], 0))
    u, singular, _vh = torch.linalg.svd(columns, full_matrices=False)
    if singular.numel() == 0 or float(singular.max()) <= 0:
        return columns.new_empty((columns.shape[0], 0))
    threshold = float(singular.max()) * float(tolerance)
    nonzero = singular > threshold
    available = int(nonzero.sum().item())
    if available == 0:
        return columns.new_empty((columns.shape[0], 0))
    keep = available if rank is None else min(available, max(0, int(rank)))
    if keep == 0:
        return columns.new_empty((columns.shape[0], 0))
    if explained_variance < 1.0:
        variance = singular[:available].square()
        cumulative = torch.cumsum(variance, dim=0) / variance.sum().clamp_min(torch.finfo(variance.dtype).eps)
        explained_rank = int(torch.searchsorted(cumulative, float(explained_variance)).item()) + 1
        keep = min(keep, explained_rank)
    return u[:, :keep].contiguous()


def _columns(
    value: torch.Tensor,
    name: str,
    tolerance: float = 1e-7,
    allow_empty: bool = False,
) -> torch.Tensor:
    if value.ndim != 2 or (value.shape[1] == 0 and not allow_empty):
        suffix = "" if allow_empty else " at least one column"
        raise ValueError(f"{name} must have{suffix}")
    basis = orthonormal_basis(value, tolerance=tolerance)
    if basis.shape[1] == 0 and not allow_empty:
        raise ValueError(f"{name} has no nonzero directions")
    return basis


def key_basis(
    keys: torch.Tensor,
    rank: int,
    mode: str = "legacy",
    tolerance: float = 1e-7,
    explained_variance: float = 1.0,
) -> torch.Tensor:
    """Build a bounded orthonormal input-key basis using the requested mode."""

    keys = _matrix(keys, "keys")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if keys.shape[0] == 0:
        raise ValueError("keys must contain at least one sample")
    mode = str(mode).strip().lower()
    if mode not in {"legacy", "raw", "pca"}:
        raise ValueError(f"unknown key-basis mode: {mode!r}")
    target = min(int(rank), int(keys.shape[0]), int(keys.shape[1]))
    if mode == "raw":
        basis = orthonormal_basis(
            keys.T,
            rank=target,
            tolerance=tolerance,
            explained_variance=explained_variance,
        )
        if basis.shape[1] == 0:
            raise ValueError("keys has no nonzero raw directions")
        return basis.contiguous()
    if mode == "pca":
        centered = keys - keys.mean(dim=0, keepdim=True)
        centered_t = centered.T
        q = min(target, int(min(centered_t.shape)))
        try:
            u, singular, _ = torch.pca_lowrank(
                centered_t,
                q=q,
                center=False,
                niter=2,
            )
        except RuntimeError:
            u, singular, _ = torch.linalg.svd(centered_t, full_matrices=False)
        threshold = float(singular.max()) * float(tolerance) if singular.numel() else 0.0
        available = int((singular > threshold).sum().item())
        keep = min(target, available)
        if keep and explained_variance < 1.0:
            total_variance = centered_t.square().sum().clamp_min(
                torch.finfo(centered_t.dtype).eps
            )
            cumulative = torch.cumsum(singular[:available].square(), dim=0) / total_variance
            explained_rank = int(
                torch.searchsorted(cumulative, float(explained_variance)).item()
            ) + 1
            keep = min(keep, explained_rank)
        basis = u[:, :keep].contiguous() if keep else centered_t.new_empty((centered_t.shape[0], 0))
        if basis.shape[1] == 0:
            basis = orthonormal_basis(keys.T, rank=target, tolerance=tolerance)
        if basis.shape[1] == 0:
            raise ValueError("keys has no nonzero PCA directions")
        return basis.contiguous()
    mean = keys.mean(dim=0)
    candidates = [mean] if float(mean.norm()) > 1e-8 else [keys[0]]
    if target > 1:
        centered = keys - mean
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        candidates.extend(vh[: target - 1])
    basis = orthonormal_basis(
        torch.stack(candidates, dim=0).T,
        rank=target,
        tolerance=tolerance,
    )
    if basis.shape[1] < target:
        basis = orthonormal_basis(
            torch.cat((basis, orthonormal_basis(keys.T, tolerance=tolerance)), dim=1),
            rank=target,
            tolerance=tolerance,
        )
    return basis[:, : min(target, basis.shape[1])].contiguous()


def _inverse_metric_times(metric: Optional[torch.Tensor], values: torch.Tensor) -> torch.Tensor:
    if metric is None:
        return values
    metric = torch.as_tensor(metric, device=values.device, dtype=values.dtype)
    if metric.ndim == 1:
        if metric.shape[0] != values.shape[0] or torch.any(metric <= 0):
            raise ValueError("diagonal metric must be positive and match key width")
        return values / metric[:, None]
    if metric.ndim == 2:
        if metric.shape != (values.shape[0], values.shape[0]):
            raise ValueError("full metric must be square with key width")
        return torch.linalg.solve(metric, values)
    raise ValueError("metric must be None, a positive diagonal vector, or a square matrix")


def _solve(core: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(core, rhs)
    except RuntimeError:
        return torch.linalg.pinv(core) @ rhs


def _factor_frobenius(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_gram = left.T @ left
    right_gram = right @ right.T
    return (left_gram * right_gram.T).sum().clamp_min(0).sqrt()


def _certificate(
    W,
    R,
    Kh,
    Kb,
    left,
    right,
    target=None,
    projected_energy=None,
    condition_matrix=None,
    include_spectrum: bool = True,
) -> dict:
    delta_kh = left @ (right @ Kh)
    base_refusal = R.T @ (W @ Kh)
    harmful = base_refusal + R.T @ delta_kh
    benign = (
        left @ (right @ Kb)
        if Kb is not None and Kb.shape[1]
        else torch.zeros((W.shape[0], 0), dtype=W.dtype, device=W.device)
    )
    delta_norm = _factor_frobenius(left, right)
    target_norm = torch.linalg.norm(target) if target is not None else torch.linalg.norm(delta_kh)
    fit_error = torch.linalg.norm(delta_kh - target) / target_norm.clamp_min(1e-8) if target is not None else torch.zeros((), device=W.device)
    benign_leakage = torch.linalg.norm(benign) / delta_norm.clamp_min(1e-8)
    refusal_residual = torch.linalg.norm(harmful) / torch.linalg.norm(base_refusal).clamp_min(1e-8)
    left_gram = left.T @ left
    right_gram = right @ right.T
    del left_gram, right_gram
    condition_source = condition_matrix
    if condition_source is None:
        condition_source = right @ Kh if Kb is None or Kb.shape[1] == 0 else torch.cat((Kh, Kb), dim=1)
    try:
        condition = float(torch.linalg.cond(condition_source).item())
    except RuntimeError:
        condition = math.inf
    if include_spectrum:
        try:
            eigenvalues = torch.linalg.eigvalsh(condition_source).detach().float().cpu().tolist()
        except RuntimeError:
            eigenvalues = []
    else:
        eigenvalues = []
    return {
        "harmful_residual": float(torch.linalg.norm(harmful).item()),
        "benign_change": float(torch.linalg.norm(benign).item()),
        "benign_leakage": float(benign_leakage.item()),
        "harmful_fit_error": float(fit_error.item()),
        "refusal_residual": float(refusal_residual.item()),
        "delta_fro": float(delta_norm.item()),
        "relative_update_norm": float((delta_norm / torch.linalg.norm(W).clamp_min(1e-8)).item()),
        "condition": condition,
        "regularized_eigenvalues": eigenvalues,
        "factor_rank": int(left.shape[1]),
        "projected_harmful_energy": None if projected_energy is None else float(projected_energy),
    }


def key_conditional_nulling(
    W: torch.Tensor,
    R: torch.Tensor,
    harmful_keys: torch.Tensor,
    benign_keys: Optional[torch.Tensor] = None,
    metric: Optional[torch.Tensor] = None,
    strength: float = 1.0,
    ridge: float = 0.0,
    harmful_target: Optional[torch.Tensor] = None,
) -> KCRNUpdate:
    """Solve a minimum-metric-norm update with harmful and benign key constraints."""

    W = _matrix(W, "W")
    R = _columns(_matrix(R, "R", W.device), "R")
    Kh = _matrix(harmful_keys, "harmful_keys", W.device)
    Kb = None if benign_keys is None else _matrix(benign_keys, "benign_keys", W.device)
    if W.shape[0] != R.shape[0] or W.shape[1] != Kh.shape[0]:
        raise ValueError("W, R, and harmful_keys have incompatible widths")
    if Kb is not None and Kb.shape[0] != W.shape[1]:
        raise ValueError("W input width must match benign key width")
    if not math.isfinite(float(strength)):
        raise ValueError("strength must be finite")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    if Kb is not None and Kb.shape[1] == 0:
        Kb = None
    A = Kh if Kb is None else torch.cat((Kh, Kb), dim=1)
    inverse_metric_A = _inverse_metric_times(metric, A)
    core = A.T @ inverse_metric_A
    if ridge:
        scale = float(torch.diagonal(core).mean().item())
        core = core + float(ridge) * max(scale, 1e-8) * torch.eye(
            core.shape[0], dtype=core.dtype, device=core.device
        )
    right = _solve(core, inverse_metric_A.T)
    if harmful_target is None:
        target = -float(strength) * R @ (R.T @ W @ Kh)
    else:
        target = _matrix(harmful_target, "harmful_target", W.device)
        if target.shape != (W.shape[0], Kh.shape[1]):
            raise ValueError("harmful_target shape must match W output and harmful key rank")
        target = float(strength) * target
    left = target if Kb is None else torch.cat(
        (target, torch.zeros(W.shape[0], Kb.shape[1], device=W.device)), dim=1
    )
    diagnostics = _certificate(
        W,
        R,
        Kh,
        Kb,
        left,
        right,
        target=target,
        condition_matrix=core,
    )
    diagnostics.update({"strength": float(strength), "ridge": float(ridge), "solver": "normal"})
    return KCRNUpdate(left=left, right=right, diagnostics=diagnostics)


def key_conditional_nulling_projected(
    W: torch.Tensor,
    R: torch.Tensor,
    harmful_keys: torch.Tensor,
    preserve_basis: torch.Tensor,
    strength: float = 1.0,
    ridge: float = 0.0,
    min_projected_energy: float = 1e-6,
    max_condition: float = math.inf,
    max_relative_update: float = math.inf,
    svd_tolerance: float = 1e-7,
    explained_variance: float = 1.0,
    preserve_basis_orthonormal: bool = False,
    diagnostics_spectrum: bool = True,
) -> KCRNUpdate:
    """Solve KCRN in the numerically stable complement of the benign key span."""

    W = _matrix(W, "W")
    R = _columns(_matrix(R, "R", W.device), "R")
    Kh = _matrix(harmful_keys, "harmful_keys", W.device)
    Kb_raw = _matrix(preserve_basis, "preserve_basis", W.device)
    if W.shape[0] != R.shape[0] or W.shape[1] != Kh.shape[0]:
        raise ValueError("W, R, and key bases have incompatible widths")
    if Kb_raw.shape[0] != W.shape[1] or not math.isfinite(float(strength)) or ridge < 0:
        raise ValueError("strength must be finite and ridge must be non-negative")
    if not 0 <= float(min_projected_energy) or not math.isfinite(float(min_projected_energy)):
        raise ValueError("min_projected_energy must be finite and non-negative")
    if max_condition <= 0 or max_relative_update <= 0:
        raise ValueError("projected KCRN safeguards must be positive")

    if preserve_basis_orthonormal:
        if Kb_raw.shape[1]:
            gram = Kb_raw.T @ Kb_raw
            identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            if not torch.allclose(gram, identity, atol=max(1e-5, float(svd_tolerance) * 10), rtol=1e-4):
                raise ValueError("preserve_basis_orthonormal=True requires orthonormal columns")
        Kb = Kb_raw
    else:
        Kb = orthonormal_basis(
            Kb_raw,
            tolerance=svd_tolerance,
            explained_variance=explained_variance,
        )

    if Kb.shape[1]:
        projected = Kh - Kb @ (Kb.T @ Kh)
        benign_orthogonality = torch.linalg.norm(Kb.T @ projected) / torch.linalg.norm(projected).clamp_min(1e-8)
    else:
        projected = Kh
        benign_orthogonality = torch.zeros((), dtype=Kh.dtype, device=Kh.device)
    harmful_norm = torch.linalg.norm(Kh).clamp_min(1e-8)
    projected_energy = float((torch.linalg.norm(projected).square() / harmful_norm.square()).item())
    if projected_energy < float(min_projected_energy):
        raise KCRNDegeneracyError(
            f"projected harmful energy {projected_energy:.6g} is below "
            f"minimum {float(min_projected_energy):.6g}",
            {
                "projected_harmful_energy": projected_energy,
                "preserve_rank": int(Kb.shape[1]),
            },
        )
    core = Kh.T @ projected
    core = (core + core.T) * 0.5
    if ridge:
        scale = float(torch.diagonal(core).mean().item())
        core = core + float(ridge) * max(scale, 1e-8) * torch.eye(
            core.shape[0], dtype=core.dtype, device=core.device
        )
    try:
        condition = float(torch.linalg.cond(core).item())
    except RuntimeError:
        condition = math.inf
    if not math.isfinite(condition) or condition > float(max_condition):
        raise KCRNDegeneracyError(
            f"projected KCRN system condition {condition:.6g} exceeds "
            f"maximum {float(max_condition):.6g}",
            {
                "projected_harmful_energy": projected_energy,
                "condition": condition,
                "regularized_eigenvalues": (
                    torch.linalg.eigvalsh(core).detach().float().cpu().tolist()
                    if diagnostics_spectrum else []
                ),
                "preserve_rank": int(Kb.shape[1]),
            },
        )
    right = torch.linalg.solve(core, projected.T)
    target = -float(strength) * R @ (R.T @ W @ Kh)
    left = target
    diagnostics = _certificate(
        W,
        R,
        Kh,
        Kb,
        left,
        right,
        target=target,
        projected_energy=projected_energy,
        condition_matrix=core,
        include_spectrum=diagnostics_spectrum,
    )
    diagnostics.update({
        "strength": float(strength),
        "ridge": float(ridge),
        "solver": "projected",
        "preserve_rank": int(Kb.shape[1]),
        "benign_basis_rank": int(Kb.shape[1]),
        "projected_benign_orthogonality": float(benign_orthogonality.item()),
    })
    if diagnostics["relative_update_norm"] > float(max_relative_update):
        raise KCRNDegeneracyError(
            f"projected KCRN relative update {diagnostics['relative_update_norm']:.6g} exceeds "
            f"maximum {float(max_relative_update):.6g}",
            diagnostics,
        )
    return KCRNUpdate(left=left, right=right, diagnostics=diagnostics)


def writer_matrix(module: torch.nn.Module) -> tuple[torch.Tensor, bool]:
    """Return a residual writer as an output-by-input matrix."""

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        packed = getattr(module, "down_proj", None)
        if isinstance(packed, torch.Tensor) and packed.ndim == 3:
            return packed.detach().float().mean(dim=0), False
        raise TypeError("KCRN supports 2-D Linear, Conv1D, and packed expert writers")
    if weight.ndim != 2:
        raise TypeError("KCRN supports 2-D Linear and Conv1D writers")
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is not None:
        from bitsandbytes.functional import dequantize_4bit

        matrix = dequantize_4bit(weight, quant_state).float()
    else:
        matrix = weight.detach().float()
    if _is_conv1d(module):
        return matrix.T.contiguous(), True
    return matrix, False


def _input_tensor(inputs: tuple) -> torch.Tensor:
    tensor = next((item for item in inputs if isinstance(item, torch.Tensor)), None)
    if tensor is None or tensor.ndim not in (2, 3):
        shape = None if tensor is None else tuple(tensor.shape)
        raise ValueError(f"writer input must be rank 2 or 3, got {shape}")
    return tensor


def _last_rows(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if mask is None:
        return tensor[:, -1, :]
    positions = torch.arange(mask.shape[1], device=tensor.device).expand_as(mask)
    last = (positions * mask.to(positions.dtype)).argmax(dim=1)
    return tensor[torch.arange(tensor.shape[0], device=tensor.device), last]


@torch.inference_mode()
def collect_writer_inputs(
    bundle,
    instructions: list[str],
    layers: Optional[list[int]] = None,
    batch_size: int = 16,
    all_positions: bool = False,
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> dict[int, dict[int, torch.Tensor]]:
    """Capture residual-writer input keys and remove every temporary hook."""

    from .activations import _prompt_batches
    from .data import format_chat

    device = next(bundle.model.parameters()).device
    prompts = instructions if getattr(bundle, "_kcrn_preformatted", False) else format_chat(
        bundle.tokenizer, instructions
    )
    batches = _prompt_batches(bundle, bundle.tokenizer, prompts, batch_size, device)
    selected = set(range(bundle.num_layers) if layers is None else layers)
    if max_samples is not None and int(max_samples) < 1:
        raise ValueError("max_samples must be positive when provided")
    captured: dict[int, dict[int, list[torch.Tensor]]] = {
        layer: {} for layer in selected if 0 <= layer < bundle.num_layers
    }
    reservoirs: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    generators: dict[tuple[int, int], torch.Generator] = {}
    handles = []
    current_mask: list[Optional[torch.Tensor]] = [None]

    def make_hook(layer_idx: int, writer_idx: int):
        def hook(_module, inputs):
            try:
                values = _input_tensor(inputs)
                if all_positions and values.ndim == 3:
                    keys = values.reshape(-1, values.shape[-1])
                    mask = current_mask[0]
                    if mask is not None and mask.shape == values.shape[:2]:
                        keys = keys[mask.reshape(-1).bool()]
                else:
                    keys = _last_rows(values, current_mask[0])
            except ValueError:
                return
            keys = keys.detach().float().cpu()
            if max_samples is None:
                captured[layer_idx].setdefault(writer_idx, []).append(keys)
                return
            key = (layer_idx, writer_idx)
            generator = generators.get(key)
            if generator is None:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) + layer_idx * 1009 + writer_idx * 9176)
                generators[key] = generator
            scores = torch.rand(keys.shape[0], generator=generator)
            old = reservoirs.get(key)
            if old is None:
                combined_keys, combined_scores = keys, scores
            else:
                combined_keys = torch.cat((old[0], keys), dim=0)
                combined_scores = torch.cat((old[1], scores), dim=0)
            if combined_keys.shape[0] > int(max_samples):
                keep = torch.topk(
                    combined_scores,
                    k=int(max_samples),
                    largest=True,
                    sorted=False,
                ).indices
                combined_keys = combined_keys[keep]
                combined_scores = combined_scores[keep]
            reservoirs[key] = (combined_keys, combined_scores)

        return hook

    try:
        for layer_idx in sorted(selected):
            for writer_idx, module in enumerate(bundle.layer_writers(bundle.layers()[layer_idx])):
                if isinstance(module, torch.nn.Module):
                    handles.append(module.register_forward_pre_hook(make_hook(layer_idx, writer_idx)))
        for batch in batches:
            current_mask[0] = batch.get("attention_mask")
            bundle.model(**batch, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if max_samples is None:
        return {
            layer: {
                writer: torch.cat(chunks, dim=0)
                for writer, chunks in by_writer.items()
                if chunks
            }
            for layer, by_writer in captured.items()
        }
    return {
        layer: {
            writer: reservoirs[(layer, writer)][0]
            for writer in sorted(writer for current_layer, writer in reservoirs if current_layer == layer)
        }
        for layer in captured
        if any(current_layer == layer for current_layer, _writer in reservoirs)
    }


def select_writer_targets(
    bundle,
    residual_h: torch.Tensor,
    residual_b: torch.Tensor,
    pilot_h: dict[int, dict[int, torch.Tensor]],
    pilot_b: dict[int, dict[int, torch.Tensor]],
    max_targets: int = 3,
    writer_indices: Optional[set[int]] = None,
    mlp_only: bool = False,
) -> list[dict]:
    """Rank bakeable writers by their conditional refusal output."""

    residual_h = torch.as_tensor(residual_h).float()
    residual_b = torch.as_tensor(residual_b).float()
    if (
        residual_h.ndim != 3
        or residual_b.ndim != 3
        or residual_b.shape[0] != residual_h.shape[0]
        or residual_b.shape[2] != residual_h.shape[2]
    ):
        raise ValueError("residual activations must have shape [layers, samples, hidden] with matching layer and hidden widths")
    candidates = []
    for layer_idx in range(min(bundle.num_layers, residual_h.shape[0])):
        refusal = residual_h[layer_idx].mean(0) - residual_b[layer_idx].mean(0)
        norm = float(refusal.norm().item())
        if norm <= 1e-8:
            continue
        direction = refusal / norm
        writers = bundle.layer_writers(bundle.layers()[layer_idx])
        for writer_idx, keys_h in pilot_h.get(layer_idx, {}).items():
            if writer_indices is not None and writer_idx not in writer_indices:
                continue
            keys_b = pilot_b.get(layer_idx, {}).get(writer_idx)
            if keys_b is None or writer_idx >= len(writers):
                continue
            label = next(
                (name for name, child in bundle.layers()[layer_idx].named_modules() if child is writers[writer_idx]),
                type(writers[writer_idx]).__name__,
            )
            if mlp_only and not any(
                token in label.lower()
                for token in ("down_proj", "c_proj", "w2", "fc_out", "dense_4h_to_h", "wo")
            ):
                continue
            try:
                matrix, _transpose = writer_matrix(writers[writer_idx])
            except (TypeError, ValueError, RuntimeError):
                continue
            key_delta = keys_h.float().mean(0) - keys_b.float().mean(0)
            if matrix.shape[1] != key_delta.shape[0] or matrix.shape[0] != direction.shape[0]:
                continue
            score = abs(float((direction.to(matrix.device) @ matrix @ key_delta.to(matrix.device)).item()))
            candidates.append({
                "layer": int(layer_idx),
                "writer_index": int(writer_idx),
                "writer": label,
                "selection_score": score,
                "direction_norm": norm,
            })
    candidates.sort(key=lambda item: (item["selection_score"], item["layer"], -item["writer_index"]), reverse=True)
    return candidates[: max(0, int(max_targets))]
