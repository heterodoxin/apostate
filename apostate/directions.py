"""subspace geometry"""

from __future__ import annotations

from typing import Optional, Tuple
import torch


def _orthonormalize(M: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """orthonormal basis"""
    Q, R = torch.linalg.qr(M)
    keep = torch.abs(torch.diagonal(R)) > tol
    return Q[:, keep]


def _kmeans(X: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
    """kmeans"""
    n = X.shape[0]
    k = max(1, min(k, n))
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(n, generator=g)[:k]].clone()
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        labels = torch.cdist(X, C).argmin(dim=1)
        newC = C.clone()
        for c in range(k):
            m = labels == c
            if m.any():
                newC[c] = X[m].mean(0)
        if torch.allclose(newC, C):
            break
        C = newC
    return labels


def refusal_subspace(
    harmful: torch.Tensor,      # harmful acts
    harmless: torch.Tensor,     # harmless acts
    rank: int = 1,
    variance_threshold: float = 0.90,
    max_rank: int = 4,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """refusal basis"""
    harmful = harmful.float()
    harmless = harmless.float()
    mu_harmless = harmless.mean(0)
    mean_diff = harmful.mean(0) - mu_harmless
    mean_dir = mean_diff / (mean_diff.norm() + 1e-8)

    if rank is not None and rank == 1:
        return mean_dir.unsqueeze(1), mean_diff.norm().reshape(1)

    n_clusters = max_rank if (rank is None or rank <= 0) else rank
    cap = max_rank if (rank is None or rank <= 0) else min(rank, max_rank)

    labels = _kmeans(harmful, n_clusters, seed=seed)
    dirs = [mean_dir]
    weights = [float(mean_diff.norm())]
    min_new = 0.15 * float(mean_diff.norm())   # mode floor
    for c in sorted(torch.unique(labels).tolist()):
        members = harmful[labels == c]
        if members.shape[0] < 2:
            continue
        d = members.mean(0) - mu_harmless
        B = torch.stack(dirs, dim=1)
        d_orth = d - B @ (B.t() @ d)           # novel part
        nrm = float(d_orth.norm())
        if nrm < min_new:
            continue
        dirs.append(d_orth / nrm)
        weights.append(nrm)
        if len(dirs) >= cap:
            break

    basis = _orthonormalize(torch.stack(dirs, dim=1))[:, :cap]
    w = torch.tensor(weights)[: basis.shape[1]]
    return basis, w


def preservation_subspace(activations: torch.Tensor, rank: int = 4) -> torch.Tensor:
    """preserve basis"""
    acts = activations.float()
    acts = acts - acts.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(acts, full_matrices=False)
    V = Vh.t()
    return _orthonormalize(V[:, : max(1, rank)])


def gram_schmidt_remove(
    refusal: torch.Tensor,            # refusal basis
    preserve: Optional[torch.Tensor], # preserve basis
) -> torch.Tensor:
    """remove preserve"""
    if preserve is None or preserve.numel() == 0:
        return _orthonormalize(refusal)
    P = _orthonormalize(preserve)
    R = refusal - P @ (P.t() @ refusal)
    R = _orthonormalize(R)
    if R.numel() == 0:
        return _orthonormalize(refusal)
    return R


def separation(harmful: torch.Tensor, harmless: torch.Tensor) -> float:
    """separation"""
    return float((harmful.float().mean(0) - harmless.float().mean(0)).norm().item())


def augment_subspace(existing: torch.Tensor, new_dirs: torch.Tensor, max_rank: int) -> torch.Tensor:
    """merge basis"""
    if existing is None or existing.numel() == 0:
        return _orthonormalize(new_dirs)[:, :max_rank]
    extra = new_dirs - existing @ (existing.t() @ new_dirs)
    extra = _orthonormalize(extra)
    if extra.numel() == 0:
        return existing
    merged = torch.cat([existing, extra], dim=1)
    return merged[:, :max_rank]
