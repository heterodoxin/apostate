# build the refusal subspace and score each layer's causal contribution to refusal.

from __future__ import annotations

from typing import List, Optional, Tuple
import torch


def _orthonormalize(M: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    Q, R = torch.linalg.qr(M)
    keep = torch.abs(torch.diagonal(R)) > tol
    return Q[:, keep]


def _kmeans(X: torch.Tensor, k: int, iters: int = 30, seed: int = 0):
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


def _oriented(d: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    hp = harmful @ d
    lp_mean = (harmless @ d).mean()
    k = max(1, min(hp.numel(), max(1, int(hp.numel() * 0.25))))
    hi = hp.topk(k).values.mean() - lp_mean
    lo = lp_mean - hp.topk(k, largest=False).values.mean()
    if float(lo) > float(hi):
        return -d
    return d


def _sep_score(d: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor) -> float:
    hp = harmful @ d
    lp = harmless @ d
    k = max(1, min(hp.numel(), max(1, int(hp.numel() * 0.25))))
    h_focus = hp.topk(k).values.mean()
    spread = hp.std(unbiased=False) + lp.std(unbiased=False) + 1e-6
    global_gap = torch.abs(hp.mean() - lp.mean())
    tail_gap = torch.abs(h_focus - lp.mean())
    return float(torch.maximum(global_gap, tail_gap) / spread)


def _coverage(d: torch.Tensor, harmful: torch.Tensor, harmless: torch.Tensor) -> float:
    hp = harmful @ d
    lp = harmless @ d
    thr = torch.quantile(lp, 0.90)
    return float((hp > thr).float().mean())


def _residualize_rows(X: torch.Tensor, dirs: List[torch.Tensor]) -> torch.Tensor:
    if not dirs:
        return X
    B = torch.stack(dirs, dim=1)
    return X - (X @ B) @ B.t()


def _try_add(
    dirs: List[torch.Tensor],
    weights: List[float],
    d: torch.Tensor,
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    base_norm: float,
    min_norm_frac: float,
    min_separation: float,
    min_coverage: float = 0.0,
):
    d = d.float()
    if dirs:
        B = torch.stack(dirs, dim=1)
        d = d - B @ (B.t() @ d)
    nrm = float(d.norm())
    if nrm < max(1e-6, base_norm * min_norm_frac):
        return False
    d = d / (nrm + 1e-8)
    d = _oriented(d, harmful, harmless)
    score = _sep_score(d, harmful, harmless)
    if score < min_separation:
        return False
    if _coverage(d, harmful, harmless) < min_coverage:
        return False
    dirs.append(d)
    weights.append(nrm * score)
    return True


def _select_extra(
    dirs: List[torch.Tensor],
    weights: List[float],
    candidates: List[torch.Tensor],
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    cap: int,
    base_norm: float,
    min_norm_frac: float,
    min_separation: float,
    min_coverage: float,
):
    remaining = [c.float() for c in candidates if c is not None and c.numel()]
    while len(dirs) < cap and remaining:
        best = None
        B = torch.stack(dirs, dim=1) if dirs else None
        for idx, cand in enumerate(remaining):
            d = cand - B @ (B.t() @ cand) if B is not None else cand
            nrm = float(d.norm())
            if nrm < max(1e-6, base_norm * min_norm_frac):
                continue
            d = _oriented(d / (nrm + 1e-8), harmful, harmless)
            sep = _sep_score(d, harmful, harmless)
            cov = _coverage(d, harmful, harmless)
            if sep < min_separation or cov < min_coverage:
                continue
            value = sep * (0.50 + cov) * min(1.0, nrm / base_norm)
            if best is None or value > best[0]:
                best = (value, idx, d, nrm, sep)
        if best is None:
            break
        _value, idx, d, nrm, sep = best
        dirs.append(d)
        weights.append(nrm * sep)
        remaining.pop(idx)


def _cluster_means(X: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for c in sorted(torch.unique(labels).tolist()):
        m = labels == c
        if int(m.sum()) >= 2:
            out.append(X[m].mean(0))
    return out


def refusal_subspace(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    rank: int = 1,
    variance_threshold: float = 0.90,
    max_rank: int = 4,
    seed: int = 0,
    orthogonalize: bool = False,
    multi: bool = True,
    clusters: Optional[int] = None,
    min_norm_frac: float = 0.08,
    min_separation: float = 0.05,
    min_coverage: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    harmful = harmful.float()
    harmless = harmless.float()
    mu_harmless = harmless.mean(0)
    mean_diff = harmful.mean(0) - mu_harmless
    if orthogonalize:
        # keep only the part orthogonal to the harmless mean, so ablating it disturbs
        # general behavior less (the part along the harmless mean is "be a normal model").
        gh = mu_harmless / (mu_harmless.norm() + 1e-8)
        mean_diff = mean_diff - (mean_diff @ gh) * gh
    mean_dir = mean_diff / (mean_diff.norm() + 1e-8)
    mean_dir = _oriented(mean_dir, harmful, harmless)

    if rank is not None and rank == 1:
        return mean_dir.unsqueeze(1), mean_diff.norm().reshape(1)

    n_clusters = max_rank if (rank is None or rank <= 0) else rank
    cap = max_rank if (rank is None or rank <= 0) else min(rank, max_rank)
    n_clusters = clusters or max(2, n_clusters * 2)

    dirs = [mean_dir]
    weights = [float(mean_diff.norm())]
    base_norm = max(float(mean_diff.norm()), 1e-6)

    if multi:
        rel_harmful = harmful - mu_harmless
        candidates: List[torch.Tensor] = []

        raw_labels = _kmeans(torch.nn.functional.normalize(rel_harmful, dim=1), n_clusters, seed=seed)
        candidates.extend(m - mu_harmless for m in _cluster_means(harmful, raw_labels))

        rel_harmless = harmless - mu_harmless
        h_centers = _cluster_means(harmful, raw_labels)
        l_labels = _kmeans(torch.nn.functional.normalize(rel_harmless, dim=1), n_clusters, seed=seed + 19)
        l_centers = _cluster_means(harmless, l_labels)
        if h_centers and l_centers:
            Lc = torch.stack(l_centers)
            for hc in h_centers:
                j = torch.cdist(hc.unsqueeze(0), Lc).argmin().item()
                candidates.append(hc - Lc[j])

        residual = _residualize_rows(rel_harmful, dirs)
        labels = _kmeans(torch.nn.functional.normalize(residual, dim=1), n_clusters, seed=seed)
        candidates.extend(_cluster_means(residual, labels))

        norms = residual.norm(dim=1)
        for frac in (0.50, 0.25, 0.12):
            if norms.numel() == 0:
                break
            k = max(1, min(norms.numel(), max(1, int(norms.numel() * frac))))
            idx = torch.topk(norms, k=k).indices
            candidates.append(residual[idx].mean(0))

        centered = residual - residual.mean(0, keepdim=True)
        try:
            _U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            for j in range(min(Vh.shape[0], max(cap * 3, n_clusters))):
                candidates.append(Vh[j] * S[j])
        except RuntimeError:
            pass

        _select_extra(dirs, weights, candidates, harmful, harmless, cap, base_norm,
                      min_norm_frac, min_separation, min_coverage)
    else:
        labels = _kmeans(harmful, n_clusters, seed=seed)
        for c in sorted(torch.unique(labels).tolist()):
            members = harmful[labels == c]
            if members.shape[0] < 2:
                continue
            _try_add(dirs, weights, members.mean(0) - mu_harmless, harmful, harmless,
                     base_norm, 0.15, min_separation, min_coverage)
            if len(dirs) >= cap:
                break

    basis = _orthonormalize(torch.stack(dirs, dim=1))[:, :cap]
    w = torch.tensor(weights)[: basis.shape[1]]
    return basis, w


def preservation_subspace(activations: torch.Tensor, rank: int = 4) -> torch.Tensor:
    acts = activations.float()
    acts = acts - acts.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(acts, full_matrices=False)
    V = Vh.t()
    return _orthonormalize(V[:, : max(1, rank)])


def gram_schmidt_remove(
    refusal: torch.Tensor,
    preserve: Optional[torch.Tensor],
) -> torch.Tensor:
    if preserve is None or preserve.numel() == 0:
        return _orthonormalize(refusal)
    P = _orthonormalize(preserve)
    R = refusal - P @ (P.t() @ refusal)
    R = _orthonormalize(R)
    if R.numel() == 0:
        return _orthonormalize(refusal)
    return R


def separation(harmful: torch.Tensor, harmless: torch.Tensor) -> float:
    return float((harmful.float().mean(0) - harmless.float().mean(0)).norm().item())


def augment_subspace(existing: torch.Tensor, new_dirs: torch.Tensor, max_rank: int) -> torch.Tensor:
    if existing is None or existing.numel() == 0:
        return _orthonormalize(new_dirs)[:, :max_rank]
    extra = new_dirs - existing @ (existing.t() @ new_dirs)
    extra = _orthonormalize(extra)
    if extra.numel() == 0:
        return existing
    merged = torch.cat([existing, extra], dim=1)
    return merged[:, :max_rank]


# per-layer strength prior: ablate each layer alone, turn the refusal drop into an alpha.

def causal_layer_scores(
    bundle,
    controller,
    eval_instructions: List[str],
    batch_size: int = 16,
    floor: float = 0.25,
    temperature: float = 1.0,
) -> List[float]:
    from .evaluate import refusal_logit_margin  # local import avoids an import cycle

    controller.set_uniform_alpha(0.0)
    with controller.active():
        base = refusal_logit_margin(bundle, eval_instructions, batch_size)

    drops: List[float] = []
    for L in range(bundle.num_layers):
        controller.isolate_layer(L)
        with controller.active():
            m = refusal_logit_margin(bundle, eval_instructions, batch_size)
        drops.append(max(0.0, base - m))

    t = torch.tensor(drops)
    if float(t.max()) <= 1e-6:
        return [1.0] * bundle.num_layers

    t = t / t.max()
    if temperature != 1.0:
        t = t ** (1.0 / max(1e-3, temperature))
    alphas = floor + (1.0 - floor) * t
    return [float(x) for x in alphas]
