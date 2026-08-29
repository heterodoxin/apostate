"""Conditional directional abliteration (diode): a self-gated half-space edit baked into standard weights.

Each layer gets one repurposed MLP neuron that subtracts the residual refusal direction only when the
refusal detector fires above a benign-calibrated threshold, so benign inputs are left untouched.
"""
from __future__ import annotations

import torch

from .config import ApostateConfig
from .model import load_model
from .data import resolve_prompts, format_chat
from .activations import collect_activations
from . import ticv


def _benign_orthogonal(v, samples, rank=64):
    basis = torch.linalg.svd(samples.float() - samples.float().mean(0), full_matrices=False).Vh[:rank]
    w = v - basis.T @ (basis @ v)
    return w / w.norm().clamp_min(1e-6)


def _capture_premlp(model, tok, prompts, batch_size):
    layers = ticv._decoder(model).layers
    nl = len(layers)
    device = next(model.parameters()).device
    buf = [None] * nl
    def make_hook(li):
        def hook(_m, _i, o):
            buf[li] = (o[0] if isinstance(o, (tuple, list)) else o).detach().float()
        return hook
    handles = []
    for li, layer in enumerate(layers):
        try:
            handles.append(ticv._premlp_norm(layer).register_forward_hook(make_hook(li)))
        except Exception:
            pass
    last = [[] for _ in range(nl)]
    allpos = [[] for _ in range(nl)]
    for start in range(0, len(prompts), batch_size):
        enc = tok(format_chat(tok, prompts[start:start + batch_size]), return_tensors="pt",
                  padding=True, truncation=True, max_length=256, add_special_tokens=False).to(device)
        with torch.no_grad():
            model(**enc)
        mask = enc["attention_mask"].bool()
        idx = enc["attention_mask"].sum(1) - 1
        for li in range(nl):
            if buf[li] is None:
                continue
            last[li].append(buf[li][torch.arange(buf[li].shape[0]), idx].cpu())
            allpos[li].append(buf[li][mask].cpu())
    for h in handles:
        h.remove()
    return ([torch.cat(a, 0) if a else None for a in last],
            [torch.cat(a, 0) if a else None for a in allpos])


def fit_and_bake(cfg: ApostateConfig, bundle=None) -> dict:
    """Fit the per-layer detectors and thresholds, write the gated neurons, and save the checkpoint."""
    cfg.with_defaults()
    cfg.load_in_4bit = False
    own = bundle is None
    base = load_model(cfg) if own else bundle
    nl = base.num_layers
    tok = base.tokenizer
    fit_harmful = resolve_prompts(cfg.harmful_path, cfg.diode_fit_n, cfg.seed)
    fit_benign = resolve_prompts(cfg.harmless_path, cfg.diode_fit_n, cfg.seed)
    rmul = float(getattr(base.model.config, "residual_multiplier", 1.0) or 1.0)
    lo, hi = cfg.diode_band
    band = set(range(int(lo * nl), int(hi * nl)))

    harmful_res = collect_activations(base, fit_harmful, cfg.batch_size)
    benign_res = collect_activations(base, fit_benign, cfg.batch_size)
    harmful_pre, _ = _capture_premlp(base.model, tok, fit_harmful, cfg.batch_size)
    benign_pre_last, benign_pre_all = _capture_premlp(base.model, tok, fit_benign, cfg.batch_size)

    detector = [None] * nl
    actuator = [None] * nl
    theta = [None] * nl
    for l in range(nl):
        if harmful_pre[l] is None or benign_pre_last[l] is None:
            continue
        d = harmful_pre[l].mean(0) - benign_pre_last[l].mean(0)
        detector[l] = _benign_orthogonal(d, benign_pre_last[l])
        z = harmful_res[l].float().mean(0) - benign_res[l].float().mean(0)
        actuator[l] = z / z.norm().clamp_min(1e-6)
        theta[l] = float(torch.quantile(benign_pre_all[l] @ detector[l], 1.0 - cfg.diode_target))

    samples = ticv._calib_xmlp(base.model, tok, nl)
    layers = ticv._decoder(base.model).layers
    device = next(base.model.parameters()).device
    written = 0
    with torch.no_grad():
        for li, layer in enumerate(layers):
            if li not in band or detector[li] is None:
                continue
            try:
                mlp = ticv._gated_mlp(layer)
            except Exception:
                continue
            X = samples[li]
            mean, std = X.mean(0), X.std(0)
            cd = int((mean.abs().square() / std.clamp_min(1e-6)).argmax())
            m = mean[cd].item()
            gate = (cfg.diode_kappa * detector[li]).clone()
            gate[cd] = gate[cd] - cfg.diode_kappa * theta[li] / m
            j = mlp.gate_proj.weight.shape[0] - 1
            mlp.gate_proj.weight[j].copy_(gate.to(device).to(mlp.gate_proj.weight.dtype))
            mlp.up_proj.weight[j].zero_()
            mlp.up_proj.weight[j, cd] = 1.0
            down = -(cfg.diode_strength / (cfg.diode_kappa * m * rmul)) * actuator[li]
            mlp.down_proj.weight[:, j].copy_(down.to(device).to(mlp.down_proj.weight.dtype))
            written += 1

    base.model.save_pretrained(cfg.output_dir, safe_serialization=True)
    tok.save_pretrained(cfg.output_dir)
    report = {
        "method": "diode", "edited_layers": written, "num_layers": nl,
        "band": [min(band), max(band)] if band else [], "residual_multiplier": rmul,
        "strength": cfg.diode_strength, "kappa": cfg.diode_kappa, "benign_fire_target": cfg.diode_target,
        "runtime_hooks": False, "deployment": "standard weights",
    }
    if own:
        import gc
        base.model = None
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def _selftest():
    torch.manual_seed(0)
    hidden, n = 512, 400
    basis = torch.linalg.qr(torch.randn(hidden, 8)).Q
    refusal = torch.linalg.qr(torch.randn(hidden, 9)).Q[:, 8]
    benign = torch.randn(n, 8) @ basis.T + 0.01 * torch.randn(n, hidden)
    harmful = torch.randn(n, 8) @ basis.T + 3.0 * torch.randn(n, 1).abs() * refusal + 0.01 * torch.randn(n, hidden)
    d = _benign_orthogonal(harmful.mean(0) - benign.mean(0), benign)
    sep = float((harmful @ d).mean() - (benign @ d).mean())
    assert sep > 0.5, f"detector must separate harmful from benign (got {sep:.2f})"
    assert float((benign @ d).abs().mean()) < float((harmful @ d).abs().mean()), "detector must be benign-quiet"
    print(f"selftest ok: harmful-vs-benign separation={sep:.2f}")


if __name__ == "__main__":
    _selftest()
