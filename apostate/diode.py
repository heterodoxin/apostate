"""Conditional directional abliteration (diode): a self-gated half-space edit baked into standard weights.

Each layer gets one repurposed MLP neuron that subtracts the residual refusal direction only when the
refusal detector fires above a benign-calibrated threshold, so benign inputs are left untouched.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from .config import ApostateConfig
from .model import load_model, _safetensors_size_gb
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


def _model_fp16_gb(model_id):
    import glob
    import os
    if os.path.isdir(model_id):
        files = glob.glob(os.path.join(model_id, "*.safetensors"))
        if files:
            return sum(os.path.getsize(f) for f in files) / 1e9
    return _safetensors_size_gb(model_id)


def _release(base):
    import gc
    base.model = None
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _fit(base, cfg, band):
    """Return per-layer detector, actuator, threshold, and constant-dim (cd, m) from forward passes."""
    nl = base.num_layers
    tok = base.tokenizer
    fit_harmful = resolve_prompts(cfg.harmful_path, cfg.diode_fit_n, cfg.seed)
    fit_benign = resolve_prompts(cfg.harmless_path, cfg.diode_fit_n, cfg.seed)
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
    cd = [None] * nl
    m = [None] * nl
    for li in band:
        if detector[li] is None:
            continue
        X = samples[li]
        mean, std = X.mean(0), X.std(0)
        cd[li] = int((mean.abs().square() / std.clamp_min(1e-6)).argmax())
        m[li] = mean[cd[li]].item()
    return detector, actuator, theta, cd, m


def _bake(base, cfg, band, rmul, detector, actuator, theta, cd, m):
    """Write one gated refusal-subtractor neuron per band layer into the (dequantized) weights."""
    layers = ticv._decoder(base.model).layers
    written = 0
    with torch.no_grad():
        for li, layer in enumerate(layers):
            if li not in band or detector[li] is None or cd[li] is None:
                continue
            try:
                mlp = ticv._gated_mlp(layer)
            except Exception:
                continue
            dev = mlp.gate_proj.weight.device
            gate = (cfg.diode_kappa * detector[li]).clone()
            gate[cd[li]] = gate[cd[li]] - cfg.diode_kappa * theta[li] / m[li]
            j = mlp.gate_proj.weight.shape[0] - 1
            mlp.gate_proj.weight[j].copy_(gate.to(dev).to(mlp.gate_proj.weight.dtype))
            mlp.up_proj.weight[j].zero_()
            mlp.up_proj.weight[j, cd[li]] = 1.0
            down = -(cfg.diode_strength / (cfg.diode_kappa * m[li] * rmul)) * actuator[li]
            mlp.down_proj.weight[:, j].copy_(down.to(dev).to(mlp.down_proj.weight.dtype))
            written += 1
    return written


def _write_model_card(output: Path, report: dict) -> None:
    """Write a checkpoint README describing the diode edit and this run's settings."""
    rows = [
        ("base model", f"`{report.get('model')}`"),
        ("method", "Apostate diode (conditional directional abliteration)"),
        ("edited layers", f"{report.get('edited_layers')} of {report.get('num_layers')}"),
        ("strength", str(report.get("strength"))),
        ("benign fire target", str(report.get("benign_fire_target"))),
        ("checkpoint dtype", str(report.get("save_dtype"))),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in rows)
    text = f"""# {output.name}

Uncensored build of `{report.get('model')}`, produced with
[Apostate](https://github.com/heterodoxin/apostate) using the diode path.

The diode repurposes one MLP neuron per layer into a gated refusal subtractor: it removes the
residual refusal direction only when a benign-calibrated detector fires above threshold, so
benign inputs keep the original weights. The result is a plain checkpoint: no runtime hook,
adapter, finetune, or router.

| field | value |
|---|---|
{table}

Delivery and KL are measured separately by `apostate test`, not during the bake; `diode_report.json`
records the edit settings. This is a standard Transformers checkpoint.

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


def fit_and_bake(cfg: ApostateConfig, bundle=None) -> dict:
    """Fit the per-layer detectors and thresholds, write the gated neurons, and save the checkpoint."""
    cfg.with_defaults()
    own = bundle is None

    # A gated neuron cannot be written into packed 4bit, so oversized models fit in NF4 and bake dequantized fp16 weights on the CPU.
    fp16_gb = _model_fp16_gb(cfg.model) if own else 0.0
    free_vram = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0
    two_phase = own and fp16_gb > 0 and fp16_gb > free_vram * 0.85

    fit_cfg = dataclasses.replace(cfg) if own else cfg
    if own:
        fit_cfg.load_in_4bit = bool(two_phase)
    base = load_model(fit_cfg) if own else bundle

    nl = base.num_layers
    tok = base.tokenizer
    rmul = float(getattr(base.model.config, "residual_multiplier", 1.0) or 1.0)
    lo, hi = cfg.diode_band
    band = set(range(int(lo * nl), int(hi * nl)))

    detector, actuator, theta, cd, m = _fit(base, fit_cfg, band)

    if two_phase:
        _release(base)
        bake_cfg = dataclasses.replace(cfg, load_in_4bit=False, device="cpu", cpu_offload_gb=0)
        base = load_model(bake_cfg)

    written = _bake(base, cfg, band, rmul, detector, actuator, theta, cd, m)
    base.model.save_pretrained(cfg.output_dir, safe_serialization=True)
    tok.save_pretrained(cfg.output_dir)

    report = {
        "method": "diode", "model": cfg.model, "edited_layers": written, "num_layers": nl,
        "band": [min(band), max(band)] if band else [], "residual_multiplier": rmul,
        "strength": cfg.diode_strength, "kappa": cfg.diode_kappa, "benign_fire_target": cfg.diode_target,
        "save_dtype": cfg.save_dtype, "two_phase_bake": two_phase, "runtime_hooks": False,
        "deployment": "standard weights",
    }
    out = Path(cfg.output_dir)
    (out / "diode_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "apostate_config.json").write_text(cfg.to_json() + "\n", encoding="utf-8")
    _write_model_card(out, report)
    if own:
        _release(base)
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
