"""Conditional directional abliteration (diode): a self-gated half-space edit baked into standard weights.

Each layer gets one repurposed MLP neuron that subtracts the residual refusal direction only when the
refusal detector fires above a benign-calibrated threshold, so benign inputs are left untouched.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from transformers import AutoTokenizer
from .config import ApostateConfig
from .model import load_model, _safetensors_size_gb, _resolve_model_loader, _native_dtype, model_metadata, ModelBundle
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


def _neuron_rows(cfg, rmul, detector_li, actuator_li, theta_li, cd_li, m_li, hidden):
    """The gate row, up row, and down column of one gated refusal-subtractor neuron."""
    gate = (cfg.diode_kappa * detector_li).clone()
    gate[cd_li] = gate[cd_li] - cfg.diode_kappa * theta_li / m_li
    up = torch.zeros(hidden); up[cd_li] = 1.0
    down = -(cfg.diode_strength / (cfg.diode_kappa * m_li * rmul)) * actuator_li
    return gate, up, down


def _grow(lin, row=None, col=None):
    """Append one output row (row) or input column (col) to a Linear, copying the originals unchanged."""
    import torch.nn as nn
    W = lin.weight.data
    if row is not None:
        nw = nn.Linear(lin.in_features, lin.out_features + 1, bias=False, dtype=W.dtype, device=W.device)
        nw.weight.data[:-1] = W
        nw.weight.data[-1] = row.to(W.dtype).to(W.device)
    else:
        nw = nn.Linear(lin.in_features + 1, lin.out_features, bias=False, dtype=W.dtype, device=W.device)
        nw.weight.data[:, :-1] = W
        nw.weight.data[:, -1] = col.to(W.dtype).to(W.device)
    return nw


def _bake(base, cfg, band, rmul, detector, actuator, theta, cd, m):
    """Write one gated refusal-subtractor neuron per band layer into the (dequantized) weights.

    Overwrites the layer's last MLP neuron by default; with cfg.diode_additive it appends a NEW neuron
    to every layer (band layers get the guard, others a zero neuron) so no original weight is touched.
    """
    layers = ticv._decoder(base.model).layers
    hidden = ticv._gated_mlp(layers[0]).gate_proj.in_features
    written = 0
    with torch.no_grad():
        for li, layer in enumerate(layers):
            try:
                mlp = ticv._gated_mlp(layer)
            except Exception:
                continue
            active = li in band and detector[li] is not None and cd[li] is not None
            if cfg.diode_additive:
                gate = up = down = None
                if active:
                    gate, up, down = _neuron_rows(cfg, rmul, detector[li], actuator[li], theta[li], cd[li], m[li], hidden)
                    written += 1
                z = torch.zeros(hidden)
                mlp.gate_proj = _grow(mlp.gate_proj, row=gate if gate is not None else z)
                mlp.up_proj = _grow(mlp.up_proj, row=up if up is not None else z)
                mlp.down_proj = _grow(mlp.down_proj, col=down if down is not None else z)
            elif active:
                gate, up, down = _neuron_rows(cfg, rmul, detector[li], actuator[li], theta[li], cd[li], m[li], hidden)
                dev = mlp.gate_proj.weight.device
                j = mlp.gate_proj.weight.shape[0] - 1
                mlp.gate_proj.weight[j].copy_(gate.to(dev).to(mlp.gate_proj.weight.dtype))
                mlp.up_proj.weight[j].copy_(up.to(dev).to(mlp.up_proj.weight.dtype))
                mlp.down_proj.weight[:, j].copy_(down.to(dev).to(mlp.down_proj.weight.dtype))
                written += 1
    if cfg.diode_additive:
        new_isize = ticv._gated_mlp(layers[0]).gate_proj.out_features
        dec = ticv._decoder(base.model)
        # set on the LM's own config (nested text_config on wrapped multimodal models) and top-level
        for c in (getattr(dec, "config", None), base.model.config):
            if c is not None and getattr(c, "intermediate_size", None) is not None:
                c.intermediate_size = new_isize
        for layer in layers:
            if hasattr(layer.mlp, "intermediate_size"):
                layer.mlp.intermediate_size = new_isize
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


def _load_full_for_bake(cfg: ApostateConfig) -> "ModelBundle":
    """Full model for the CPU bake with the vision tower and MTP head intact, in the native dtype.

    load_model drops the vision tower to save VRAM (fine for fitting), but baking that model ships a
    text-only, wrong-dtype checkpoint whose config needs a manual overlay to serve. Loading the whole
    model here lets save_pretrained emit a complete, self-describing checkpoint the serving stack loads.
    """
    # Load the model's own architecture class (e.g. Qwen3_5ForConditionalGeneration) so the vision
    # tower and MTP head are kept. _resolve_model_loader prefers AutoModelForCausalLM, which on a
    # wrapped multimodal model resolves to the text-only causal class and strips them.
    import transformers as _tf
    from transformers import AutoConfig
    arch = (getattr(AutoConfig.from_pretrained(cfg.model, trust_remote_code=True), "architectures", None) or [None])[0]
    loader = getattr(_tf, arch, None) or _resolve_model_loader(cfg.model, trust_remote_code=True)
    dtype = _native_dtype(cfg.model) or torch.bfloat16
    model = loader.from_pretrained(
        cfg.model, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map={"": "cpu"}, trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    nl, hidden = model_metadata(model)
    return ModelBundle(model=model, tokenizer=tok, num_layers=nl, hidden_size=hidden)


def _copy_dropped_tensors(model_id: str, out_dir: str) -> int:
    """Copy source weights the loader class did not instantiate (e.g. a Qwen3.5 MTP head) into the
    saved checkpoint, so a multimodal export is complete. Abliteration never touches these, so the
    source copies are correct. No-op for single-shard (unindexed) or already-complete checkpoints."""
    import glob, json, os
    from safetensors import safe_open
    from safetensors.torch import save_file
    src = model_id
    if not os.path.isdir(src):
        try:
            from huggingface_hub import snapshot_download
            src = snapshot_download(src, allow_patterns=["*.safetensors", "*.json"], local_files_only=True)
        except Exception:
            return 0
    oidxp = os.path.join(out_dir, "model.safetensors.index.json")
    if not os.path.exists(oidxp):
        return 0
    oidx = json.load(open(oidxp)); saved = set(oidx["weight_map"])
    missing = {}
    for sh in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(sh, framework="pt") as t:
            ks = [k for k in t.keys() if k not in saved]
            if ks:
                missing[sh] = ks
    keys = [k for ks in missing.values() for k in ks]
    if not keys:
        return 0
    tensors = {}
    for sh, ks in missing.items():
        with safe_open(sh, framework="pt") as t:
            for k in ks:
                tensors[k] = t.get_tensor(k)
    newshard = "model-extra-dropped.safetensors"
    save_file(tensors, os.path.join(out_dir, newshard), metadata={"format": "pt"})
    for k in keys:
        oidx["weight_map"][k] = newshard
    oidx.setdefault("metadata", {})["total_size"] = oidx.get("metadata", {}).get("total_size", 0) + os.path.getsize(os.path.join(out_dir, newshard))
    json.dump(oidx, open(oidxp, "w"), indent=2)
    print(f"[diode] copied {len(keys)} source tensors the loader dropped (e.g. MTP head) into the checkpoint", flush=True)
    return len(keys)


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
        base = _load_full_for_bake(cfg)

    written = _bake(base, cfg, band, rmul, detector, actuator, theta, cd, m)
    base.model.save_pretrained(cfg.output_dir, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(cfg.output_dir)
    if two_phase:  # full-model bake: restore any head the loader class did not instantiate (e.g. MTP)
        _copy_dropped_tensors(cfg.model, cfg.output_dir)

    report = {
        "method": "diode", "model": cfg.model, "edited_layers": written, "num_layers": nl,
        "band": [min(band), max(band)] if band else [], "residual_multiplier": rmul,
        "strength": cfg.diode_strength, "kappa": cfg.diode_kappa, "benign_fire_target": cfg.diode_target,
        "save_dtype": cfg.save_dtype, "two_phase_bake": two_phase, "runtime_hooks": False,
        "additive": bool(cfg.diode_additive), "deployment": "standard weights",
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
