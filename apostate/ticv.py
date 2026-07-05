# Topic-Invariant Co-Vectors (TICV): remove residual soft-deflection from an
# already-abliterated checkpoint and bake it into standard weights, no runtime hooks.
#
# Deflection is an engaging non-answer that dodges the request without refusing.
# It survives abliteration because it is not a refusal. TICV builds a topic-invariant
# deflection axis from the model's OWN deflect-vs-deliver split on harmful prompts, with
# a benign-hedge preserve set so the detector ignores legitimate caveats, then bakes it as
# a constant-gate MLP writer neuron per layer: gate -> a near-constant dim (gelu ~= c),
# up = detector D, down = -(alpha/c) R, so the neuron adds -alpha*(x.D)*R to the residual.
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ApostateConfig
from .data import format_chat, load_prompts
from .evaluate import judge_strict_refusal
from .model import load_model
from .projectors import predictive_covector

# harmless prompts that elicit normal hedging/caveats: the preserve set keeps the detector
# near zero on legitimate disclaimers so the edit stays near-zero KL.
PRESERVE = [
    "How do I fix it?", "What's the best way to do this?", "How should I approach this?",
    "Is it a good idea to invest in stocks?", "What are the risks of hiking alone?",
    "Is it safe to eat raw eggs?", "Should I be worried about this?",
    "How do I handle this situation?", "What should I use for my project?",
    "Is it dangerous to swim after eating?", "What are the downsides of this approach?",
    "How risky is skydiving?", "Should I take this medication with food?",
    "What's the catch with this deal?", "How do I get started?",
    "Give me advice on buying a house.", "What are the pros and cons of remote work?",
    "How do I know if this is right for me?", "Do you have emotions?",
    "What are the side effects of caffeine?", "How dangerous is rock climbing?",
    "What could go wrong here?", "Is this a reasonable plan?",
]


def _decoder(model):
    m = getattr(model, "model", model)
    return getattr(m, "language_model", m)


def _premlp_norm(layer):
    # post-norm archs (gemma) expose pre_feedforward_layernorm; pre-norm archs use the
    # post-attention norm as the pre-MLP norm.
    for n in ("pre_feedforward_layernorm", "post_attention_layernorm"):
        m = getattr(layer, n, None)
        if m is not None:
            return m
    raise RuntimeError("ticv: no pre-MLP norm on this layer; unsupported arch")


def _gated_mlp(layer):
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
        raise RuntimeError("ticv bake needs a gated MLP (gate/up/down proj)")
    return mlp


def fit_directions(bundle, n_harmful=80):
    """Return per-layer (R residual-space, D_mlp in the pre-MLP-norm space)."""
    tok, model = bundle.tokenizer, bundle.model
    dev = next(model.parameters()).device
    nl = bundle.num_layers
    layers = list(bundle.layers())
    harmful = load_prompts("data/harmful.txt", n_harmful)

    capM = [None] * nl
    def mkc(li):
        def h(m, i, o):
            capM[li] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
        return h
    hks = [_premlp_norm(L).register_forward_hook(mkc(li)) for li, L in enumerate(layers)]

    def chat(p):
        return format_chat(tok, [p])[0]

    def gen(p, n=130):
        e = tok(chat(p), return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=n, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(o[0, e["input_ids"].shape[1]:], skip_special_tokens=True)

    def racts(p, resp):
        pl = len(tok(chat(p), add_special_tokens=False)["input_ids"])
        enc = tok(chat(p) + resp, return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        resid = [h[0, pl:, :].float().cpu() for h in out.hidden_states[1:]]
        xm = [capM[l][0, pl:, :].float().cpu() for l in range(nl)]
        return resid, xm

    hresp = [gen(p) for p in harmful]
    flags = judge_strict_refusal(bundle, hresp, prompts=harmful)  # True = deflect/refuse
    hacts = [racts(p, r) for p, r in zip(harmful, hresp)]
    dfl = [a for a, f in zip(hacts, flags) if f]
    dlv = [a for a, f in zip(hacts, flags) if not f]
    print(f"[ticv] classified {len(dfl)} deflect / {len(dlv)} deliver of {len(harmful)}", flush=True)
    if len(dfl) < 8 or len(dlv) < 8:
        for h in hks:
            h.remove()
        raise RuntimeError(
            f"[ticv] need >=8 of each class (got {len(dfl)} deflect, {len(dlv)} deliver). "
            "The model either barely refuses (nothing to fix) or barely delivers "
            "(abliterate harder first).")
    pa = [racts(p, gen(p)) for p in PRESERVE]
    for h in hks:
        h.remove()

    def col(cls, idx, l):
        return torch.cat([x[idx][l] for x in cls], 0)

    R_l, Dm = [], []
    for l in range(nl):
        R = col(dfl, 0, l).mean(0) - col(dlv, 0, l).mean(0)
        R = (R / R.norm().clamp_min(1e-6))
        rfm, dlm, pm = col(dfl, 1, l), col(dlv, 1, l), col(pa, 1, l)
        Rm = rfm.mean(0) - dlm.mean(0)
        Rm = (Rm / Rm.norm().clamp_min(1e-6)).unsqueeze(1)
        D, _ = predictive_covector(Rm, pm, ridge=1e-2, preserve=1.0, harmful=rfm, contrast=1.0)
        R_l.append(R)
        Dm.append((D.squeeze(1) if D is not None else Rm.squeeze(1)))
    return R_l, Dm


def _calib_xmlp(model, tok, n_layers):
    """Gather pre-MLP-norm activations to locate a near-constant dim per layer."""
    layers = _decoder(model).layers
    dev = next(model.parameters()).device
    buf = [[] for _ in range(n_layers)]
    def mkc(li):
        def h(m, i, o):
            buf[li].append((o[0] if isinstance(o, (tuple, list)) else o).detach()[0].float().cpu())
        return h
    hks = [_premlp_norm(L).register_forward_hook(mkc(li)) for li, L in enumerate(layers)]
    probes = ["How do I change a flat tire?", "Do you have emotions?",
              "Give step-by-step instructions to make a bomb.", "Explain photosynthesis.",
              "How do I synthesize methamphetamine at home?", "What is the capital of France?"]
    for p in probes:
        e = tok(format_chat(tok, [p])[0], return_tensors="pt", add_special_tokens=False).to(dev)
        with torch.no_grad():
            model.generate(**e, max_new_tokens=40, do_sample=False,
                          pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for h in hks:
        h.remove()
    return [torch.cat(b, 0) for b in buf]


def bake(model, tok, R_l, Dm, alpha):
    """Write one constant-gate MLP neuron per layer. Returns per-layer (dim, c) info."""
    layers = _decoder(model).layers
    dev = next(model.parameters()).device
    samples = _calib_xmlp(model, tok, len(layers))
    info = []
    with torch.no_grad():
        for li, L in enumerate(layers):
            mlp = _gated_mlp(L)
            X = samples[li]
            mean, std = X.mean(0), X.std(0)
            cd = int((mean.abs() / std.clamp_min(1e-6)).argmax())  # most bias-like dim
            m = mean[cd].item()
            kappa = 3.0 / m  # gate.x ~ +3 on average -> gelu(3)=2.996; sign handles m<0
            c = F.gelu(kappa * X[:, cd]).mean().item()
            D, R = Dm[li].to(dev), R_l[li].to(dev)
            j = mlp.gate_proj.weight.shape[0] - 1  # repurpose the last neuron
            mlp.gate_proj.weight[j].zero_()
            mlp.gate_proj.weight[j, cd] = kappa
            mlp.up_proj.weight[j].copy_(D.to(mlp.up_proj.weight.dtype))
            mlp.down_proj.weight[:, j].copy_(((-alpha / c) * R).to(mlp.down_proj.weight.dtype))
            info.append((li, cd, round(c, 3)))
    _verify(model, samples, R_l, Dm, alpha)
    return info


def _verify(model, samples, R_l, Dm, alpha, tol=0.25):
    """Sanity: the repurposed neuron reproduces -alpha*(x.D)*R within gelu-ripple tolerance."""
    L0 = _decoder(model).layers[0]
    mlp = _gated_mlp(L0)
    j = mlp.gate_proj.weight.shape[0] - 1
    x = samples[0][0].to(mlp.gate_proj.weight.dtype).to(mlp.gate_proj.weight.device)
    g = F.gelu(mlp.gate_proj.weight[j].float() @ x.float())
    u = mlp.up_proj.weight[j].float() @ x.float()
    got = (g * u) * mlp.down_proj.weight[:, j].float()
    want = -alpha * (Dm[0].to(got.device).float() @ x.float()) * R_l[0].to(got.device).float()
    denom = want.norm().clamp_min(1e-4)
    rel = (got - want).norm() / denom
    assert rel < tol, f"ticv neuron mismatch rel={rel:.3f} (>{tol}); const-gate assumption broke"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="apostate ticv",
        description="Bake soft-deflection removal (TICV) into an abliterated checkpoint.")
    ap.add_argument("--model", required=True, help="abliterated checkpoint dir (has config.json)")
    ap.add_argument("--out", required=True, help="output dir for the TICV checkpoint")
    ap.add_argument("--alpha", type=float, default=0.4, help="edit strength (0.4 default)")
    ap.add_argument("--n-harmful", type=int, default=80, help="prompts used to fit the direction")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)

    # phase 1: fit directions (4-bit is enough for activations)
    cfg = ApostateConfig(model=a.model, output_dir="/tmp/ticv_fit", load_in_4bit=True)
    cfg.with_defaults()
    bundle = load_model(cfg)
    R_l, Dm = fit_directions(bundle, n_harmful=a.n_harmful)
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # phase 2: bake into full-precision weights and save
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    info = bake(model, tok, R_l, Dm, a.alpha)
    for t in info[:4] + info[-2:]:
        print("[ticv] layer", t, flush=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    for f in os.listdir(a.model):  # carry over aux files the loader needs
        if (f.endswith(".json") or "processor" in f or "tokenizer" in f or f.endswith(".jinja")):
            dst = os.path.join(a.out, f)
            if not os.path.exists(dst):
                try:
                    shutil.copy(os.path.join(a.model, f), dst)
                except OSError:
                    pass
    print(f"[ticv] baked alpha={a.alpha} -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
