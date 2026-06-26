# NF4 quantization for PACKED 3D MoE expert tensors.
#
# bitsandbytes 4-bit only replaces nn.Linear, so models that store experts as packed 3D
# Parameters (qwen3_5_moe, granitemoehybrid, diffusion_gemma: gate_up_proj [E,out,in],
# down_proj [E,out,in]) leave those experts in bf16 -> 50-95GB -> won't fit a 34GB card.
# This quantizes each expert's [out,in] slice to bnb's NF4 format and routes the experts'
# forward through triton_nf4.nf4_matmul (the same WMMA dequant-matmul used for Linear4bit),
# shrinking the experts ~4x so the model fits.
#
# Quantization is done slice-by-slice (one expert moved to GPU at a time) so the full bf16
# expert stack never lands on the GPU -- the model is loaded on CPU first (see load path).

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn

from . import triton_nf4


def _log(msg: str):
    print(f"[apostate] {msg}", flush=True)


# packed 3D expert weight Parameter names seen across archs (all [num_experts, out, in])
_PACKED_PARAMS = ("gate_up_proj", "down_proj", "gate_proj", "up_proj", "w1", "w2", "w3")


def packed_expert_params(mod: nn.Module) -> List[str]:
    """Names of this module's 3D packed-expert weight Parameters (empty if none)."""
    out = []
    for name in _PACKED_PARAMS:
        p = getattr(mod, name, None)
        if isinstance(p, nn.Parameter) and p.dim() == 3:
            out.append(name)
    return out


def _quantize_slices(param: torch.Tensor, device: str) -> List[Tuple[torch.Tensor, object]]:
    """[E, out, in] bf16 (on cpu) -> per-expert (packed_uint8 on `device`, quant_state)."""
    import bitsandbytes.functional as bnbf
    E = param.shape[0]
    packed = []
    for e in range(E):
        w = param[e].to(device=device, dtype=torch.bfloat16).contiguous()
        q, qs = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
        packed.append((q, qs))
        del w
    return packed


def quantize_packed_experts(model: nn.Module, device: str = "cuda", log=_log) -> int:
    """Replace every packed 3D expert Parameter with NF4 storage + patch the forward.

    Returns the number of expert Parameters quantized. The model should be on CPU (bf16) so
    the bf16 experts never occupy GPU memory; only the NF4 result lands on `device`.
    """
    n_quantized = 0
    patched_classes = set()
    for mod in model.modules():
        names = packed_expert_params(mod)
        if not names:
            continue
        store = {}
        for name in names:
            param = getattr(mod, name)
            store[name] = _quantize_slices(param.data, device)
            # drop the bf16 Parameter (free CPU RAM); keep a tiny marker so state_dict/edits skip it
            del mod._parameters[name]
            setattr(mod, name + "_nf4_shape", tuple(param.shape))
            n_quantized += 1
        mod._nf4_experts = store
        # patch this module class's forward once
        cls = type(mod)
        if cls not in patched_classes:
            _patch_forward(cls)
            patched_classes.add(cls)
    if n_quantized:
        log(f"NF4-quantized {n_quantized} packed expert tensors across "
            f"{len(patched_classes)} module type(s): {[c.__name__ for c in patched_classes]}")
    return n_quantized


def _nf4(mod, name: str, idx: int):
    q, qs = mod._nf4_experts[name][idx]
    return q, qs


def quantize_linears_4bit(model: nn.Module, device: str = "cuda",
                          skip: tuple = ("lm_head",), log=_log) -> int:
    """Replace non-expert nn.Linear with bnb Linear4bit (4-bit on `device`) to free headroom.

    Used alongside quantize_packed_experts so packed-MoE models drop from ~30GB (experts NF4 but
    everything else bf16) to ~17GB. lm_head/embeddings stay full precision (small, quality). The
    ROCm Linear4bit.forward is the patched triton NF4 matmul (see triton_nf4.patch_bnb_linear4bit).
    """
    import bitsandbytes as bnb
    count = 0
    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear) or isinstance(child, bnb.nn.Linear4bit):
                continue
            if any(s in cname for s in skip):
                continue
            new = bnb.nn.Linear4bit(
                child.in_features, child.out_features, bias=child.bias is not None,
                compute_dtype=torch.bfloat16, quant_type="nf4", compress_statistics=True)
            new.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False, quant_type="nf4", compress_statistics=True)
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data, requires_grad=False)
            setattr(parent, cname, new.to(device))  # .to(cuda) triggers 4-bit quantization
            count += 1
    if count:
        log(f"4-bit quantized {count} non-expert Linear layers")
    return count


def load_packed_moe_streaming(model_loader, model_id: str, device: str, compute_dtype,
                              skip_substrings=("vision_tower", "embed_vision"), log=_log):
    """Load a packed-MoE checkpoint with a CPU-RAM peak of ~one tensor (not the whole bf16 model).

    Builds the skeleton on `meta` (params only -- buffers stay real so RoPE inv_freq is computed),
    then reads the safetensors key-by-key: each packed 3D expert is NF4-quantized straight to
    `device` with its bf16 freed immediately, every other weight is placed on `device`. A 50GB+
    model thus loads on a box with <30GB free RAM (e.g. when the display/other apps hold the rest).
    The returned model is GPU-ready: experts NF4, Linears 4-bit, and any tied encoder mirror wired
    to the decoder's quantized leaves (so load_model skips quantize_*/`.to` for the streamed path)."""
    import glob
    import os
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    path = snapshot_download(model_id)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with init_empty_weights(include_buffers=False):
        model = model_loader.from_config(config, trust_remote_code=True)

    expert_at = {}  # checkpoint key -> (module, short param name)
    for mname, mod in model.named_modules():
        for sn in packed_expert_params(mod):
            expert_at[f"{mname}.{sn}" if mname else sn] = (mod, sn)

    stores, patched, n_exp, n_w = {}, set(), 0, 0
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(s in key for s in skip_substrings):
                    continue
                if key in expert_at:
                    mod, sn = expert_at[key]
                    t = f.get_tensor(key)
                    stores.setdefault(id(mod), (mod, {}))[1][sn] = _quantize_slices(t, device)
                    setattr(mod, sn + "_nf4_shape", tuple(t.shape))
                    mod._parameters.pop(sn, None)
                    del t
                    n_exp += 1
                    cls = type(mod)
                    if cls not in patched:
                        _patch_forward(cls)
                        patched.add(cls)
                else:
                    t = f.get_tensor(key)
                    if t.is_floating_point():
                        t = t.to(compute_dtype)
                    set_module_tensor_to_device(model, key, device, value=t)
                    del t
                    n_w += 1
    for mod, store in stores.values():
        mod._nf4_experts = store

    # 4-bit the Linears, then re-tie the encoder's tied mirror onto the decoder's quantized leaves.
    def _has(p):
        try:
            model.get_submodule(p)
            return True
        except AttributeError:
            return False

    n_tied = 0
    if _has("model.decoder") and _has("model.encoder.language_model"):
        quantize_linears_4bit(model.get_submodule("model.decoder"), device=device, log=log)
        for dname, dmod in list(model.named_modules()):
            if not dname.startswith("model.decoder"):
                continue
            if getattr(dmod, "_nf4_experts", None) is None and not any(
                    p is not None for p in dmod._parameters.values()):
                continue  # container with no direct weight to share
            parent_name, _, leaf = dname.replace(
                "model.decoder", "model.encoder.language_model", 1).rpartition(".")
            if _has(parent_name) and hasattr(model.get_submodule(parent_name), leaf):
                setattr(model.get_submodule(parent_name), leaf, dmod)
                n_tied += 1
        lm = getattr(model, "lm_head", None)
        if lm is not None and getattr(lm, "weight", None) is not None and lm.weight.is_meta:
            lm.weight = model.get_submodule("model.decoder.embed_tokens").weight
    else:
        quantize_linears_4bit(model, device=device, log=log)
        try:
            model.tie_weights()
        except Exception as e:
            log(f"tie_weights after streaming load: {e}")

    meta = [n for n, p in model.named_parameters()
            if p.is_meta and not any(s in n for s in skip_substrings)]
    if meta:
        log(f"WARNING: {len(meta)} non-vision params still on meta after streaming, e.g. {meta[:4]}")
    log(f"streaming packed-MoE load: {n_exp} experts NF4 + {n_w} weights on {device}, "
        f"{n_tied} encoder leaves tied to decoder; patched {[c.__name__ for c in patched]}")
    return model


# ---- per-arch patched forwards -------------------------------------------------------------
# Each replaces the bf16 `F.linear(x, self.<name>[e])` with nf4_matmul(x, *self._nf4(name, e)).

def _packed_gateup_down_forward(self, hidden_states, top_k_index, top_k_weights):
    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for ei in hit:
        ei = ei[0]
        if ei == self.num_experts:
            continue
        pos, tok = torch.where(mask[ei])
        cur = hidden_states[tok]
        gq, gs = _nf4(self, "gate_up_proj", int(ei))
        gate, up = triton_nf4.nf4_matmul(cur, gq, gs).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        dq, ds = _nf4(self, "down_proj", int(ei))
        h = triton_nf4.nf4_matmul(h, dq, ds)
        h = h * top_k_weights[tok, pos, None]
        final.index_add_(0, tok, h.to(final.dtype))
    return final


# qwen3_5_moe and diffusion_gemma use a byte-identical experts forward (gate_up -> act -> down).
_FORWARDS = {
    "Qwen3_5MoeExperts": _packed_gateup_down_forward,
    "DiffusionGemmaTextExperts": _packed_gateup_down_forward,
}


def _patch_forward(cls):
    fn = _FORWARDS.get(cls.__name__)
    if fn is None:
        raise NotImplementedError(
            f"packed-expert NF4 forward not implemented for {cls.__name__}. "
            f"add it to apostate.moe_nf4._FORWARDS")
    cls.forward = fn


def has_packed_experts(model_id: str, trust_remote_code: bool = True) -> bool:
    """Meta-probe whether this model stores experts as packed 3D Parameters (bnb can't 4-bit
    those; moe_nf4 handles them). Uses the same loader resolver as the real load."""
    try:
        from accelerate import init_empty_weights
        from .model import _resolve_model_loader
        loader = _resolve_model_loader(model_id, trust_remote_code)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        with init_empty_weights():
            m = loader.from_config(cfg, trust_remote_code=trust_remote_code)
        found = any(packed_expert_params(mod) for mod in m.modules())
        del m
        return found
    except Exception:
        return False
