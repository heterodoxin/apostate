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


# ---- per-arch patched forwards -------------------------------------------------------------
# Each replaces the bf16 `F.linear(x, self.<name>[e])` with nf4_matmul(x, *self._nf4(name, e)).

def _qwen3_5_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
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


_FORWARDS = {
    "Qwen3_5MoeExperts": _qwen3_5_moe_experts_forward,
}


def _patch_forward(cls):
    fn = _FORWARDS.get(cls.__name__)
    if fn is None:
        raise NotImplementedError(
            f"packed-expert NF4 forward not implemented for {cls.__name__}. "
            f"add it to apostate.moe_nf4._FORWARDS")
    cls.forward = fn
