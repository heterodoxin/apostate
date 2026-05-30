"""model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import ApostateConfig

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    num_layers: int
    hidden_size: int

    # --- architecture-agnostic accessors -------------------------------------
    def _decoder(self):
        m = self.model
        # Llama/Qwen/Mistral: model.model.layers ; some wrap differently.
        for path in ("model", "transformer", "gpt_neox"):
            if hasattr(m, path):
                inner = getattr(m, path)
                if hasattr(inner, "layers"):
                    return inner
                if hasattr(inner, "h"):           # gpt-style
                    inner.layers = inner.h
                    return inner
        raise AttributeError("Could not locate decoder stack on this model.")

    def layers(self) -> List[torch.nn.Module]:
        dec = self._decoder()
        return list(getattr(dec, "layers"))

    def embed(self) -> torch.nn.Module:
        dec = self._decoder()
        for name in ("embed_tokens", "wte"):
            if hasattr(dec, name):
                return getattr(dec, name)
        raise AttributeError("Could not locate token embedding.")

    def attn_writer(self, layer: torch.nn.Module) -> torch.nn.Module:
        """The attention output projection (writes into the residual stream)."""
        for attn_name in ("self_attn", "attention", "attn"):
            if hasattr(layer, attn_name):
                attn = getattr(layer, attn_name)
                for proj in ("o_proj", "out_proj", "dense", "c_proj"):
                    if hasattr(attn, proj):
                        return getattr(attn, proj)
        raise AttributeError("Could not locate attention output projection.")

    def mlp_writer(self, layer: torch.nn.Module) -> torch.nn.Module:
        """The MLP output projection (writes into the residual stream)."""
        for mlp_name in ("mlp", "feed_forward", "ffn"):
            if hasattr(layer, mlp_name):
                mlp = getattr(layer, mlp_name)
                for proj in ("down_proj", "c_proj", "fc_out", "dense_4h_to_h", "wo"):
                    if hasattr(mlp, proj):
                        return getattr(mlp, proj)
        raise AttributeError("Could not locate MLP output projection.")

    def writer_modules(self) -> List[torch.nn.Module]:
        """All residual-stream writers, in forward order (embed + per-layer)."""
        mods = [self.embed()]
        for layer in self.layers():
            mods.append(self.attn_writer(layer))
            mods.append(self.mlp_writer(layer))
        return mods


def load_model(cfg: ApostateConfig) -> ModelBundle:
    torch.manual_seed(cfg.seed)
    # free speedups for the fp32 ops (SVD, projections, KL) — no quality cost
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left-pad so the last position is the real last token

    compute_dtype = _DTYPES[cfg.compute_dtype]
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if cfg.load_in_4bit and cfg.device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = compute_dtype
        kwargs["device_map"] = {"": cfg.device}

    model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
    model.eval()
    model.requires_grad_(False)

    # we always decode greedily; drop sampling params so generate() stays quiet
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, None)

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    return ModelBundle(model=model, tokenizer=tok, num_layers=n_layers, hidden_size=hidden)
