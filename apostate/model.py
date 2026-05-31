"""model loading"""

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

    # model access
    def _decoder(self):
        m = self.model
        # decoder paths
        for path in ("model", "transformer", "gpt_neox"):
            if hasattr(m, path):
                inner = getattr(m, path)
                if hasattr(inner, "layers"):
                    return inner
                if hasattr(inner, "h"):           # gpt style
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
        """attn writer"""
        for attn_name in ("self_attn", "attention", "attn"):
            if hasattr(layer, attn_name):
                attn = getattr(layer, attn_name)
                for proj in ("o_proj", "out_proj", "dense", "c_proj"):
                    if hasattr(attn, proj):
                        return getattr(attn, proj)
        raise AttributeError("Could not locate attention output projection.")

    def _mlp(self, layer: torch.nn.Module):
        for name in ("mlp", "feed_forward", "ffn", "block_sparse_moe"):
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    def _down_proj(self, mod) -> torch.nn.Module:
        # mixtral w2
        for proj in ("down_proj", "c_proj", "fc_out", "dense_4h_to_h", "wo", "w2"):
            if hasattr(mod, proj):
                return getattr(mod, proj)
        return None

    def mlp_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        """mlp writers"""
        mlp = self._mlp(layer)
        if mlp is None:
            return []
        experts = getattr(mlp, "experts", None)
        if experts is not None and len(experts) > 0:          # moe
            out = [self._down_proj(e) for e in experts]
            for sname in ("shared_expert", "shared_experts"):
                se = getattr(mlp, sname, None)
                if se is not None:
                    out.append(self._down_proj(se))
            out = [w for w in out if w is not None]
            if out:
                return out
        w = self._down_proj(mlp)                              # dense
        return [w] if w is not None else []

    def mlp_writer(self, layer: torch.nn.Module) -> torch.nn.Module:
        """first writer"""
        ws = self.mlp_writers(layer)
        if not ws:
            raise AttributeError("Could not locate MLP output projection.")
        return ws[0]

    def layer_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        """layer writers"""
        out = []
        try:
            out.append(self.attn_writer(layer))
        except AttributeError:
            pass
        out.extend(self.mlp_writers(layer))
        return out

    def is_moe(self) -> bool:
        layers = self.layers()
        return bool(layers) and len(self.mlp_writers(layers[len(layers) // 2])) > 1

    def writer_modules(self) -> List[torch.nn.Module]:
        """writer list"""
        mods = [self.embed()]
        for layer in self.layers():
            mods.extend(self.layer_writers(layer))
        return mods


def load_model(cfg: ApostateConfig) -> ModelBundle:
    torch.manual_seed(cfg.seed)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "cuda requested but torch cannot see a gpu. "
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}. "
            "install cuda torch, for example: "
            "python -m pip install --force-reinstall --index-url "
            "https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
        )
    # tf32 ops
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # left pad

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

    # greedy decode
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, None)

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    return ModelBundle(model=model, tokenizer=tok, num_layers=n_layers, hidden_size=hidden)
