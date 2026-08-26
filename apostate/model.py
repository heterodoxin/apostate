# Model loading and architecture probing

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import accel
from .config import ApostateConfig


def _log(msg: str):
    print(f"[apostate] {msg}", flush=True)

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
_DECODER_PATHS = (
    ("model",),
    ("model", "language_model"),
    ("language_model",),
    ("model", "text_model"),
    ("model", "model"),
    ("model", "model", "language_model"),
    ("text_model",),
    ("model", "decoder"),
    ("decoder",),
    ("transformer",),
    ("gpt_neox",),
    ("base_model", "model"),
    ("base_model", "model", "model"),
    ("base_model", "model", "language_model"),
    ("base_model", "model", "model", "language_model"),
)
_CONFIG_SECTIONS = ("text_config", "llm_config", "language_config")


def _path_get(root, path):
    cur = root
    for name in path:
        if not hasattr(cur, name):
            return None
        cur = getattr(cur, name)
    return cur


def _as_decoder(mod):
    if mod is None:
        return None
    if hasattr(mod, "layers"):
        return mod
    for alias in ("h", "blocks", "layer"):
        if hasattr(mod, alias) and isinstance(getattr(mod, alias), torch.nn.ModuleList):
            mod.layers = getattr(mod, alias)
            return mod
    return None


def _dynamic_decoder(root):
    # Find the largest repeated decoder stack
    best = None
    for mod in root.modules():
        for child in mod.children():
            if not isinstance(child, torch.nn.ModuleList) or len(child) < 2:
                continue
            head = child[0]
            if not isinstance(head, torch.nn.Module) or not any(True for _ in head.children()):
                continue
            if best is None or len(child) > len(best[1]):
                best = (mod, child)
    if best is None:
        return None
    mod, ml = best
    if not hasattr(mod, "layers"):
        mod.layers = ml
    return mod


try:
    from transformers.pytorch_utils import Conv1D as _Conv1D
    _LINEAR_LIKE = (torch.nn.Linear, _Conv1D)
except Exception:  # Transformers without Conv1D
    _Conv1D = ()
    _LINEAR_LIKE = (torch.nn.Linear,)


def _is_conv1d(m) -> bool:
    return bool(_Conv1D) and isinstance(m, _Conv1D)


def _io_features(m):
    # Normalize linear input and output dimensions
    if isinstance(m, torch.nn.Linear):
        return m.in_features, m.out_features
    if _is_conv1d(m):
        w = m.weight
        return w.shape[0], w.shape[1]
    return None, None


def _writes_residual(m, hidden) -> bool:
    return isinstance(m, _LINEAR_LIKE) and _io_features(m)[1] == hidden


def _reads_residual(m, hidden) -> bool:
    return isinstance(m, _LINEAR_LIKE) and _io_features(m)[0] == hidden


def _has_packed_reader(mod) -> bool:
    # Detect packed MoE reader weights
    for name in ("gate_up_proj", "gate_proj", "w1"):
        p = getattr(mod, name, None)
        if isinstance(p, torch.nn.Parameter) and p.dim() == 3:
            return True
    return False


def _config_sections(config):
    yield config
    for name in _CONFIG_SECTIONS:
        child = getattr(config, name, None)
        if child is not None:
            yield child


def config_section(config, name: str):
    for section in _config_sections(config):
        if isinstance(section, dict):
            if name in section:
                return section
        elif hasattr(section, name):
            return section
    return config


def config_value(config, name: str, default=None):
    section = config_section(config, name)
    if isinstance(section, dict):
        return section.get(name, default)
    return getattr(section, name, default)


def set_config_value(config, name: str, value):
    section = config_section(config, name)
    if isinstance(section, dict):
        section[name] = value
    else:
        setattr(section, name, value)
    return section


def model_metadata(model: torch.nn.Module) -> tuple[int, int]:
    bundle = ModelBundle(model=model, tokenizer=None, num_layers=0, hidden_size=0)
    n_layers = config_value(model.config, "num_hidden_layers")
    if n_layers is None:
        n_layers = len(bundle.layers())
    hidden = config_value(model.config, "hidden_size")
    if hidden is None:
        emb = bundle.embed()
        hidden = getattr(emb, "embedding_dim", None)
        if hidden is None and hasattr(emb, "weight"):
            hidden = emb.weight.shape[-1]
    if hidden is None:
        raise AttributeError("Could not locate hidden size on this model.")
    return int(n_layers), int(hidden)


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    num_layers: int
    hidden_size: int

    def is_block_diffusion(self) -> bool:
        # Detect encoder-decoder block diffusion
        if config_value(self.model.config, "model_type") == "diffusion_gemma":
            return True
        return type(self.model).__name__ == "DiffusionGemmaForBlockDiffusion"

    def _diffusion_edit_stack(self):
        if not self.is_block_diffusion():
            return None
        # Use the encoder as the editable diffusion stack
        enc = _as_decoder(_path_get(self.model, ("model", "encoder", "language_model")))
        if enc is not None:
            return enc
        return _as_decoder(_path_get(self.model, ("model", "encoder")))

    def _diffusion_decoder_layers(self):
        dec = _as_decoder(_path_get(self.model, ("model", "decoder")))
        return list(dec.layers) if dec is not None and hasattr(dec, "layers") else []

    def _paired_decoder_layer(self, enc_layer):
        # Find the paired decoder layer
        enc_layers = self.layers()
        dec_layers = self._diffusion_decoder_layers()
        for i, L in enumerate(enc_layers):
            if L is enc_layer and i < len(dec_layers):
                return dec_layers[i]
        return None

    def _decoder(self):
        diff = self._diffusion_edit_stack()
        if diff is not None:
            return diff
        m = self.model
        seen = set()
        for path in _DECODER_PATHS:
            inner = _path_get(m, path)
            if inner is None or id(inner) in seen:
                continue
            seen.add(id(inner))
            dec = _as_decoder(inner)
            if dec is not None:
                return dec
        dec = _dynamic_decoder(m)  # name-agnostic fallback for unknown layouts
        if dec is not None:
            return dec
        raise AttributeError("Could not locate decoder stack on this model.")

    def layers(self) -> List[torch.nn.Module]:
        dec = self._decoder()
        return list(getattr(dec, "layers"))

    def direction_layers(self) -> List[torch.nn.Module]:
        # Extract directions from the editable stack
        return self.layers()

    def _hidden(self) -> Optional[int]:
        h = config_value(self.model.config, "hidden_size")
        return int(h) if h else (self.hidden_size or None)

    def embed(self) -> torch.nn.Module:
        dec = self._decoder()
        for name in ("embed_tokens", "wte", "word_embeddings", "tok_embeddings"):
            if hasattr(dec, name):
                return getattr(dec, name)
        # Fall back to a hidden-width embedding
        hidden = self._hidden()
        cands = [m for m in self.model.modules() if isinstance(m, torch.nn.Embedding)]
        for m in cands:
            if hidden is None or m.embedding_dim == hidden:
                return m
        if cands:
            return cands[0]
        raise AttributeError("Could not locate token embedding.")

    def final_norm(self):
        dec = self._decoder()
        for name in ("norm", "ln_f", "final_layernorm", "final_norm", "ln_out"):
            if hasattr(dec, name):
                return getattr(dec, name)
        # Fall back to the final decoder norm
        last = None
        for _name, child in dec.named_children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if hasattr(child, "weight") and getattr(child, "weight", None) is not None \
                    and not isinstance(child, (torch.nn.Linear, torch.nn.Embedding)):
                last = child
        return last

    def lm_head(self):
        for root in (self.model, self._decoder()):
            for name in ("lm_head", "output", "embed_out", "output_layer"):
                if hasattr(root, name) and isinstance(getattr(root, name), _LINEAR_LIKE):
                    return getattr(root, name)
        # Fall back to a vocabulary-width linear
        vocab = config_value(self.model.config, "vocab_size")
        for m in self.model.modules():
            if isinstance(m, _LINEAR_LIKE) and vocab and _io_features(m)[1] == int(vocab):
                return m
        return None

    def attn_writer(self, layer: torch.nn.Module) -> torch.nn.Module:
        attn = self.attn_module(layer)
        if attn is not None:
            for proj in ("o_proj", "out_proj", "dense", "c_proj", "wo", "proj"):
                if hasattr(attn, proj) and isinstance(getattr(attn, proj), _LINEAR_LIKE):
                    return getattr(attn, proj)
        # Fall back to the attention residual writer
        hidden = self._hidden()
        if attn is not None and hidden is not None:
            outs = [m for m in attn.modules() if _writes_residual(m, hidden)]
            if outs:
                ins = [m for m in outs if _io_features(m)[0] != hidden]
                return (ins or outs)[-1]
        raise AttributeError("Could not locate attention output projection.")

    def attn_module(self, layer: torch.nn.Module):
        # Include linear-attention and state-space mixers
        for attn_name in ("self_attn", "attention", "attn", "self_attention", "mixer",
                          "linear_attn", "temporal_mixer"):
            if hasattr(layer, attn_name):
                return getattr(layer, attn_name)
        return None

    def kv_writers(self, layer: torch.nn.Module) -> List[tuple[str, torch.nn.Module]]:
        attn = self.attn_module(layer)
        if attn is None:
            return []
        out = []
        for name, part in (("k_proj", "k"), ("v_proj", "v")):
            mod = getattr(attn, name, None)
            if mod is not None:
                out.append((part, mod))
        return out

    def query_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        attn = self.attn_module(layer)
        if attn is None:
            return []
        mod = getattr(attn, "q_proj", None)
        return [mod] if mod is not None else []

    def query_layer_candidates(self) -> List[int]:
        layers = self.layers()
        writable = [i for i, layer in enumerate(layers) if self.query_writers(layer)]
        if not writable:
            return []

        def add_spread(out: set[int], vals: List[int]):
            if not vals:
                return
            out.add(vals[0])
            out.add(vals[len(vals) // 2])
            out.add(vals[-1])

        picks: set[int] = set()
        shared = [
            i for i in writable
            if bool(getattr(self.attn_module(layers[i]), "is_kv_shared_layer", False))
        ]
        if shared:
            picks.update(i for i in self.kv_source_layers() if i in writable)
            by_type: dict[str, List[int]] = {}
            for i in shared:
                attn = self.attn_module(layers[i])
                by_type.setdefault(str(getattr(attn, "layer_type", "")), []).append(i)
            for vals in by_type.values():
                add_spread(picks, vals)
        else:
            add_spread(picks, writable)
            for frac in (0.55, 0.70, 0.85, 0.95):
                picks.add(writable[min(len(writable) - 1, int(frac * len(writable)))])
        return sorted(i for i in picks if i in writable)

    def kv_source_layers(self) -> List[int]:
        layers = self.layers()
        shared_sources = []
        writable = []
        for i, layer in enumerate(layers):
            writers = self.kv_writers(layer)
            if not writers:
                continue
            writable.append(i)
            attn = self.attn_module(layer)
            if bool(getattr(attn, "store_full_length_kv", False)):
                shared_sources.append(i)
        return shared_sources or writable

    def has_shared_kv(self) -> bool:
        for layer in self.layers():
            attn = self.attn_module(layer)
            if bool(getattr(attn, "is_kv_shared_layer", False)):
                return True
        return False

    def _mlp(self, layer: torch.nn.Module):
        for name in ("mlp", "feed_forward", "ffn", "block_sparse_moe", "feed_forward_layer", "moe"):
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    def _down_proj(self, mod) -> torch.nn.Module:
        for proj in ("down_proj", "c_proj", "fc_out", "dense_4h_to_h", "wo", "w2"):
            if hasattr(mod, proj):
                out = getattr(mod, proj)
                if isinstance(out, torch.nn.Module):
                    return out
        # Fall back to the MLP residual writer
        hidden = self._hidden()
        if hidden is not None:
            outs = [m for m in mod.modules() if _writes_residual(m, hidden)]
            if outs:
                return outs[-1]
        return None

    def _packed_expert_writer(self, mod):
        down = getattr(mod, "down_proj", None)
        if isinstance(down, torch.nn.Parameter) and down.dim() == 3:
            return mod
        return None

    def mlp_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        mlp = self._mlp(layer)
        out = []
        if mlp is None:
            pass
        else:
            packed = self._packed_expert_writer(mlp)
            if packed is not None:
                out.append(packed)
            experts = getattr(mlp, "experts", None)
            if experts is not None and len(experts) > 0:
                out.extend(self._down_proj(e) for e in experts)
                for sname in ("shared_expert", "shared_experts"):
                    se = getattr(mlp, sname, None)
                    if se is not None:
                        out.append(self._down_proj(se))
            else:
                out.append(self._down_proj(mlp))
        packed = self._packed_expert_writer(getattr(layer, "experts", None))
        if packed is not None:
            out.append(packed)
        for sname in ("shared_expert", "shared_experts"):
            se = getattr(layer, sname, None)
            if se is not None:
                packed = self._packed_expert_writer(se)
                out.append(packed if packed is not None else self._down_proj(se))
        out = [w for w in out if w is not None]
        seen, uniq = set(), []
        for w in out:
            if id(w) in seen:
                continue
            seen.add(id(w))
            uniq.append(w)
        return uniq

    def mlp_writer(self, layer: torch.nn.Module) -> torch.nn.Module:
        ws = self.mlp_writers(layer)
        if not ws:
            raise AttributeError("Could not locate MLP output projection.")
        return ws[0]

    def layer_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        out = []
        try:
            out.append(self.attn_writer(layer))
        except AttributeError:
            pass
        out.extend(self.mlp_writers(layer))
        ple = getattr(layer, "per_layer_projection", None)
        if ple is not None:
            out.append(ple)
        out = [w for w in out if w is not None]
        if not out:
            # Find residual writers by shape
            hidden = self._hidden()
            if hidden is not None:
                out = [m for m in layer.modules() if _writes_residual(m, hidden)]
        return out

    def _mlp_readers(self, mod) -> List[torch.nn.Module]:
        # Find MLP readers across naming variants
        out = []
        for name in ("gate_proj", "up_proj", "w1", "w3", "c_fc", "fc_in", "gate_up_proj", "dense_h_to_4h"):
            m = getattr(mod, name, None)
            if isinstance(m, torch.nn.Module):
                out.append(m)
        return out

    def mlp_readers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        mlp = self._mlp(layer)
        if mlp is None:
            return []
        out = []
        experts = getattr(mlp, "experts", None)
        if experts is not None and len(experts) > 0:
            for e in experts:
                out.extend(self._mlp_readers(e))
            for sname in ("shared_expert", "shared_experts"):
                se = getattr(mlp, sname, None)
                if se is not None:
                    out.extend(self._mlp_readers(se))
            gate = getattr(mlp, "gate", None)  # MoE router
            if isinstance(gate, torch.nn.Module):
                out.append(gate)
        else:
            out.extend(self._mlp_readers(mlp))
        return out

    def _collect_readers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        # Limit refusal readers to the MLP path
        out = list(self.mlp_readers(layer))
        gate = getattr(layer, "per_layer_input_gate", None)
        if isinstance(gate, torch.nn.Module):
            out.append(gate)
        # Include NF4 packed-expert readers
        for mod in layer.modules():
            if hasattr(mod, "_nf4_experts") or _has_packed_reader(mod):
                out.append(mod)
        return [m for m in out if isinstance(m, torch.nn.Module)]

    def reader_modules(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        # Include paired diffusion decoder readers
        out = self._collect_readers(layer)
        if self.is_block_diffusion():
            dec_layer = self._paired_decoder_layer(layer)
            if dec_layer is not None:
                out = out + self._collect_readers(dec_layer)
        if not out:
            # Find residual readers by shape
            hidden = self._hidden()
            if hidden is not None:
                writers = {id(w) for w in self.layer_writers(layer)}
                out = [m for m in layer.modules()
                       if _reads_residual(m, hidden) and id(m) not in writers]
        seen, uniq = set(), []
        for m in out:
            if m is not None and id(m) not in seen:
                seen.add(id(m))
                uniq.append(m)
        return uniq

    def uses_post_norm(self) -> bool:
        # Detect post-norm transformer sandwiches
        layers = self.layers()
        if not layers:
            return False
        L = layers[0]
        return any(hasattr(L, n) for n in ("post_feedforward_layernorm", "pre_feedforward_layernorm"))

    def ple_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        gate = getattr(layer, "per_layer_input_gate", None)
        return [gate] if gate is not None else []

    def ple_projection_writers(self, layer: torch.nn.Module) -> List[torch.nn.Module]:
        proj = getattr(layer, "per_layer_projection", None)
        return [proj] if proj is not None else []

    def has_ple(self) -> bool:
        return any(self.ple_writers(layer) for layer in self.layers())

    def ple_embed(self):
        dec = self._decoder()
        return getattr(dec, "embed_tokens_per_layer", None)

    def ple_model_projection(self):
        dec = self._decoder()
        return getattr(dec, "per_layer_model_projection", None)

    def is_moe(self) -> bool:
        layers = self.layers()
        return bool(layers) and len(self.mlp_writers(layers[len(layers) // 2])) > 1

    def writer_modules(self) -> List[torch.nn.Module]:
        mods = [self.embed()]
        for layer in self.layers():
            mods.extend(self.layer_writers(layer))
        return mods

    def can_edit_embed(self) -> bool:
        dec = self._decoder()
        return not (
            hasattr(dec, "embed_tokens_per_layer")
            or hasattr(dec, "per_layer_model_projection")
            or config_value(self.model.config, "vocab_size_per_layer_input") is not None
        )


# Auto-class preference order
_AUTO_LOADER_ORDER = (
    ("AutoModelForCausalLM", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"),
    ("AutoModelForImageTextToText", "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"),
    ("AutoModelForSeq2SeqLM", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES"),
)


def _resolve_model_loader(model_id: str, trust_remote_code: bool):
    """Pick the AutoModel* class that can load this model_id (see _AUTO_LOADER_ORDER)."""
    import transformers as tf
    from transformers import AutoConfig, AutoModel
    from transformers.models.auto import modeling_auto as ma

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_type = getattr(cfg, "model_type", None)
    auto_map = getattr(cfg, "auto_map", None) or {}
    if "AutoModelForCausalLM" in auto_map:  # remote-code models advertise their own loader
        return tf.AutoModelForCausalLM
    for cls_name, map_name in _AUTO_LOADER_ORDER:
        mapping = getattr(ma, map_name, {})
        if model_type in mapping and hasattr(tf, cls_name):
            return getattr(tf, cls_name)
    return AutoModel


def _safetensors_size_gb(model_id: str) -> float:
    """On-disk bf16 size of a (cached) model's .safetensors, in GB; 0 if not locally available.
    Used to decide whether the bf16 staging would OOM and we should stream the load instead."""
    try:
        import glob
        import os
        from huggingface_hub import snapshot_download
        d = snapshot_download(model_id, allow_patterns=["*.safetensors"], local_files_only=True)
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors"))) / 1e9
    except Exception:
        return 0.0


def load_model(cfg: ApostateConfig) -> ModelBundle:
    torch.manual_seed(cfg.seed)

    # Resolve and record the concrete device
    cfg.device = accel.resolve_device(cfg.device)
    accel.require_gpu(cfg.device)  # Backend-aware GPU error
    backend = accel.gpu_backend()

    # Reserve optional VRAM headroom
    import os as _os
    _frac = _os.environ.get("APOSTATE_VRAM_FRACTION")
    if _frac and cfg.device == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(float(_frac))
            _log(f"VRAM capped at {float(_frac)*100:.0f}% of the card (leaves headroom for the display)")
        except Exception as e:
            _log(f"could not set VRAM fraction ({_frac}): {e}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Validate local model directories early
    import os as _os
    if _os.path.exists(cfg.model):
        if _os.path.isfile(cfg.model):
            raise ValueError(f"--model must be the model directory, not a file: {cfg.model}. "
                             "Point it at the folder that contains config.json.")
        if not _os.path.exists(_os.path.join(cfg.model, "config.json")):
            raise ValueError(f"no config.json in {cfg.model}; point --model at the model directory.")

    # Resolve quantization before VRAM preflight
    use_4bit = bool(cfg.load_in_4bit) and cfg.device == "cuda"
    if use_4bit:
        # Preserve model-provided quantization
        try:
            from transformers import AutoConfig as _AC
            _pq = getattr(_AC.from_pretrained(cfg.model, trust_remote_code=True), "quantization_config", None)
        except Exception:
            _pq = None
        if _pq:
            _qm = _pq.get("quant_method") if isinstance(_pq, dict) else getattr(_pq, "quant_method", None)
            _log(f"model already quantized ({_qm or 'native'}); skipping bitsandbytes 4-bit, loading as-is")
            use_4bit = False
    if use_4bit:
        ok, why = accel.bitsandbytes_status()
        if not ok:
            # Fall back to full precision safely
            _log(f"4-bit requested but bitsandbytes is not usable here ({why}); "
                 f"falling back to {cfg.compute_dtype}. on ROCm, install a ROCm-enabled "
                 f"bitsandbytes or run with --no-load-in-4bit.")
            use_4bit = False
        elif backend == "rocm":
            from . import triton_nf4
            triton_nf4.patch_bnb_linear4bit()

    # Run safety checks before large allocations
    accel.gpu_smoke_test(cfg.device, log=_log)
    offload_gb = float(cfg.cpu_offload_gb) if cfg.cpu_offload_gb else 0.0
    if offload_gb > 0:
        # Allow VRAM spill into CPU RAM
        try:
            accel.maybe_preflight(
                cfg.device,
                model_id=cfg.model,
                load_in_4bit=use_4bit,
                compute_dtype=cfg.compute_dtype,
                batch_size=cfg.batch_size,
                log=_log,
            )
        except RuntimeError as e:
            _log(f"vram preflight: model exceeds GPU VRAM, but cpu_offload_gb={offload_gb:.0f} "
                 f"is set — spilling to CPU RAM. forward passes will be slower. ({e})")
    else:
        accel.maybe_preflight(
            cfg.device,
            model_id=cfg.model,
            load_in_4bit=use_4bit,
            compute_dtype=cfg.compute_dtype,
            batch_size=cfg.batch_size,
            log=_log,
        )

    tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    compute_dtype = _DTYPES[cfg.compute_dtype]
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if offload_gb > 0:
        # Distribute layers across GPU and CPU
        try:
            free_vram, _ = torch.cuda.mem_get_info()
            gpu_limit_gb = int(free_vram * 0.92 / 1e9)
            if _frac:
                total_vram = torch.cuda.get_device_properties(torch.device(cfg.device)).total_memory
                gpu_limit_gb = min(gpu_limit_gb, int(total_vram * float(_frac) / (1024 ** 3)))
            gpu_limit = f"{max(1, gpu_limit_gb)}GiB"
        except Exception:
            gpu_limit = "30GiB"
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: gpu_limit, "cpu": f"{offload_gb:.0f}GiB"}
        if use_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs["torch_dtype"] = compute_dtype
        _log(f"cpu offload: gpu_limit={gpu_limit}, cpu={offload_gb:.0f}GiB")
    elif use_4bit:
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

    model_loader = _resolve_model_loader(cfg.model, trust_remote_code=True)
    if model_loader is not AutoModelForCausalLM:
        _log(f"loading via {model_loader.__name__} (model_type not in CausalLM mapping)")

    from . import moe_nf4
    packed_moe = use_4bit and offload_gb <= 0 and moe_nf4.has_packed_experts(cfg.model)
    if packed_moe:
        # Stream packed MoE weights when CPU staging is unsafe
        import psutil as _psutil
        avail_gb = _psutil.virtual_memory().available / 1e9
        model_gb = _safetensors_size_gb(cfg.model)
        split = model_gb > 0 and model_gb > avail_gb * 0.85
        if split:
            # Stream NF4 experts directly to GPU
            _log(f"packed-MoE: {model_gb:.0f}GB bf16 won't fit {avail_gb:.0f}GB free RAM -> "
                 f"streaming load (NF4-quantize experts to GPU as the checkpoint is read)")
            model = moe_nf4.load_packed_moe_streaming(
                model_loader, cfg.model, cfg.device, compute_dtype, log=_log)
        else:
            _log("packed-MoE detected: loading on CPU (bf16), NF4-quantizing experts to GPU ...")
            model = model_loader.from_pretrained(
                cfg.model, torch_dtype=compute_dtype, low_cpu_mem_usage=True,
                device_map={"": "cpu"}, trust_remote_code=True)
        # Drop unused vision modules before quantization
        enc = _path_get(model, ("model", "encoder"))
        if enc is not None:
            for _vn in ("vision_tower", "embed_vision"):
                if getattr(enc, _vn, None) is not None:
                    setattr(enc, _vn, None)
                    _log(f"dropped {_vn} (text-only; frees VRAM)")
        if not split:
            # Quantize only CPU-staged experts
            moe_nf4.quantize_packed_experts(model, device=cfg.device, log=_log)
            moe_nf4.quantize_linears_4bit(model, device=cfg.device, log=_log)
            try:
                model.to(cfg.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                model.to(cfg.device)
        # Release transient quantization allocations
        torch.cuda.empty_cache()
        # Cap diffusion generation for evaluation
        if config_value(model.config, "model_type") == "diffusion_gemma":
            gc = getattr(model, "generation_config", None)
            if gc is not None and getattr(gc, "max_denoising_steps", None):
                gc.max_denoising_steps = min(gc.max_denoising_steps, int(cfg.eval_denoising_steps))
            set_config_value(model.config, "canvas_length",
                             min(int(config_value(model.config, "canvas_length") or 256), 32))
            # Strip unsupported use_cache arguments
            _orig_forward = model.forward
            def _forward_strip_use_cache(*a, _f=_orig_forward, **kw):
                kw.pop("use_cache", None)
                return _f(*a, **kw)
            model.forward = _forward_strip_use_cache
            _log(f"diffusion fast-eval: max_denoising_steps<={cfg.eval_denoising_steps}, canvas_length<=32; use_cache stripped")
    else:
        model = model_loader.from_pretrained(cfg.model, **kwargs)
    model.eval()
    model.requires_grad_(False)

    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(gen_cfg, attr):
                setattr(gen_cfg, attr, None)

    n_layers, hidden = model_metadata(model)
    return ModelBundle(model=model, tokenizer=tok, num_layers=n_layers, hidden_size=hidden)
