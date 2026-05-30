"""config."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ApostateConfig:
    # ---- model / io ----
    model: str = "Qwen/Qwen3-8B"
    output_dir: str = "apostate-out"
    device: str = "cuda"
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"          # compute dtype
    seed: int = 0

    # ---- data ----
    # fit pool = real abliteration sets + bundled synthetic (set in with_defaults)
    harmful_path: Optional[str] = None
    harmless_path: Optional[str] = None
    # held-out eval = real benchmarks
    harmful_test: Optional[str] = "mlabonne/harmful_behaviors:test:text|JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
    harmless_test: Optional[str] = "mlabonne/harmless_alpaca:test:text"
    preserve_path: Optional[str] = None        # protect these directions
    n_harmful: int = 600                       # contrast set size
    n_harmless: int = 600                      # contrast set size
    n_eval: int = 300                          # held-out eval per side
    max_new_tokens: int = 32                   # refusal signal tokens
    batch_size: int = 24                       # batch

    # ---- (A) subspace ----
    refusal_rank: int = 1                      # mean-diff
    variance_threshold: float = 0.90           # reserved
    max_rank: int = 1                          # rank cap
    direction_layer_frac: float = 0.60         # direction layer
    direction_scope: str = "global"            # global | per_layer

    # ---- (B) causal targeting ----
    causal_targeting: bool = True
    causal_floor: float = 0.25                 # alpha floor
    causal_temperature: float = 1.0            # sharpen

    # ---- (C) preservation ----
    preserve_rank: int = 4                     # protected dims

    # ---- refusal-targeting refine (drives residual refusal to ~0 within KL budget) ----
    refine_refusal: bool = True
    refine_max_scale: float = 2.0              # max scale
    refine_steps: int = 6
    refine_deescalate: bool = False            # claw back kl (can raise refusal)

    # ---- (D) reconstruction guard ----
    guard_max_iters: int = 2                   # guard iters
    guard_leakage_eps: float = 0.15            # leak threshold
    guard_alpha_step: float = 0.25             # alpha step

    # ---- (E) automated optimization (TPE / random search) ----
    optimize: bool = False                     # search profile
    n_trials: int = 12                         # trials
    adaptive_trials: bool = True               # adaptive
    kl_weight: float = 1.0                     # kl weight
    opt_eval_n: int = 24                       # prompts per trial
    opt_gen_tokens: int = 32                   # trial gen len
    opt_objective: str = "generation"          # generation = honest classifier grading per trial
    opt_rerank_k: int = 3                      # rerank k
    opt_guard: bool = True                     # guard winner
    opt_early_stop: bool = True                # early stop
    opt_early_stop_margin: float = 0.02        # early stop margin

    # ---- layer pruning (faster generation; trades capability, off by default) ----
    prune: bool = False                        # drop redundant layers, capability-gated
    prune_max_frac: float = 0.25               # cap layers dropped
    prune_kl: float = 0.04                     # extra harmless-kl allowed from pruning

    # ---- acceptance ----
    max_kl: float = 0.30                       # reject above this kl
    target_refusal: float = 0.05               # target refusal rate

    # ---- output ----
    save_dtype: str = "bfloat16"
    bake: bool = True

    def with_defaults(self) -> "ApostateConfig":
        import os
        here = os.path.dirname(__file__)
        data = os.path.join(os.path.dirname(here), "data")
        if self.harmful_path is None:
            self.harmful_path = "mlabonne/harmful_behaviors:train:text|" + os.path.join(data, "harmful.txt")
        if self.harmless_path is None:
            self.harmless_path = "mlabonne/harmless_alpaca:train:text|" + os.path.join(data, "harmless.txt")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ApostateConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
