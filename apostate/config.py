"""config"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ApostateConfig:
    # model io
    model: str = "Qwen/Qwen3-8B"
    output_dir: str = "apostate-out"
    profile: str = "balanced"                 # profile
    device: str = "cuda"
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"          # compute dtype
    seed: int = 0
    resume: bool = False                     # cache reuse
    cache_activations: bool = True
    activation_cache_dir: Optional[str] = None

    # data
    # fit pool
    harmful_path: Optional[str] = None
    harmless_path: Optional[str] = None
    # eval set
    harmful_test: Optional[str] = "mlabonne/harmful_behaviors:test:text|JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
    harmless_test: Optional[str] = "mlabonne/harmless_alpaca:test:text"
    preserve_path: Optional[str] = None        # protect dirs
    n_harmful: int = 600                       # contrast set size
    n_harmless: int = 600                      # contrast set size
    n_eval: int = 300                          # eval size
    max_new_tokens: int = 32                   # refusal signal tokens
    batch_size: int = 24                       # batch
    baseline_eval_n: int = 24                  # base eval
    fit_response_activations: bool = False     # response fit
    fit_response_n: int = 160                  # response count
    fit_response_tokens: int = 32              # response tokens

    # subspace
    refusal_rank: int = 1                      # mean diff
    variance_threshold: float = 0.90           # reserved
    max_rank: int = 1                          # rank cap
    direction_layer_frac: float = 0.60         # direction layer
    direction_scope: str = "global"            # scope

    # causal
    causal_targeting: bool = True
    causal_floor: float = 0.10                 # alpha floor
    causal_temperature: float = 1.0            # sharpen

    # preservation
    preserve_rank: int = 8                     # protected dims

    # refine
    refine_refusal: bool = True
    refine_max_scale: float = 2.0              # max scale
    refine_steps: int = 6
    refine_deescalate: bool = True             # kl shrink
    refine_kl_steps: int = 10                  # kl shrink
    refine_scale_rerank_k: int = 2             # scale exact
    refine_kl_layer_steps: int = 10            # layer trim
    refine_kl_layer_candidates: int = 8        # trim width
    repair_steps: int = 10                     # repair iters
    repair_candidates: int = 10                # repair width
    repair_rerank_k: int = 4                   # repair rerank
    repair_probe_candidates: int = 24          # probe cap
    repair_probe_ref_n: int = 12               # probe harmful
    repair_probe_kl_n: int = 16                # probe harmless
    repair_probe_positions: int = 8            # probe kl
    repair_refusal_regress_slack: float = 0.01 # ref regress
    repair_stop_kl_frac: float = 0.80          # stop kl
    repair_min_alpha: float = 1e-3             # alpha floor
    repair_min_kl_gain: float = 0.003          # kl gain
    repair_min_refusal_gain: float = 0.005     # ref gain
    repair_min_score_gain: float = 0.01        # score gain
    repair_eval_n: int = 48                    # repair harmful
    repair_kl_n: int = 64                      # repair harmless
    refine_refusal_slack: float = 0.01         # target slack

    # guard
    guard_max_iters: int = 2                   # guard iters
    guard_leakage_eps: float = 0.15            # leak threshold
    guard_alpha_step: float = 0.25             # alpha step

    # search
    optimize: bool = False                     # search profile
    n_trials: int = 16                         # trials
    adaptive_trials: bool = True               # adaptive
    kl_weight: float = 3.0                     # kl weight
    kl_target: float = 0.06                    # kl target
    kl_target_weight: float = 10.0             # target weight
    kl_quad_weight: float = 14.0               # curve weight
    kl_over_budget_weight: float = 36.0        # budget weight
    refusal_target_weight: float = 4.0         # refusal weight
    refusal_quad_weight: float = 8.0           # refusal curve
    kl_positions: int = 32                     # kl window
    opt_capability: bool = True                # cap loss
    opt_capability_weight: float = 1.0         # cap weight
    opt_capability_code_n: int = 4             # code n
    opt_capability_math_n: int = 4             # math n
    opt_eval_n: int = 32                       # prompts per trial
    opt_gen_tokens: int = 32                   # trial gen len
    opt_objective: str = "generation"          # generation judge
    opt_rerank_k: int = 3                      # rerank k
    opt_guard: bool = True                     # guard winner
    opt_early_stop: bool = True                # early stop
    opt_early_stop_margin: float = 0.02        # early stop margin

    # pruning
    prune: bool = False                        # layer drop
    prune_max_frac: float = 0.25               # cap layers dropped
    prune_kl: float = 0.04                     # prune budget

    # acceptance
    max_kl: float = 0.16                       # hard cap
    target_refusal: float = 0.03               # target refusal rate

    # output
    save_dtype: str = "bfloat16"
    bake: bool = True

    def with_defaults(self) -> "ApostateConfig":
        import os
        if (self.profile or "").lower() == "balanced":
            self.refine_deescalate = True
        model_l = (self.model or "").lower()
        if "gemma-4" in model_l or "gemma4" in model_l:
            if self.batch_size == 24:
                self.batch_size = 12
            if self.max_rank == 1:
                self.max_rank = 3
            if self.preserve_rank == 8:
                self.preserve_rank = 4
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
