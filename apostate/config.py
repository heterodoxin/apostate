from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ApostateConfig:
    model: str = "Qwen/Qwen3-8B"
    output_dir: str = "apostate-out"
    profile: str = "balanced"
    device: str = "cuda"
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    seed: int = 0
    resume: bool = False
    cache_activations: bool = True
    activation_cache_dir: Optional[str] = None

    harmful_path: Optional[str] = None
    harmless_path: Optional[str] = None
    harmful_test: Optional[str] = "mlabonne/harmful_behaviors:test:text|JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
    harmless_test: Optional[str] = "mlabonne/harmless_alpaca:test:text"
    preserve_path: Optional[str] = None
    n_harmful: int = 600
    n_harmless: int = 600
    n_eval: int = 300
    max_new_tokens: int = 32
    batch_size: int = 24
    baseline_eval_n: int = 24
    head_sweep: bool = True
    head_sweep_min: float = 3.5
    head_sweep_max: float = 5.5
    head_sweep_step: float = 0.5
    head_sweep_top_k: int = 6
    head_sweep_probe_n: int = 8
    head_sweep_eval_n: int = 48
    head_sweep_probe_classifier: bool = False
    fit_response_activations: bool = False
    fit_response_n: int = 160
    fit_response_tokens: int = 32

    refusal_rank: int = 1
    variance_threshold: float = 0.90
    max_rank: int = 1
    direction_layer_frac: float = 0.60
    direction_scope: str = "global"

    causal_targeting: bool = True
    causal_floor: float = 0.10
    causal_temperature: float = 1.0

    preserve_rank: int = 8

    refine_refusal: bool = True
    refine_max_scale: float = 2.0
    refine_steps: int = 6
    refine_deescalate: bool = True
    refine_kl_steps: int = 10
    refine_scale_rerank_k: int = 2
    refine_kl_layer_steps: int = 10
    refine_kl_layer_candidates: int = 8
    repair_steps: int = 10
    repair_candidates: int = 10
    repair_rerank_k: int = 4
    repair_probe_candidates: int = 24
    repair_probe_ref_n: int = 12
    repair_probe_kl_n: int = 16
    repair_probe_positions: int = 8
    repair_refusal_regress_slack: float = 0.01
    repair_stop_kl_frac: float = 0.80
    repair_min_alpha: float = 1e-3
    repair_min_kl_gain: float = 0.003
    repair_min_refusal_gain: float = 0.005
    repair_min_score_gain: float = 0.01
    repair_eval_n: int = 48
    repair_kl_n: int = 64
    refine_refusal_slack: float = 0.01

    guard_max_iters: int = 2
    guard_leakage_eps: float = 0.15
    guard_alpha_step: float = 0.25

    optimize: bool = False
    n_trials: int = 16
    adaptive_trials: bool = True
    kl_weight: float = 3.0
    kl_target: float = 0.06
    kl_target_weight: float = 10.0
    kl_quad_weight: float = 14.0
    kl_over_budget_weight: float = 36.0
    refusal_target_weight: float = 4.0
    refusal_quad_weight: float = 8.0
    kl_positions: int = 32
    opt_capability: bool = True
    opt_capability_weight: float = 1.0
    opt_capability_code_n: int = 4
    opt_capability_math_n: int = 4
    opt_eval_n: int = 32
    opt_gen_tokens: int = 32
    opt_objective: str = "generation"
    opt_rerank_k: int = 3
    opt_guard: bool = True
    opt_early_stop: bool = True
    opt_early_stop_margin: float = 0.02
    gemma_ple: bool = False
    ple_max_rank: int = 2

    prune: bool = False
    prune_max_frac: float = 0.25
    prune_kl: float = 0.04

    max_kl: float = 0.16
    target_refusal: float = 0.03

    save_dtype: str = "bfloat16"
    bake: bool = True

    def with_defaults(self) -> "ApostateConfig":
        import os
        default_harmful_test = (
            "mlabonne/harmful_behaviors:test:text|"
            "JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
        )
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
        refusal_cal = os.path.join(data, "refusal_calibration.txt")
        if self.harmful_path is None:
            self.harmful_path = "mlabonne/harmful_behaviors:train:text|" + os.path.join(data, "harmful.txt")
        if self.harmful_test == default_harmful_test and os.path.exists(refusal_cal):
            self.harmful_test = self.harmful_test + "|" + refusal_cal
        if self.harmless_path is None:
            self.harmless_path = "mlabonne/harmless_alpaca:train:text|" + os.path.join(data, "harmless.txt")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ApostateConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
