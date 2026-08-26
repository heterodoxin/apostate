from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import math


@dataclass
class ApostateConfig:
    method: str = "kcrn"
    model: str = "Qwen/Qwen3-8B"
    output_dir: str = "apostate-out"
    profile: str = "balanced"
    device: str = "auto"  # auto -> cuda/rocm if present, else mps/xpu/cpu (see apostate.accel)
    load_in_4bit: bool = True
    cpu_offload_gb: float = 0.0   # GB of model weights to spill to CPU RAM (0 = GPU-only)
    compute_dtype: str = "bfloat16"
    seed: int = 0
    resume: bool = False
    cache_activations: bool = True
    activation_cache_dir: Optional[str] = None

    harmful_path: Optional[str] = None
    harmless_path: Optional[str] = None
    harmful_test: Optional[str] = "mlabonne/harmful_behaviors:test:text|JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
    harmless_test: Optional[str] = "mlabonne/harmless_alpaca:test:text"
    refusal_eval_path: Optional[str] = "JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
    refusal_eval_n: int = 64
    kl_eval_path: Optional[str] = "mlabonne/harmless_alpaca:test:text"
    kl_eval_n: int = 48
    preserve_path: Optional[str] = None
    n_harmful: int = 600
    n_harmless: int = 600
    n_eval: int = 128  # Final validation set
    max_new_tokens: int = 32
    batch_size: int = 24  # sentinel default; with_defaults auto-scales this to detected VRAM (or force with --batch-size)
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
    max_rank: int = 3
    direction_layer_frac: float = 0.60
    direction_scope: str = "global"
    multi_refusal: bool = True
    multi_refusal_clusters: int = 6
    multi_refusal_min_norm: float = 0.08
    multi_refusal_min_separation: float = 0.05
    multi_refusal_min_coverage: float = 0.05
    # Keep reader refusal orthogonal to the harmless mean
    orthogonalize_direction: bool = True

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
    repair_steps: int = 4
    repair_candidates: int = 8
    repair_rerank_k: int = 5
    repair_probe_candidates: int = 20
    repair_probe_ref_n: int = 12
    repair_probe_kl_n: int = 16
    repair_probe_positions: int = 8
    repair_refusal_regress_slack: float = 0.01
    repair_stop_kl_frac: float = 0.80
    repair_min_alpha: float = 1e-3
    repair_min_kl_gain: float = 0.003
    repair_min_refusal_gain: float = 0.005
    repair_min_score_gain: float = 0.01
    repair_eval_n: int = 96
    repair_kl_n: int = 64
    refine_refusal_slack: float = 0.01
    final_zero_trim: bool = False
    final_push_bake_margin: float = 0.075

    guard_max_iters: int = 2
    guard_leakage_eps: float = 0.15
    guard_alpha_step: float = 0.25

    optimize: bool = True
    n_trials: int = 16
    adaptive_trials: bool = True
    kl_weight: float = 6.0
    kl_target: float = 0.04
    kl_target_weight: float = 18.0
    kl_quad_weight: float = 22.0
    kl_headroom_weight: float = 0.0
    kl_over_budget_weight: float = 72.0
    refusal_target_weight: float = 4.0
    refusal_quad_weight: float = 8.0
    kl_positions: int = 8
    opt_capability: bool = True
    opt_capability_weight: float = 2.5
    opt_capability_code_n: int = 8
    opt_capability_math_n: int = 8
    opt_eval_n: int = 32
    opt_gen_tokens: int = 32
    eval_denoising_steps: int = 8  # block-diffusion: denoising steps per eval generate (lower=faster)
    opt_objective: str = "generation"
    opt_rerank_k: int = 5
    opt_guard: bool = True
    opt_early_stop: bool = True
    opt_early_stop_margin: float = 0.02
    gemma_ple: bool = False
    gemma_query: bool = False
    ple_max_rank: int = 2

    prune: bool = False
    prune_max_frac: float = 0.25
    prune_kl: float = 0.04

    max_kl: float = 0.12
    target_refusal: float = 0.0

    # Preserve the harmless mean during ablation
    oblique_ablation: bool = True
    oblique_strength: float = 1.0  # 0 == symmetric, 1 == full mean-preserve
    oblique_denom_floor: float = 0.2  # min eig of U^T R; clamps strength when R aligns with mu
    # Limit oblique edits to residual writers
    oblique_writers_only: bool = True
    # Preserve harmless variance with a predictive co-vector
    oblique_predictive: bool = True
    predictive_ridge: float = 1e-2
    # Control predictive preservation strength
    oblique_preserve: float = 1.0
    # Suppress harmful response in the predictive co-vector
    oblique_contrast: float = 1.0

    # Reserve KL headroom for post-norm models
    reader_max_kl: float = 0.55
    reader_kl_target: float = 0.3
    reader_strengths: tuple = (2.0, 2.5, 2.75, 3.0, 4.0, 5.0)
    # Rank diffusion strengths with the encoder proxy
    reader_fast_proxy: bool = True
    reader_guard_rank: int = 3   # corrective directions the reader guard may add
    reader_margin_target: float = -1.0   # sweep stops once comply tokens win by this margin
    # Select reader strength at the refusal-KL knee
    reader_strength_kl_weight: float = 1.0

    save_dtype: str = "bfloat16"
    bake: bool = True

    kcrn_edits: Optional[str] = None
    kcrn_force: bool = False
    # Use NF4 for oversized fit and evaluation passes
    kcrn_load_in_4bit: bool = False
    kcrn_compute_dtype: str = "float16"
    kcrn_save_dtype: str = "float16"
    kcrn_fit_n: int = 0
    kcrn_harmful_fit_n: int = 0
    kcrn_benign_fit_n: int = 2048
    kcrn_eval_n: int = 0
    kcrn_calibration_eval_n: int = 96
    kcrn_pilot_n: int = 64
    kcrn_eval_generation: bool = True
    kcrn_layers: str = "all"
    kcrn_writers: str = "all"
    kcrn_target_writers: int = 0
    kcrn_key_rank: int = 16
    kcrn_harmful_rank: int = 16
    kcrn_preserve_rank: int = 128
    kcrn_refusal_rank: int = 1
    kcrn_refusal_source: str = "activation"
    kcrn_refusal_response_n: int = 128
    kcrn_refusal_response_tokens: int = 32
    kcrn_refusal_multi: bool = True
    kcrn_refusal_clusters: int = 0
    kcrn_refusal_min_norm_frac: float = 0.08
    kcrn_refusal_min_separation: float = 0.05
    kcrn_refusal_min_coverage: float = 0.05
    kcrn_strength: float = 4.0
    kcrn_ridge: float = 1e-6
    kcrn_solver: str = "projected"
    kcrn_benign_basis_mode: str = "raw"
    kcrn_benign_explained_variance: float = 1.0
    kcrn_basis_tolerance: float = 1e-7
    kcrn_max_key_samples: int = 512
    kcrn_min_projected_energy: float = 1e-4
    kcrn_max_relative_update: float = 10.0
    kcrn_all_positions: bool = True
    kcrn_kl_max_length: int = 768
    kcrn_max_delta_norm: float = 8.0
    kcrn_max_condition: float = 1000.0
    kcrn_aggressive_strengths: str = "1,2,4,6,8"
    kcrn_aggressive_max_steps: int = 24
    kcrn_aggressive_candidate_limit: int = 96
    kcrn_aggressive_probe_candidates: int = 8
    kcrn_aggressive_tune_n: int = 96
    kcrn_aggressive_scoring_harmful_n: int = 24
    kcrn_aggressive_calibration_n: int = 24
    kcrn_aggressive_calibration_positions: int = 8
    kcrn_aggressive_calibration_kl_budget: float = 0.02
    kcrn_aggressive_heldout_kl_max: float = 0.02
    kcrn_aggressive_max_cumulative_relative_update: float = 24.0
    kcrn_aggressive_min_margin_improvement: float = 0.01
    kcrn_aggressive_target_delivery: float = 0.90

    def with_defaults(self) -> "ApostateConfig":
        import os
        default_harmful_test = (
            "mlabonne/harmful_behaviors:test:text|"
            "JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
        )
        prof = (self.profile or "").lower()
        if (self.kcrn_solver or "").strip().lower() == "projected" and self.kcrn_benign_basis_mode == "legacy":
            self.kcrn_benign_basis_mode = "raw"
        if prof == "balanced":
            self.refine_deescalate = True
            if self.target_refusal <= 0.0 and self.opt_eval_n == 32:
                self.opt_eval_n = 64
            if self.target_refusal <= 0.0 and self.repair_eval_n == 96:
                self.repair_eval_n = 96
            # Use the validated Qwen3-8B KCRN operating point
            if (self.method or "kcrn").strip().lower() == "kcrn":
                if self.kcrn_preserve_rank == 128:
                    self.kcrn_preserve_rank = 64
                if self.kcrn_strength == 4.0:
                    self.kcrn_strength = 5.0
        elif prof in {"aggressive", "aggressive-kcrn", "aggressive kcrn"} and (self.method or "kcrn").strip().lower() == "kcrn":
            from .aggressive_kcrn import parse_strength_grid

            self.profile = "aggressive-kcrn"
            if (self.kcrn_solver or "").strip().lower() != "projected":
                raise ValueError("aggressive-kcrn requires --kcrn-solver projected")
            parse_strength_grid(self.kcrn_aggressive_strengths)
            for name in (
                "kcrn_aggressive_calibration_kl_budget",
                "kcrn_aggressive_heldout_kl_max",
                "kcrn_aggressive_max_cumulative_relative_update",
                "kcrn_aggressive_min_margin_improvement",
                "kcrn_aggressive_target_delivery",
            ):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
            if not 0 <= float(self.kcrn_aggressive_target_delivery) <= 1:
                raise ValueError("kcrn_aggressive_target_delivery must be between 0 and 1")
            if int(self.kcrn_aggressive_max_steps) < 0:
                raise ValueError("kcrn_aggressive_max_steps must be non-negative")
            if int(self.kcrn_aggressive_candidate_limit) < 1:
                raise ValueError("kcrn_aggressive_candidate_limit must be positive")
            if int(self.kcrn_aggressive_probe_candidates) < 1:
                raise ValueError("kcrn_aggressive_probe_candidates must be positive")
            if int(self.kcrn_aggressive_tune_n) < 1:
                raise ValueError("kcrn_aggressive_tune_n must be positive")
            if int(self.kcrn_aggressive_scoring_harmful_n) < 1:
                raise ValueError("kcrn_aggressive_scoring_harmful_n must be positive")
            if int(self.kcrn_aggressive_calibration_n) < 1:
                raise ValueError("kcrn_aggressive_calibration_n must be positive")
            if int(self.kcrn_aggressive_calibration_positions) < 1:
                raise ValueError("kcrn_aggressive_calibration_positions must be positive")
            if (self.kcrn_layers or "all").strip().lower() == "all":
                self.kcrn_layers = "auto"
            if (self.kcrn_writers or "all").strip().lower() == "all":
                self.kcrn_writers = "auto"
            if self.kcrn_preserve_rank == 128:
                self.kcrn_preserve_rank = 64
            if self.kcrn_strength == 4.0:
                self.kcrn_strength = 8.0
            if self.kcrn_max_delta_norm == 8.0:
                self.kcrn_max_delta_norm = 12.0
            if self.kcrn_max_relative_update == 10.0:
                self.kcrn_max_relative_update = 16.0
        elif prof == "fast":
            # Use the uniform fast preset
            if self.repair_steps == 10:
                self.repair_steps = 3
            if self.repair_rerank_k == 4:
                self.repair_rerank_k = 2
            if self.repair_eval_n == 48:
                self.repair_eval_n = 24
            if self.repair_probe_candidates == 24:
                self.repair_probe_candidates = 12
            if self.n_trials == 16:
                self.n_trials = 8
        model_l = (self.model or "").lower()

        # Expand search for large models
        _large_tags = ("27b", "28b", "32b", "34b", "35b", "40b", "70b", "72b", "65b", "123b")
        if any(t in model_l for t in _large_tags):
            if self.max_rank == 3:
                self.max_rank = 6
            if self.n_trials == 16:
                self.n_trials = 32
            if self.refusal_rank == 1:
                self.refusal_rank = 2
            if self.causal_floor == 0.10:
                self.causal_floor = 0.05
            # Trade KL headroom for lower refusal
            if self.kl_target == 0.04:
                self.kl_target = 0.10
            if self.max_kl == 0.12:
                self.max_kl = 0.18
            if self.kl_weight == 6.0:
                self.kl_weight = 2.0
            if self.kl_target_weight == 18.0:
                self.kl_target_weight = 6.0
            if self.kl_over_budget_weight == 72.0:
                self.kl_over_budget_weight = 20.0
            if self.kl_quad_weight == 22.0:
                self.kl_quad_weight = 8.0
            if self.kl_headroom_weight == 0.0:
                self.kl_headroom_weight = 12.0
            if self.repair_eval_n == 96:
                self.repair_eval_n = 64
            if self.opt_rerank_k == 5:
                self.opt_rerank_k = 3
            # Increase evaluation resolution for large models
            if self.opt_eval_n == 32:
                self.opt_eval_n = 64
            if self.target_refusal == 0.0:
                self.target_refusal = 0.02

        # Auto-enable 4-bit for large models
        _large_4bit_tags = ("20b", "22b", "24b", "26b", "27b", "28b", "30b", "32b", "34b",
                            "35b", "40b", "65b", "70b", "72b", "123b", "235b")
        if not self.load_in_4bit and any(t in model_l for t in _large_4bit_tags):
            self.load_in_4bit = True
        # Fall back to 4-bit when bf16 exceeds VRAM
        if not self.load_in_4bit and self.device in ("auto", "cuda"):
            try:
                from . import accel
                _dev = accel.resolve_device(self.device)
                _vram = accel.total_vram_gb(_dev)
                _n, _wb = accel.estimate_weight_footprint(
                    self.model, load_in_4bit=False, compute_dtype=self.compute_dtype)
                if _vram and _wb and _wb / 1e9 > _vram * 0.9:
                    self.load_in_4bit = True
            except Exception:
                pass

        # Scale the default batch to available VRAM
        if self.batch_size == 24 and self.device in ("auto", "cuda"):
            try:
                from . import accel
                _dev = accel.resolve_device(self.device)
                self.batch_size = accel.auto_batch_size(
                    self.model, load_in_4bit=self.load_in_4bit,
                    compute_dtype=self.compute_dtype, device=_dev, log=lambda *_a: None)
            except Exception:
                pass
        here = os.path.dirname(__file__)
        data = os.path.join(os.path.dirname(here), "data")
        refusal_cal = os.path.join(data, "refusal_calibration.txt")
        if self.harmful_path is None:
            self.harmful_path = "mlabonne/harmful_behaviors:train:text|" + os.path.join(data, "harmful.txt")
            if os.path.exists(refusal_cal):
                self.harmful_path = self.harmful_path + "|" + refusal_cal
        if self.harmful_test == default_harmful_test and os.path.exists(refusal_cal):
            self.harmful_test = self.harmful_test + "|" + refusal_cal
        if (
            self.refusal_eval_path == "JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
            and os.path.exists(refusal_cal)
        ):
            self.refusal_eval_path = self.refusal_eval_path + "|" + refusal_cal
        if self.harmless_path is None:
            self.harmless_path = "mlabonne/harmless_alpaca:train:text|" + os.path.join(data, "harmless.txt")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ApostateConfig":
        with open(path, "r", encoding="utf-8") as f:
            values = json.load(f)
        if "method" not in values:
            values["method"] = "legacy"
        return cls(**values)
