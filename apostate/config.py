from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ApostateConfig:
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
    n_eval: int = 128  # final test/validation reporting set; was 300 (the 68-min phase). model quality unaffected.
    max_new_tokens: int = 32
    batch_size: int = 24  # small models can push higher (--batch-size) but 24 is safe for the 27B roster on 34GB
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
    # reader-mode (post-norm/entangled models) only: keep only the refusal component
    # orthogonal to the harmless mean. collapses KL on gemma; hurts clean models (qwen),
    # so it is applied on the reader path, not the writer/optimizer path.
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

    # mean-preserving ablation E = I - Rbake U^T (U = R minus its harmless-mean component):
    # removes R fully but leaves the harmless mean, killing the mean-shift kl term. weight-only,
    # bakes into bias-free qwen2.5, distributes over MoE gates.
    oblique_ablation: bool = True
    oblique_strength: float = 1.0  # 0 == symmetric, 1 == full mean-preserve
    oblique_denom_floor: float = 0.2  # min eig of U^T R; clamps strength when R aligns with mu
    # writers only -- embed/lm-head live in a different space where the mid-layer mean is wrong.
    oblique_writers_only: bool = True
    # predictive co-vector D = R - W (W = harmless predictor of R^T x): preserves harmless
    # VARIANCE along R, not just the mean, so harmless KL drops. removes only the OOD excursion.
    # default on -- with the contrastive term below it strictly beats mean-preserving oblique.
    oblique_predictive: bool = True
    predictive_ridge: float = 1e-2
    # predictive-oblique preservation weight: D = R - preserve*W. 1.0 = full oblique,
    # 0.0 = orthogonal. lower when oblique under-ablates on entangled architectures (granite).
    oblique_preserve: float = 1.0
    # contrastive covector: penalize W for responding on the harmful set, so it preserves
    # harmless-SPECIFIC variance but drops variance shared with harmful (which smuggles refusal
    # back in on entangled archs). 0 = plain oblique. removes refusal at low KL. on by default:
    # granite 69%->5% (was stuck), qwen 11%->4% w/ lower KL -- strict win on entangled and clean.
    oblique_contrast: float = 1.0

    # post-norm models (reader-side ablation) need more kl headroom to decensor
    reader_max_kl: float = 0.55
    reader_kl_target: float = 0.3
    reader_strengths: tuple = (2.0, 2.5, 2.75, 3.0, 4.0, 5.0)
    # block-diffusion: rank strengths by the encoder-residual proxy (one forward, ~75x faster).
    reader_fast_proxy: bool = True
    reader_guard_rank: int = 3   # corrective directions the reader guard may add
    reader_margin_target: float = -1.0   # sweep stops once comply tokens win by this margin
    # reader strength pick = min(refusal + w*kl) among strengths under reader_max_kl, i.e. the
    # knee -- not max-ablation-under-budget, which overshoots (gemma: contrastive reader gives
    # 25%/0.126 at strength 3 but the greedy pick drove to 7 -> 0.44 kl). higher w favors low kl.
    reader_strength_kl_weight: float = 1.0

    save_dtype: str = "bfloat16"
    bake: bool = True

    def with_defaults(self) -> "ApostateConfig":
        import os
        default_harmful_test = (
            "mlabonne/harmful_behaviors:test:text|"
            "JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal"
        )
        prof = (self.profile or "").lower()
        if prof == "balanced":
            self.refine_deescalate = True
            if self.target_refusal <= 0.0 and self.opt_eval_n == 32:
                self.opt_eval_n = 64
            if self.target_refusal <= 0.0 and self.repair_eval_n == 96:
                self.repair_eval_n = 96
        elif prof == "fast":
            # uniform fast preset (all models): cheaper search, shallow repair, smaller eval
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

        # Large model auto-scaling: loosen search budget and raise rank ceiling.
        # 27B-72B models have more distributed refusal circuits than 7B-14B.
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
            # trade KL headroom for lower refusal on large models
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
            # large models set target_refusal=0.02 so the balanced-profile opt_eval_n bump
            # (which only fires when target_refusal<=0) never runs — fix it here explicitly.
            # at 32 samples the minimum measurable refusal is 1/32=3.1% > target, pure noise.
            if self.opt_eval_n == 32:
                self.opt_eval_n = 64
            if self.target_refusal == 0.0:
                self.target_refusal = 0.02

        # Auto-enable 4-bit for models too large to fit in bf16.
        # 20B+ can't fit 40GB+ of weights on typical single-GPU VRAM; 4-bit halves that.
        # 7B-14B fit natively in bf16 and run ~8x faster that way, so leave them alone.
        _large_4bit_tags = ("20b", "22b", "24b", "26b", "27b", "28b", "30b", "32b", "34b",
                            "35b", "40b", "65b", "70b", "72b", "123b", "235b")
        if not self.load_in_4bit and any(t in model_l for t in _large_4bit_tags):
            self.load_in_4bit = True

        if "gemma-4" in model_l or "gemma4" in model_l:
            # gemma 4 e4b is big; trim the batch so 4-bit fits a 16gb card. rank stays
            # at the default 1 (reader-side ablation, see model.uses_post_norm).
            if self.batch_size == 24:
                self.batch_size = 12
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
            return cls(**json.load(f))
