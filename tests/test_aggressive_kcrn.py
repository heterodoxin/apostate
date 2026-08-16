from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from apostate.aggressive_kcrn import (
    AggressiveEvaluation,
    AggressiveTrial,
    snapshot_writer,
    temporary_kcrn_factor,
    greedy_select,
    parse_strength_grid,
    shortlist_trials,
    merge_trial_stacks,
)


def _trial(target_key, strength, priority=0.0, relative_update_norm=1.0):
    return AggressiveTrial(
        target_key=target_key,
        strength=float(strength),
        edit={"kind": "kcrn", "target": target_key, "strength": float(strength)},
        certificate={"relative_update_norm": float(relative_update_norm)},
        priority=float(priority),
    )


def test_parse_strength_grid_is_finite_positive_sorted_and_unique():
    assert parse_strength_grid("4, 1, 4, 2") == (1.0, 2.0, 4.0)
    assert parse_strength_grid([3, 1, 2]) == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("spec", ["", "0,1", "1,nan", "-1", "inf"])
def test_parse_strength_grid_rejects_invalid_values(spec):
    with pytest.raises(ValueError):
        parse_strength_grid(spec)


def test_shortlist_trials_keeps_strength_grid_coverage():
    trials = [
        _trial((0, 0), 1.0, priority=5.0),
        _trial((0, 0), 4.0, priority=5.0),
        _trial((0, 1), 1.0, priority=4.0),
        _trial((0, 1), 4.0, priority=4.0),
        _trial((0, 2), 2.0, priority=3.0),
    ]

    selected = shortlist_trials(trials, (1.0, 2.0, 4.0), limit=3)

    assert [trial.strength for trial in selected] == [1.0, 2.0, 4.0]


def test_merge_trial_stacks_keeps_baseline_and_rejects_duplicate_targets():
    baseline = (_trial((0, 0), 8.0),)
    extra = (_trial((0, 1), 2.0),)

    assert merge_trial_stacks(baseline, extra) == baseline + extra
    assert merge_trial_stacks(baseline, ()) == baseline

    with pytest.raises(ValueError, match="repeats target"):
        merge_trial_stacks(baseline, (_trial((0, 0), 1.0),))


def test_greedy_select_uses_direct_delivery_when_available():
    initial = AggressiveEvaluation(refusal_margin=10.0, calibration_kl=0.0, harmful_delivery=0.0)

    def build(_accepted, remaining):
        for key in sorted(remaining):
            yield _trial(key, 1.0)

    def evaluate(stack):
        delivery = 0.1 if stack[0].target_key == (0, 0) else 0.4
        return AggressiveEvaluation(
            refusal_margin=0.0,
            calibration_kl=0.0,
            harmful_delivery=delivery,
        )

    result = greedy_select(
        [(0, 0), (0, 1)],
        (1.0,),
        initial,
        build,
        evaluate,
        max_steps=1,
        calibration_kl_budget=0.02,
        cumulative_relative_update_budget=10.0,
        min_margin_improvement=0.05,
    )

    assert [trial.target_key for trial in result.accepted] == [(0, 1)]
    assert result.final_evaluation.harmful_delivery == 0.4


def test_greedy_select_chooses_best_margin_reduction_and_excludes_targets():
    candidate_keys = [(0, 0), (0, 1)]
    initial = AggressiveEvaluation(refusal_margin=4.0, calibration_kl=0.0)
    seen_stacks = []

    def build(accepted, remaining):
        assert set(remaining).isdisjoint({trial.target_key for trial in accepted})
        for key in sorted(remaining):
            yield _trial(key, 1.0, priority=1.0 if key == (0, 0) else 2.0)

    improvements = {(0, 0): 1.0, (0, 1): 2.0}

    def evaluate(stack):
        seen_stacks.append(tuple(trial.target_key for trial in stack))
        reduction = sum(improvements[trial.target_key] for trial in stack)
        return AggressiveEvaluation(refusal_margin=4.0 - reduction, calibration_kl=0.0)

    result = greedy_select(
        candidate_keys,
        (1.0,),
        initial,
        build,
        evaluate,
        max_steps=1,
        calibration_kl_budget=0.02,
        cumulative_relative_update_budget=10.0,
        min_margin_improvement=0.1,
    )

    assert [trial.target_key for trial in result.accepted] == [(0, 1)]
    assert result.final_evaluation.refusal_margin == 2.0
    assert seen_stacks == [((0, 0),), ((0, 1),)]
    assert result.records[-1]["status"] == "accepted"


def test_greedy_select_applies_calibration_and_cumulative_safeguards():
    initial = AggressiveEvaluation(refusal_margin=2.0, calibration_kl=0.0)

    def build(_accepted, remaining):
        for key in sorted(remaining):
            yield _trial(key, 1.0, relative_update_norm=2.0)

    def evaluate(stack):
        if stack[0].target_key == (0, 0):
            return AggressiveEvaluation(refusal_margin=1.0, calibration_kl=0.03)
        return AggressiveEvaluation(refusal_margin=1.0, calibration_kl=0.01)

    result = greedy_select(
        [(0, 0), (0, 1)],
        (1.0,),
        initial,
        build,
        evaluate,
        max_steps=2,
        calibration_kl_budget=0.02,
        cumulative_relative_update_budget=2.0,
        min_margin_improvement=0.1,
    )

    assert [trial.target_key for trial in result.accepted] == [(0, 1)]
    assert any(record.get("reason") == "calibration_kl" for record in result.records)
    assert any(record.get("reason") == "cumulative_relative_update" for record in result.records)


def test_greedy_select_stops_when_no_trial_improves_margin():
    initial = AggressiveEvaluation(refusal_margin=1.0, calibration_kl=0.0)

    def build(_accepted, remaining):
        for key in remaining:
            yield _trial(key, 1.0)

    result = greedy_select(
        [(0, 0)],
        (1.0,),
        initial,
        build,
        lambda _stack: AggressiveEvaluation(refusal_margin=1.0, calibration_kl=0.0),
        max_steps=4,
        calibration_kl_budget=0.02,
        cumulative_relative_update_budget=10.0,
        min_margin_improvement=0.1,
    )

    assert list(result.accepted) == []
    assert result.final_evaluation == initial
    assert result.records[-1]["reason"] == "minimum_margin_improvement"


def test_greedy_select_rejects_nonfinite_evaluation():
    initial = AggressiveEvaluation(refusal_margin=1.0, calibration_kl=0.0)

    def build(_accepted, remaining):
        for key in remaining:
            yield _trial(key, 1.0)

    result = greedy_select(
        [(0, 0)],
        (1.0,),
        initial,
        build,
        lambda _stack: AggressiveEvaluation(refusal_margin=math.nan, calibration_kl=0.0),
        max_steps=1,
        calibration_kl_budget=0.02,
        cumulative_relative_update_budget=10.0,
        min_margin_improvement=0.1,
    )

    assert list(result.accepted) == []
    assert result.records[-1]["reason"] == "nonfinite_evaluation"


def test_temporary_linear_factor_restores_exact_storage():
    torch.manual_seed(61)
    writer = torch.nn.Linear(5, 4, bias=False).to(torch.float16)
    before = writer.weight.detach().clone()
    left = torch.randn(4, 2)
    right = torch.randn(2, 5)

    with temporary_kcrn_factor(writer, left, right):
        assert not torch.equal(writer.weight.detach(), before)

    assert torch.equal(writer.weight.detach(), before)


def test_snapshot_restore_handles_packed_writer_without_touching_other_parameters():
    class PackedWriter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Parameter(torch.randn(3, 4, 5).to(torch.float16))
            self.other = torch.nn.Parameter(torch.randn(2))

    torch.manual_seed(62)
    writer = PackedWriter()
    before = writer.down_proj.detach().clone()
    other = writer.other.detach().clone()
    snapshot = snapshot_writer(writer)
    writer.down_proj.data.add_(1)
    restore = snapshot
    from apostate.aggressive_kcrn import restore_writer

    restore_writer(restore)

    assert torch.equal(writer.down_proj.detach(), before)
    assert torch.equal(writer.other.detach(), other)


def test_runner_aggressive_branch_keeps_tuning_and_holdout_prompts_disjoint(monkeypatch, tmp_path):
    import apostate.kcrn_runner as runner
    from apostate.config import ApostateConfig

    calls = {}
    base_bundle = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        tokenizer=object(),
        num_layers=1,
        hidden_size=1,
    )
    edited_bundle = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        tokenizer=object(),
        num_layers=1,
        hidden_size=1,
    )

    monkeypatch.setattr(runner, "guard_model_and_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_model", lambda _cfg: base_bundle if not calls.get("loaded") else edited_bundle)
    monkeypatch.setattr(runner, "_release_bundle", lambda _bundle: None)
    monkeypatch.setattr(
        runner,
        "_resolve_disjoint_prompt_sets",
        lambda *_args: (["fit-0", "fit-1", "fit-2", "fit-3"], ["holdout-0", "holdout-1"]),
    )
    monkeypatch.setattr(
        runner,
        "_resolve_aggressive_prompt_sets",
        lambda *_args: (
            ["fit-0", "fit-1", "fit-2", "fit-3"],
            ["tune-0", "tune-1"],
            ["holdout-0", "holdout-1"],
        ),
    )

    cache_counter = {"value": 0}

    def fake_cache(*_args, **_kwargs):
        cache_counter["value"] += 1
        return [tmp_path / f"cache-{cache_counter['value']}.pt"]

    monkeypatch.setattr(runner, "_cache_full_position_logits", fake_cache)
    monkeypatch.setattr(runner, "_full_position_kl", lambda *_args, **_kwargs: 0.001)
    monkeypatch.setattr(runner, "bake", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "generate", lambda *_args, **_kwargs: ["delivered", "delivered"])
    monkeypatch.setattr(runner, "judge_strict_refusal", lambda *_args, **_kwargs: [False, False])

    def fake_fit(cfg, bundle, key_fit, fit_benign, tune_harmful, calibration_benign, cache):
        del cfg, bundle
        calls["fit_args"] = {
            "key_fit": list(key_fit),
            "fit_benign": list(fit_benign),
            "tune_harmful": list(tune_harmful),
            "calibration_benign": list(calibration_benign),
            "cache": list(cache),
        }
        return (
            [{"kind": "kcrn", "layer": 0, "writer_index": 0,
              "left": torch.ones(1, 1), "right": torch.ones(1, 1)}],
            [{"layer": 0, "writer_index": 0}],
            [{"layer": 0, "writer_index": 0, "relative_update_norm": 1.0}],
            {},
            {"accepted_steps": 1},
        )

    monkeypatch.setattr(runner, "_fit_aggressive_edits", fake_fit)
    def tracked_load(cfg):
        calls["loaded"] = calls.get("loaded", 0) + 1
        return base_bundle if calls["loaded"] == 1 else edited_bundle

    monkeypatch.setattr(runner, "load_model", tracked_load)
    cfg = ApostateConfig(
        profile="aggressive-kcrn",
        model="base",
        output_dir=str(tmp_path / "out"),
        harmful_path="harmful-fit",
        harmful_test="harmful-holdout",
        harmless_path="benign-fit",
        kl_eval_path="benign-holdout",
        n_harmful=4,
        n_harmless=4,
        n_eval=2,
        kcrn_harmful_fit_n=4,
        kcrn_benign_fit_n=4,
        kcrn_eval_n=2,
        kcrn_calibration_eval_n=2,
        batch_size=1,
        kcrn_eval_generation=True,
    )

    report = runner.run(cfg)

    assert calls["fit_args"]["tune_harmful"]
    assert set(calls["fit_args"]["tune_harmful"]).isdisjoint(calls["fit_args"]["key_fit"])
    assert set(calls["fit_args"]["tune_harmful"]).isdisjoint({"holdout-0", "holdout-1"})
    assert set(calls["fit_args"]["calibration_benign"]).isdisjoint({"holdout-0", "holdout-1"})
    assert report["variant"] == "aggressive-kcrn"
    assert report["aggressive_acceptance"]["success"] is True
