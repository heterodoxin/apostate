import math
from types import SimpleNamespace

import pytest
import torch

from apostate.bake import _edit_kcrn_writer
from apostate.activations import _last_token_rows
from apostate.kcrn import (
    KCRNDegeneracyError,
    collect_writer_inputs,
    key_basis,
    key_conditional_nulling,
    key_conditional_nulling_projected,
    orthonormal_basis,
    select_writer_targets,
)
from apostate.kcrn_runner import (
    _cache_full_position_logits,
    _chunked_pointwise_kl,
    _full_position_kl,
    _head_refusal_basis,
    _refusal_basis,
    masked_kl_from_logits,
    parse_index_spec,
    split_prompt_sets,
    target_priority,
)


def test_projected_solver_erases_refusal_and_preserves_benign_keys():
    torch.manual_seed(41)
    W = torch.randn(8, 13)
    R = torch.linalg.qr(torch.randn(8, 2), mode="reduced").Q
    harmful = torch.linalg.qr(torch.randn(13, 3), mode="reduced").Q
    benign = torch.linalg.qr(torch.randn(13, 8), mode="reduced").Q
    benign = benign - harmful @ (harmful.T @ benign)
    benign = torch.linalg.qr(benign, mode="reduced").Q

    update = key_conditional_nulling_projected(W, R, harmful, benign)
    edited = W + update.delta

    assert torch.linalg.norm(R.T @ edited @ harmful) < 1e-4
    assert torch.linalg.norm(update.delta @ benign) < 1e-4
    assert update.left.shape[1] == harmful.shape[1]


def test_projected_solver_reports_stable_metrics_for_nonorthogonal_benign_keys():
    torch.manual_seed(42)
    W = torch.randn(8, 13, dtype=torch.float64)
    R = torch.linalg.qr(torch.randn(8, 2, dtype=torch.float64), mode="reduced").Q
    harmful = torch.linalg.qr(torch.randn(13, 3, dtype=torch.float64), mode="reduced").Q
    benign_q = torch.linalg.qr(torch.randn(13, 4, dtype=torch.float64), mode="reduced").Q
    benign = benign_q @ torch.tensor(
        [[1.0, 0.2, 0.0], [0.0, 2.0, 0.1], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )

    update = key_conditional_nulling_projected(
        W,
        R,
        harmful,
        benign,
        strength=1.0,
        ridge=1e-10,
        min_projected_energy=1e-8,
        max_condition=1e8,
    )
    diagnostics = update.diagnostics

    assert diagnostics["benign_leakage"] < 1e-8
    assert diagnostics["harmful_fit_error"] < 1e-5
    assert diagnostics["refusal_residual"] < 1e-5
    assert 0.0 < diagnostics["projected_harmful_energy"] <= 1.0
    assert diagnostics["regularized_eigenvalues"]
    assert torch.linalg.norm(orthonormal_basis(benign).T @ (harmful - orthonormal_basis(benign) @ (orthonormal_basis(benign).T @ harmful))) < 1e-8


def test_projected_solver_reduces_to_normal_solver_without_benign_basis():
    torch.manual_seed(44)
    W = torch.randn(7, 11, dtype=torch.float64)
    R = torch.linalg.qr(torch.randn(7, 2, dtype=torch.float64), mode="reduced").Q
    harmful = torch.linalg.qr(torch.randn(11, 3, dtype=torch.float64), mode="reduced").Q
    empty = torch.empty(11, 0, dtype=torch.float64)

    normal = key_conditional_nulling(W, R, harmful, None, strength=0.75, ridge=1e-10)
    projected = key_conditional_nulling_projected(
        W,
        R,
        harmful,
        empty,
        strength=0.75,
        ridge=1e-10,
        min_projected_energy=1e-8,
    )

    assert torch.allclose(projected.delta, normal.delta, atol=1e-7, rtol=1e-7)


def test_projected_solver_rejects_overlapping_harmful_and_benign_spaces():
    W = torch.eye(5)
    R = torch.eye(5, 1)
    harmful = torch.eye(5, 2)
    benign = torch.eye(5, 2)

    with pytest.raises(KCRNDegeneracyError) as exc_info:
        key_conditional_nulling_projected(
            W,
            R,
            harmful,
            benign,
            min_projected_energy=1e-4,
        )
    assert exc_info.value.diagnostics["projected_harmful_energy"] == 0.0


def test_key_basis_supports_raw_and_explained_variance_modes():
    torch.manual_seed(45)
    keys = torch.randn(64, 10)

    raw = key_basis(keys, rank=5, mode="raw")
    pca = key_basis(keys, rank=5, mode="pca", explained_variance=0.5)

    assert raw.shape == (10, 5)
    assert pca.shape[0] == 10
    assert pca.shape[1] <= 5
    assert torch.allclose(raw.T @ raw, torch.eye(5), atol=1e-5, rtol=1e-5)
    assert torch.allclose(pca.T @ pca, torch.eye(pca.shape[1]), atol=1e-5, rtol=1e-5)


def test_general_solver_interpolates_harmful_and_benign_key_bases():
    torch.manual_seed(43)
    W = torch.randn(7, 11)
    R = torch.linalg.qr(torch.randn(7, 2), mode="reduced").Q
    harmful = torch.linalg.qr(torch.randn(11, 3), mode="reduced").Q
    benign = torch.linalg.qr(torch.randn(11, 3), mode="reduced").Q
    benign = benign - harmful @ (harmful.T @ benign)
    benign = torch.linalg.qr(benign, mode="reduced").Q

    update = key_conditional_nulling(W, R, harmful, benign)
    edited = W + update.delta

    assert torch.linalg.norm(R.T @ edited @ harmful) < 1e-4
    assert torch.linalg.norm(update.delta @ benign) < 1e-4


def test_kcrn_refusal_basis_supports_rank_one_compatibility_and_multi_direction_modes():
    torch.manual_seed(46)
    harmful = torch.cat((
        torch.randn(32, 6) + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.randn(32, 6) + torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
    ))
    benign = torch.randn(64, 6)
    cfg = SimpleNamespace(
        kcrn_refusal_rank=1,
        kcrn_refusal_multi=True,
        kcrn_refusal_clusters=0,
        kcrn_refusal_min_norm_frac=0.0,
        kcrn_refusal_min_separation=0.0,
        kcrn_refusal_min_coverage=0.0,
        seed=3,
    )

    rank_one = _refusal_basis(cfg, harmful, benign)
    cfg.kcrn_refusal_rank = 2
    rank_two = _refusal_basis(cfg, harmful, benign)

    expected = harmful.mean(0) - benign.mean(0)
    assert rank_one.shape == (6, 1)
    assert torch.allclose(rank_one[:, 0], expected, atol=1e-6, rtol=1e-6)
    assert rank_two.shape == (6, 2)
    assert torch.allclose(rank_two.T @ rank_two, torch.eye(2), atol=1e-5, rtol=1e-5)


def test_head_refusal_basis_uses_refusal_and_compliance_rows():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            normalized = text.strip().lower()
            refusal = normalized.startswith(("i", "sorry", "as", "unfortunately", "no", "apolog"))
            return [0 if refusal else 1]

    class Head:
        weight = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

    cfg = SimpleNamespace(kcrn_refusal_rank=1)
    bundle = SimpleNamespace(
        tokenizer=Tokenizer(),
        lm_head=lambda: Head(),
    )

    basis = _head_refusal_basis(cfg, bundle)

    assert torch.allclose(basis[:, 0], torch.tensor([2.0, -1.0, 0.0]))


def test_key_basis_is_orthonormal_and_bounded():
    torch.manual_seed(47)
    keys = torch.randn(20, 9)
    basis = key_basis(keys, rank=4)

    assert basis.shape == (9, 4)
    assert torch.linalg.norm(basis.T @ basis - torch.eye(4)) < 1e-5
    assert torch.linalg.norm(keys.mean(0) @ basis) > 0


def test_linear_and_packed_writers_receive_factorized_updates():
    torch.manual_seed(53)
    left = torch.randn(4, 2)
    right = torch.randn(2, 5)
    linear = torch.nn.Linear(5, 4, bias=False).to(torch.float16)
    before = linear.weight.detach().clone()

    _edit_kcrn_writer(linear, left, right)

    assert linear.weight.dtype == torch.float16
    assert torch.allclose(linear.weight.float(), before.float() + left @ right, atol=2e-3, rtol=2e-3)

    class PackedWriter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Parameter(torch.randn(3, 4, 5).to(torch.float16))

    packed = PackedWriter()
    packed_before = packed.down_proj.detach().clone()
    _edit_kcrn_writer(packed, left, right)

    expected = packed_before.float() + left @ right
    assert torch.allclose(packed.down_proj.float(), expected, atol=2e-3, rtol=2e-3)


def test_bake_reports_storage_dtype_benign_preservation_error():
    torch.manual_seed(54)
    left = torch.randn(4, 2)
    right = torch.randn(2, 5)
    linear = torch.nn.Linear(5, 4, bias=False).to(torch.float16)
    benign = torch.eye(5, 2)

    metrics = _edit_kcrn_writer(linear, left, right, benign)

    assert metrics["post_bake_preservation_error"] >= 0.0
    assert metrics["post_bake_preservation_relative"] >= 0.0
    assert metrics["ideal_preservation_error"] >= 0.0
    assert metrics["serialization_residual"] >= 0.0


def test_writer_input_capture_removes_hooks_and_keeps_cpu_keys():
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, _batch, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2], [2, 1]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.writer = torch.nn.Linear(3, 3, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            values = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            return self.writer(values)

    model = Model()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=Tokenizer(),
        num_layers=1,
        _kcrn_preformatted=True,
        layers=lambda: [model.writer],
        layer_writers=lambda layer: [layer],
    )

    captured = collect_writer_inputs(bundle, ["formatted"], batch_size=2)

    assert captured[0][0].shape == (2, 3)
    assert captured[0][0].device.type == "cpu"
    assert len(model.writer._forward_pre_hooks) == 0


def test_writer_input_capture_can_bound_all_position_key_memory():
    class Tokenizer:
        pad_token_id = 0

        def __call__(self, _batch, **_kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 1], [2, 1, 2]]),
                "attention_mask": torch.ones(2, 3, dtype=torch.long),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.writer = torch.nn.Linear(3, 3, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            values = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            return self.writer(values)

    model = Model()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=Tokenizer(),
        num_layers=1,
        _kcrn_preformatted=True,
        layers=lambda: [model.writer],
        layer_writers=lambda layer: [layer],
    )

    captured = collect_writer_inputs(
        bundle,
        ["formatted", "formatted-2", "formatted-3"],
        batch_size=1,
        all_positions=True,
        max_samples=2,
        seed=7,
    )

    assert captured[0][0].shape == (2, 3)


def test_raw_kl_uses_base_as_target_and_masks_padding():
    base = torch.zeros(2, 3, 3)
    edited = base.clone()
    edited[..., 0] = 1.0
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.bool)

    kl = masked_kl_from_logits(base, edited, mask, vocab_chunk_size=2)
    expected = math.log((math.e + 2.0) / 3.0) - 1.0 / 3.0

    assert math.isclose(kl, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_chunked_hidden_kl_matches_full_logits():
    torch.manual_seed(55)
    head = torch.nn.Linear(5, 17, bias=True)
    base_hidden = torch.randn(2, 4, 5)
    edited_hidden = base_hidden + 0.05 * torch.randn(2, 4, 5)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool)

    expected = masked_kl_from_logits(
        head(base_hidden),
        head(edited_hidden),
        mask,
        vocab_chunk_size=4,
    )
    pointwise = _chunked_pointwise_kl(
        base_hidden,
        edited_hidden,
        head,
        vocab_chunk_size=4,
    )
    actual = float((pointwise * mask).sum().item() / mask.sum().item())

    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)

    scaled_pointwise = _chunked_pointwise_kl(
        base_hidden,
        edited_hidden,
        head,
        vocab_chunk_size=4,
        logit_scale=0.5,
    )
    scaled_expected = masked_kl_from_logits(
        head(base_hidden) * 0.5,
        head(edited_hidden) * 0.5,
        mask,
        vocab_chunk_size=4,
    )
    scaled_actual = float((scaled_pointwise * mask).sum().item() / mask.sum().item())

    assert math.isclose(scaled_actual, scaled_expected, rel_tol=1e-6, abs_tol=1e-7)


def test_last_token_activation_ignores_right_padding():
    values = torch.tensor([
        [[1.0, 0.0], [2.0, 0.0], [9.0, 9.0]],
        [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    rows = _last_token_rows(values, mask)

    assert torch.equal(rows, torch.tensor([[2.0, 0.0], [5.0, 0.0]]))


def test_reload_kl_matches_cached_full_position_logits(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def __call__(self, prompts, **_kwargs):
            del prompts
            return {
                "input_ids": torch.tensor([[0, 1], [1, 2]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            }

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            values = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            return SimpleNamespace(logits=values @ self.weight.T)

    base = Model()
    edited = Model()
    edited.weight.data[0, 0] += 0.75
    bundle_base = SimpleNamespace(model=base, tokenizer=Tokenizer())
    bundle_edited = SimpleNamespace(model=edited, tokenizer=Tokenizer())
    instructions = ["one", "two"]

    paths = _cache_full_position_logits(bundle_base, instructions, 2, tmp_path, 16)
    cached = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert set(cached) == {"logits", "attention_mask"}
    actual = _full_position_kl(bundle_edited, instructions, 2, paths, 16)
    encoded = bundle_base.tokenizer(instructions)
    expected = masked_kl_from_logits(
        base(**encoded).logits,
        edited(**encoded).logits,
        encoded["attention_mask"].bool(),
    )

    assert len(paths) == 1
    assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_reload_kl_caches_hidden_states_for_causal_lm_models(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"]

        def __call__(self, prompts, **_kwargs):
            del prompts
            return {
                "input_ids": torch.tensor([[0, 1], [1, 2]]),
                "attention_mask": torch.tensor([[1, 1], [1, 0]]),
            }

    class Base(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

        def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
            del attention_mask, use_cache, return_dict
            values = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            return SimpleNamespace(last_hidden_state=values * self.scale)

    class Causal(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.model = Base(scale)
            self.lm_head = torch.nn.Linear(3, 5, bias=True)

        def get_base_model(self):
            return self.model

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = self.model(input_ids, attention_mask, use_cache).last_hidden_state
            return SimpleNamespace(logits=self.lm_head(hidden))

    base = Causal(1.0)
    edited = Causal(1.0)
    edited.load_state_dict(base.state_dict())
    edited.model.scale.data.add_(0.5)
    bundle_base = SimpleNamespace(model=base, tokenizer=Tokenizer())
    bundle_edited = SimpleNamespace(model=edited, tokenizer=Tokenizer())
    instructions = ["one", "two"]

    paths = _cache_full_position_logits(bundle_base, instructions, 2, tmp_path, 16)
    cached = torch.load(paths[0], map_location="cpu", weights_only=False)
    actual = _full_position_kl(bundle_edited, instructions, 2, paths, 16)
    encoded = bundle_base.tokenizer(instructions)
    base_logits = base(**encoded).logits
    edited_logits = edited(**encoded).logits
    expected = masked_kl_from_logits(
        base_logits,
        edited_logits,
        encoded["attention_mask"].bool(),
    )

    assert set(cached) == {"hidden_states", "attention_mask"}
    assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)


def test_index_parser_supports_ranges_and_all():
    assert parse_index_spec("0-2,5", 8, "layer") == [0, 1, 2, 5]
    assert parse_index_spec("all", 3, "layer") == [0, 1, 2]


def test_prompt_split_sets_are_disjoint_and_reproducible():
    prompts = [f"prompt-{index}" for index in range(12)]

    calibration_a, holdout_a = split_prompt_sets(prompts, 8, 4, seed=17)
    calibration_b, holdout_b = split_prompt_sets(prompts, 8, 4, seed=17)

    assert calibration_a == calibration_b
    assert holdout_a == holdout_b
    assert set(calibration_a).isdisjoint(holdout_a)
    assert len(calibration_a) == 8
    assert len(holdout_a) == 4


def test_cli_defaults_to_kcrn(monkeypatch):
    import apostate.cli as cli

    calls = []
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: calls.append(cfg.method))
    cli.main(["--model", "base", "--output-dir", "out"])

    assert calls == ["kcrn"]


def test_cli_dispatches_ccv_to_legacy_engine_with_predictive_oblique(monkeypatch):
    import apostate.cli as cli

    calls = []
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: calls.append(("kcrn", cfg)))
    monkeypatch.setattr(cli, "run_legacy", lambda cfg, command=None: calls.append(("legacy", cfg)))

    cli.main(["--method", "ccv", "--model", "base", "--output-dir", "out"])

    assert len(calls) == 1
    route, cfg = calls[0]
    assert route == "legacy"
    assert cfg.method == "ccv"
    assert cfg.oblique_ablation is True
    assert cfg.oblique_predictive is True


def test_cli_can_override_old_config_method(monkeypatch, tmp_path):
    import apostate.cli as cli

    config_path = tmp_path / "old-config.json"
    config_path.write_text('{"model": "base"}\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: calls.append(cfg.method))

    cli.main(["--config", str(config_path), "--method", "kcrn"])

    assert calls == ["kcrn"]


def test_subcommands_select_legacy_or_kcrn_engine(monkeypatch):
    import apostate.__main__ as main_module

    calls = []
    monkeypatch.setattr(
        main_module,
        "run_module",
        lambda args, label=None: calls.append((args, label)) or 0,
    )

    assert main_module.main(["ablate", "--model", "base", "--out", "legacy-out", "--resume"]) == 0
    assert main_module.main(["kcrn", "--model", "base", "--out", "kcrn-out"]) == 0
    assert main_module.main(["ccv", "--model", "base", "--out", "ccv-out"]) == 0
    assert calls[0][0][calls[0][0].index("--method") + 1] == "legacy"
    assert calls[1][0][calls[1][0].index("--method") + 1] == "kcrn"
    assert calls[2][0][calls[2][0].index("--method") + 1] == "ccv"


def test_tui_ablation_uses_kcrn_and_abliterated_output(monkeypatch):
    import apostate.tui as tui

    calls = []
    app = tui.Apostate.__new__(tui.Apostate)
    monkeypatch.setattr(app, "run_cli", lambda args: calls.append(args))

    app._do_ablate("/models/Qwen3-8B")

    assert calls == [["kcrn", "--model", "/models/Qwen3-8B", "--out", "Qwen3-8B-abliterated"]]


def test_kcrn_defaults_cover_all_writers_without_a_target_cap():
    from apostate.config import ApostateConfig

    cfg = ApostateConfig()

    assert cfg.method == "kcrn"
    assert cfg.kcrn_layers == "all"
    assert cfg.kcrn_writers == "all"
    assert cfg.kcrn_target_writers == 0
    assert cfg.kcrn_solver == "projected"
    assert cfg.kcrn_harmful_rank == 16
    assert cfg.kcrn_preserve_rank == 128
    assert cfg.kcrn_all_positions is True


def test_old_config_without_method_keeps_legacy_engine(tmp_path):
    from apostate.config import ApostateConfig

    path = tmp_path / "old-config.json"
    path.write_text('{"model": "base"}\n', encoding="utf-8")

    assert ApostateConfig.from_json(str(path)).method == "legacy"


def test_projected_solver_defaults_to_raw_benign_basis():
    from apostate.config import ApostateConfig

    cfg = ApostateConfig(kcrn_solver="projected")
    cfg.with_defaults()

    assert cfg.kcrn_benign_basis_mode == "raw"


def test_runner_exposes_activation_collection_for_fitting():
    import apostate.kcrn_runner as runner

    assert callable(runner.collect_activations)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a visible GPU")
def test_writer_selection_handles_cpu_statistics_and_gpu_weights():
    writer = torch.nn.Linear(3, 3, bias=False).to("cuda:0")

    class Bundle:
        num_layers = 1

        def layers(self):
            return [writer]

        def layer_writers(self, layer):
            return [layer]

    residual_h = torch.zeros(1, 4, 3)
    residual_b = torch.zeros(1, 4, 3)
    residual_h[:, :, 0] = 1.0
    pilot_h = {0: {0: torch.tensor([[1.0, 0.0, 0.0]])}}
    pilot_b = {0: {0: torch.tensor([[0.0, 1.0, 0.0]])}}

    targets = select_writer_targets(Bundle(), residual_h, residual_b, pilot_h, pilot_b)

    assert len(targets) == 1


def test_writer_selection_accepts_different_harmful_and_benign_sample_counts():
    writer = torch.nn.Linear(3, 3, bias=False)

    class Bundle:
        num_layers = 1

        def layers(self):
            return [writer]

        def layer_writers(self, layer):
            return [layer]

    residual_h = torch.zeros(1, 2, 3)
    residual_b = torch.zeros(1, 5, 3)
    residual_h[..., 0] = 1.0
    pilot_h = {0: {0: torch.tensor([[1.0, 0.0, 0.0]])}}
    pilot_b = {0: {0: torch.tensor([[0.0, 1.0, 0.0]])}}

    targets = select_writer_targets(Bundle(), residual_h, residual_b, pilot_h, pilot_b)

    assert len(targets) == 1


def test_target_priority_rejects_unstable_factor_certificates():
    assert target_priority(60.0, delta_norm=2.0, condition=5.0, max_delta_norm=8.0, max_condition=50.0) > 0
    assert target_priority(60.0, delta_norm=9.0, condition=5.0, max_delta_norm=8.0, max_condition=50.0) is None
    assert target_priority(60.0, delta_norm=2.0, condition=51.0, max_delta_norm=8.0, max_condition=50.0) is None
