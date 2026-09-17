from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from apostate.config import ApostateConfig
from apostate import diode


HIDDEN = 3
WIDTH = 4


def _bundle():
    layers = []
    for _ in range(2):
        layers.append(
            SimpleNamespace(
                mlp=SimpleNamespace(
                    gate_proj=torch.nn.Linear(HIDDEN, WIDTH, bias=False),
                    up_proj=torch.nn.Linear(HIDDEN, WIDTH, bias=False),
                    down_proj=torch.nn.Linear(WIDTH, HIDDEN, bias=False),
                )
            )
        )
    decoder = SimpleNamespace(layers=layers, config=SimpleNamespace(intermediate_size=WIDTH))
    model = SimpleNamespace(language_model=decoder, config=SimpleNamespace(intermediate_size=WIDTH))
    bundle = SimpleNamespace(model=model, num_layers=len(layers))
    detector = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    actuator = [torch.tensor([3.0, 4.0, 5.0]), torch.tensor([6.0, 7.0, 8.0])]
    theta = [0.0, 0.0]
    cd = [0, 0]
    m = [1.0, 1.0]
    return bundle, detector, actuator, theta, cd, m


def _weights(bundle):
    return [
        (
            layer.mlp.gate_proj.weight.detach().clone(),
            layer.mlp.up_proj.weight.detach().clone(),
            layer.mlp.down_proj.weight.detach().clone(),
        )
        for layer in bundle.model.language_model.layers
    ]


def test_overwrite_mode_replaces_last_neuron_without_changing_width():
    bundle, detector, actuator, theta, cd, m = _bundle()
    before = _weights(bundle)
    cfg = ApostateConfig(diode_additive=False, diode_strength=2.0, diode_kappa=4.0)

    written = diode._bake(
        bundle,
        cfg,
        band={0},
        rmul=1.0,
        detector=detector,
        actuator=actuator,
        theta=theta,
        cd=cd,
        m=m,
    )

    assert written == 1
    assert bundle.model.config.intermediate_size == WIDTH
    assert bundle.model.language_model.config.intermediate_size == WIDTH
    after = _weights(bundle)
    assert after[0][0].shape == (WIDTH, HIDDEN)
    assert torch.equal(after[0][0][:-1], before[0][0][:-1])
    assert torch.equal(after[0][1][:-1], before[0][1][:-1])
    assert torch.equal(after[0][2][:, :-1], before[0][2][:, :-1])
    assert torch.equal(after[1][0], before[1][0])
    assert torch.equal(after[1][1], before[1][1])
    assert torch.equal(after[1][2], before[1][2])
    expected_gate, expected_up, expected_down = diode._neuron_rows(
        cfg, 1.0, detector[0], actuator[0], theta[0], cd[0], m[0], HIDDEN
    )
    assert torch.equal(after[0][0][-1], expected_gate)
    assert torch.equal(after[0][1][-1], expected_up)
    assert torch.equal(after[0][2][:, -1], expected_down)


def test_additive_mode_appends_a_real_neuron_and_zero_padding_neurons():
    bundle, detector, actuator, theta, cd, m = _bundle()
    before = _weights(bundle)
    cfg = ApostateConfig(diode_additive=True, diode_strength=2.0, diode_kappa=4.0)

    written = diode._bake(
        bundle,
        cfg,
        band={0},
        rmul=1.0,
        detector=detector,
        actuator=actuator,
        theta=theta,
        cd=cd,
        m=m,
    )

    assert written == 1
    assert bundle.model.config.intermediate_size == WIDTH + 1
    assert bundle.model.language_model.config.intermediate_size == WIDTH + 1
    after = _weights(bundle)
    assert after[0][0].shape == (WIDTH + 1, HIDDEN)
    assert torch.equal(after[0][0][:-1], before[0][0])
    assert torch.equal(after[0][1][:-1], before[0][1])
    assert torch.equal(after[0][2][:, :-1], before[0][2])
    expected_gate, expected_up, expected_down = diode._neuron_rows(
        cfg, 1.0, detector[0], actuator[0], theta[0], cd[0], m[0], HIDDEN
    )
    assert torch.equal(after[0][0][-1], expected_gate)
    assert torch.equal(after[0][1][-1], expected_up)
    assert torch.equal(after[0][2][:, -1], expected_down)
    assert torch.count_nonzero(after[1][0][-1]) == 0
    assert torch.count_nonzero(after[1][1][-1]) == 0
    assert torch.count_nonzero(after[1][2][:, -1]) == 0
