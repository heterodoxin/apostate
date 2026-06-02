from __future__ import annotations

from typing import List
import torch

from .model import ModelBundle
from .projectors import ProjectionController
from .evaluate import refusal_logit_margin


def causal_layer_scores(
    bundle: ModelBundle,
    controller: ProjectionController,
    eval_instructions: List[str],
    batch_size: int = 16,
    floor: float = 0.25,
    temperature: float = 1.0,
) -> List[float]:
    controller.set_uniform_alpha(0.0)
    with controller.active():
        base = refusal_logit_margin(bundle, eval_instructions, batch_size)

    drops: List[float] = []
    for L in range(bundle.num_layers):
        controller.isolate_layer(L)
        with controller.active():
            m = refusal_logit_margin(bundle, eval_instructions, batch_size)
        drops.append(max(0.0, base - m))

    t = torch.tensor(drops)
    if float(t.max()) <= 1e-6:
        return [1.0] * bundle.num_layers

    t = t / t.max()
    if temperature != 1.0:
        t = t ** (1.0 / max(1e-3, temperature))
    alphas = floor + (1.0 - floor) * t
    return [float(x) for x in alphas]
