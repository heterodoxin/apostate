"""causal layer scores."""

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
    """Return a per-layer alpha in [floor, 1.0]. Requires controller.R already set."""
    # baseline: no ablation
    controller.set_uniform_alpha(0.0)
    with controller.active():
        base = refusal_logit_margin(bundle, eval_instructions, batch_size)

    drops: List[float] = []
    for L in range(bundle.num_layers):
        controller.isolate_layer(L)
        with controller.active():
            m = refusal_logit_margin(bundle, eval_instructions, batch_size)
        drops.append(max(0.0, base - m))   # how much refusal fell when ablating only L

    t = torch.tensor(drops)
    if float(t.max()) <= 1e-6:
        # no measurable signal — fall back to uniform full strength
        return [1.0] * bundle.num_layers

    t = t / t.max()                         # normalize to [0, 1]
    if temperature != 1.0:
        t = t ** (1.0 / max(1e-3, temperature))
    alphas = floor + (1.0 - floor) * t      # map into [floor, 1]
    return [float(x) for x in alphas]
