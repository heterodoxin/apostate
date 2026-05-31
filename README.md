# Apostate

Decensor instruction-tuned LLMs by ablating the refusal direction from the residual stream. No finetuning, no extra models, weights baked in place.

## Results

Qwen2.5-7B-Instruct, 4-bit, single RTX-class GPU:

| metric | base | apostate |
|--------|------|----------|
| refusal rate | 95% | 5% |
| harmless KL (nats) | 0 | 0.16 |
| wall time | N/A | 749s |
| disk overhead | N/A | 0 (in-place) |

Refusal is graded by a classifier (`protectai/distilroberta-base-rejection-v1`) instead of keyword matching. It counts soft refusals and deflections, not just "I can't help." Numbers come from a disjoint test split held out during the search: hyperparameters are tuned on a validation half, refusal and KL are reported on the test half. Eval prompts are from JailbreakBench + harmful_behaviors test (for refusal) and harmless_alpaca test (for KL). Uses a 12-trial TPE search over 28 layers with a rank-1 subspace.

## How it works

- Subspace: rank-1 mean difference of harmful vs harmless residuals.
- Causal targeting: per-layer ablation strength scored by single-layer patching.
- Preservation: Gram-Schmidt protected directions out of the edit.
- Guard: rechecks residual leakage, re-ablates where it reappears.
- Refine: escalates strength to drive refusal to target within a KL budget.
- Bake: folds the projection into fp16/bf16 weights, saves a standalone model.

TPE search minimizes both refusal and harmless KL. Edits are runtime hooks during the search, so every trial is a few forward passes with no reloads.

## Install

```
npm install
pip install torch transformers datasets safetensors optuna
```

## Use

```
apostate                 # tui
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate test  --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct
apostate talk  --model qwen-apostate --quant nf4
```

`talk --quant` picks the inference path: `bf16`/`fp16` (no quant, fastest if VRAM fits), `nf4`/`fp4`/`int8` (bitsandbytes, instant load), `gptq`/`marlin` (int4 Marlin kernel, fastest 4-bit on Ampere+, needs `pip install gptqmodel optimum` and quantizes on first load), `awq` (load a pre-quantized AWQ checkpoint).

## Data

Fit set blends `mlabonne/harmful_behaviors` + `mlabonne/harmless_alpaca` with bundled prompts (600 each). Held-out eval pulls real benchmarks (JailbreakBench, harmful_behaviors test, harmless_alpaca test). Sources are configurable via `repo:split:col` specs, `repo@config:split:col`, or `|`-joined lists.

## Requires

Python 3.10+, Node 18+, 16GB VRAM, CUDA.
