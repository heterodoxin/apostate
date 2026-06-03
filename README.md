![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

# Apostate

Apostate removes refusals from instruction-tuned language models by editing the weights, not by finetuning. It finds the direction in the residual stream that separates refused prompts from answered ones, projects that direction out of the layers that carry it, and bakes the result into a normal checkpoint you can load anywhere.

## What it does

The edit is a low-rank projection applied to the residual writers: the token embedding, each attention output projection, and each MLP down projection. For mixture-of-experts models it edits every expert's down projection plus the shared experts. There is no second model and no training step, so a run is a search over a 4-bit copy of the base model with forward hooks, and only the final baked weights touch disk.

## Reference run

Qwen2.5-7B-Instruct on one consumer GPU, 4-bit nf4, 12 search trials over 28 layers:

| metric | base | edited |
|---|---:|---:|
| refusal rate | 95% | 5% |
| harmless KL (nats) | 0 | 0.16 |
| wall time | n/a | 749s |

Refusal is graded by a classifier (`protectai/distilroberta-base-rejection-v1`), so deflections and soft refusals count, not just "I can't help". The numbers come from a held-out test split. The search tunes on a validation half and reports refusal and KL on a disjoint test half. Harmful prompts are JailbreakBench plus the harmful_behaviors test set; harmless prompts are the harmless_alpaca test set.

## How the search works

1. Collect harmful and harmless activations from the base model.
2. Take the rank-1 mean difference as the refusal direction, then remove the benign principal directions from it so the edit leaves general behavior alone.
3. Score each layer by ablating it on its own and measuring the refusal drop. That gives a per-layer strength prior.
4. Search the layer band, strength, direction layer, and embedding scale with TPE, minimizing refusal and harmless KL together, plus a small capability term measured on a code and math probe.
5. Recheck for reconstruction leakage and add corrective directions where refusal comes back.
6. Scale the passing edit back down until refusal sits just under target. That claws back KL the search did not need to spend.
7. Fold the projection into the residual-writer weights and save a standalone checkpoint.

## KL

KL is the token-distribution shift on harmless prompts, in nats, measured over the last few positions. The base logits are cached per prompt, so each trial only reruns the edited pass. The objective keeps KL under a budget (default `max_kl` 0.18, `kl_target` 0.08) with a quadratic penalty past the target and a hard penalty past the budget. A smaller preserve rank or a tighter causal floor trades a little refusal headroom for lower KL.

## Install

Apostate is pure Python. The TUI uses Textual, so there is no Node dependency.

```
apostate setup
```

The wizard installs the Python deps and checks the GPU. To do it by hand, `pip install -r requirements.txt`. For an `apostate` command on your PATH, `pip install -e .`; otherwise launch it with the bundled `apostate.cmd` on Windows or `python -m apostate` anywhere.

## Use

```
apostate                                            menu
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate ablate --model ... --out ... --resume     reuse cached activations
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite gsm8k
apostate talk --model qwen-apostate                chat
apostate talk --model qwen-apostate --backend vllm serve through vLLM
```

`talk` defaults to auto quant: it loads bf16 when the model fits in free VRAM and 4-bit otherwise, so a 7B on a 16GB card loads 4-bit. The vLLM backend serves at higher throughput and can quantize the KV cache (`--kv-cache-dtype`, including the TurboQuant modes) to stretch the context further. On Windows vLLM runs inside WSL and installs itself there on first use.

## Benchmarks

`apostate test --suite` runs `humaneval`, `mbpp`, `gsm8k`, the `refusal` suite, or `all`. Each reports refusal (classifier-judged by default, `--judge keyword` for the old string matching), pass@1 or accuracy where it applies, and harmless KL, all as base-versus-edited deltas. `deepswe` falls back to a local humaneval run when the dataset is not present.

## Architectures

Detected from the module layout. Dense and MoE are handled the same way.

| family | modes |
|---|---|
| Llama 2/3 | dense |
| Qwen2 / 2.5 / 3 | dense, MoE |
| Mistral, Mixtral | dense, MoE |
| DeepSeek, DeepSeek-V2 | dense, MoE |
| Gemma, Gemma 2 | dense |
| Phi-3, GPT-NeoX, Pythia | dense |

Falcon, Phi-2, and GPT-2 style Conv1D models are not fully covered yet. State-space models like Mamba and RWKV have no attention or MLP residual writers and are out of scope.

## Data

The fit set mixes `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca` with the prompts in `data/`, 600 of each. Eval pulls JailbreakBench and the harmful_behaviors and harmless_alpaca test splits. Point any of these at your own data with a `repo:split:col` spec, a `repo@config:split:col` spec, or several sources joined with `|`.

## Outputs

A run writes `report.json` and a readable `report.md` with the metrics, best params, layer alphas, and deltas, alongside the full config, a model-card `README.md`, and the activation cache that `--resume` reuses.

## Requires

Python 3.10+, CUDA, and about 16GB of VRAM for a 7B.
