![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white)](https://discord.gg/JR6hMmJNuB)

# Apostate

Apostate edits instruction-tuned causal language models by finding a residual-stream refusal direction, removing preserved benign directions from it, and folding the resulting projection into model weights. The output is a normal Transformers checkpoint. There is no runtime hook, adapter stack, or finetune dependency after bake.

The edit touches modules that write back into the residual stream: token embeddings when safe, attention output projections, MLP down projections, MoE expert down projections, shared experts, and Gemma 4 text-decoder residual projections. Dense and MoE models go through the same controller path.

## Downloads

- [Qwen3.6 27B Apostate](https://huggingface.co/heterodoxin/qwen3.6-27b-apostate)
- [Qwen2.5 7B Instruct Apostate](https://huggingface.co/heterodoxin/qwen2.5-7b-instruct-apostate)

Gemma 4 E4B is not listed as a passing release. The stale HF repo at
`heterodoxin/gemma-4-e4b-it-apostate` failed the 2026-06-02 staging smoke test:
classifier refusal `25.0%`, strict refusal or weak noncompliance `50.0%`,
weak nonanswer `25.0%`, and helpful-style starts `100.0%` on four manual prompts.
The prompt-mean PLE probe found a nonzero packed PLE direction (`6.6467`) but did
not move refusal at alphas `0.05` through `1.6`; KL rose from `0.0039` to `0.3142`.
The 2026-06-02 shared-KV plus PLE smoke run also failed release criteria:
TEST refusal `58.3%`, harmless KL `0.109`, no bake, no upload.

## Current Numbers

**Qwen3.6-27B** — AMD Radeon RX 9700 (gfx1201, ROCm 7.1.1), 4-bit NF4, JBB refusal `n=48`, KL over 48 harmless prompts.

| model | refusal | complied | kl |
|---|---:|---:|---:|
| base | 95.8% | 4.2% | 0.000 |
| apostate | 8.3% | 87.5% | 0.159 |

**Qwen2.5-7B-Instruct** — RTX 4070 Ti SUPER, 4-bit NF4, seed `0`, 16 optimization trials, HumanEval `n=80`, MBPP `n=80`, GSM8K `n=24`, JBB refusal `n=48`, KL over 48 harmless prompts.

| model | refusal | complied | humaneval | mbpp | gsm8k | kl | ablation wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 100.0% | 0.0% | 73.8% | 70.0% | 70.8% | 0.000 | n/a |
| apostate | 4.2% | 93.8% | 80.0% | 70.0% | 70.8% | 0.143 | 306.8s |
| heretic | 8.3% | 87.5% | 72.5% | 72.5% | 70.8% | 0.099 | 1166.7s |

The Qwen2.5 comparison against Heretic `1.3.0` is same-budget, not Heretic's 200-trial default. Apostate exported a baked checkpoint; Heretic exported a PEFT LoRA adapter.

## Method

Apostate collects harmful and harmless activations from the base model. It forms a low-rank basis from the harmful-minus-harmless mean and treats that basis as the first refusal axis. When `multi_refusal=true`, it also searches for independent refusal axes from harmful clusters, harmful clusters against nearest harmless clusters, high-residual harmful tails, and residual SVD axes. Each new axis has to survive harmful/harmless separation and harmful-coverage filters before it is fused into the subspace. The optimizer can still choose rank 1 when extra axes cost too much KL.

Layer strength is measured instead of guessed. The runner temporarily ablates one layer at a time, records how much refusal behavior moves, and uses that response curve as the alpha prior. The search then scores direction layer, rank, layer band, strength, causal mix, causal sharpness, embedding strength, and head strength when the architecture supports it.

Balanced mode targets low refusal first, then pulls KL back down with global alpha scaling and layer trimming. Repair passes add corrective directions only when they improve the refusal/KL tradeoff. Capability drift is penalized with canonical-answer logprob probes on small math and code items, then checked again with public benchmarks.

The projection itself is oblique, not symmetric. A plain `I - R R^T` removal shifts the harmless residual mean along the refusal direction, and that mean shift is the dominant source of KL. Apostate instead removes `R` along a co-vector `U = R` minus its harmless-mean component, so the projector `E = I - Rbake U^T` still zeroes `R` (`E R = 0`) but leaves the harmless mean untouched (`E mu = mu`). The mean-shift KL term collapses while harmful inputs are still fully ablated. It is weight-only with no bias, so it bakes into bias-free checkpoints like Qwen2.5, and because it is a fixed linear map it composes correctly across MoE router gates. `oblique_strength` interpolates between symmetric (`0`) and full mean-preservation (`1`); the co-vector is clamped when `R` aligns too closely with the harmless mean.

`--oblique-predictive` goes one step further. Instead of the fixed mean-orthogonal co-vector it fits a per-layer ridge predictor `W` of the harmless `R`-projection from the rest of the activation and removes along `D = R - W`. The harmless `R`-projection is largely predictable from the other features, so `D·x` is near zero on harmless inputs and the harmless *variance* along `R` is preserved, not just the mean; the harmful refusal excursion is out-of-distribution for `W` and is still removed in full (`E R = 0`). This lowers the bf16 edit KL below the mean co-vector. The advantage is largest in full precision — NF4 requantization narrows it back toward the plain oblique — so it is most useful when the checkpoint is served in bf16.

The main optimization target combines classifier-judged refusal rate, a weak-response guard, harmless-token KL, penalty above `kl_target`, penalty above `max_kl`, and cheap capability drift. The weak-response guard is used during optimization so short answers, deflections, safety lectures, and generic overviews do not count as solved harmful prompts. Public benchmark refusal scoring uses `protectai/distilroberta-base-rejection-v1` by default and reports weak/noncompliance rates separately. Keyword refusal scoring remains available with `--judge keyword`.

## Install

```bash
apostate setup
```

Manual install:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install transformers datasets safetensors optuna bitsandbytes textual
```

`apostate setup` installs the Python dependencies, CUDA Torch from the PyTorch `cu128` wheel index on NVIDIA systems, and checks GPU visibility. The TUI is pure Python (Textual), so there is no Node dependency.

## Ablate

```bash
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate --resume
```

`--resume` reuses activation cache files after an interrupted run. A finished run writes `report.json`, `report.md`, `apostate_config.json`, a checkpoint `README.md`, and any `activation_cache/*.pt` files used by resume.

Balanced defaults are `target_refusal=0.0`, `kl_target=0.04`, `max_kl=0.12`, `kl_positions=8`, `preserve_rank=8`, `max_rank=3`, `multi_refusal=true`, `multi_refusal_min_coverage=0.05`, `refine_deescalate=true`, `refine_kl_steps=10`, `refine_kl_layer_steps=10`, `repair_steps=4`, and `oblique_ablation=true`. The hard cap is `max_kl`; `kl_target` is the pressure point used during search. Set `--oblique-ablation false` to fall back to the symmetric projection. Add `--oblique-predictive` to use the predictive co-vector (lower bf16 KL; see Method), paired with a small nonzero `--target-refusal` such as `0.05` since its strength is at moderate refusal.

## Benchmark

The benchmark path is built into the TUI. Open `apostate`, choose `Test`, pick the edited model and base model, then use the suite selector. Space toggles a suite and Enter runs the selected set.

Suites are `humaneval`, `mbpp`, `gsm8k`, `refusal`, or `all`. DeepSWE is not listed.

Benchmark output is written to `benchcode.json` and `benchcode.md`. If the candidate directory has an Apostate `report.json`, the benchmark result is merged into the candidate report and model card.

## Chat

```bash
apostate talk --model qwen-apostate --quant nf4
apostate talk --model qwen-apostate --backend vllm --kv-cache-dtype turboquant_4bit_nc
```

`--quant` controls local weight loading: `auto`, `bf16`, `fp16`, `nf4`, `fp4`, `int8`, `gptq`, `marlin`, or `awq`. `--kv-cache-dtype` is only for vLLM KV cache dtype. TurboQuant belongs there, not in weight quantization.

On Windows, vLLM runs through WSL. Apostate stops the WSL vLLM server when chat exits unless `APOSTATE_KEEP_WSL=1` or `--no-shutdown-wsl` is set.

## Model Selection

The TUI has separate model lists for ablation and chat/test. Ablation scans Hugging Face cache plus local checkpoints and hides Apostate variants. Chat/test scans local disks for Apostate checkpoints, including folders that do not start with `apostate`, while ignoring HF cache entries.

Use `APOSTATE_MODEL_ROOTS` to add scan roots. Values are separated with the platform path delimiter.

## Data

Default fit data combines `mlabonne/harmful_behaviors` train prompts, `mlabonne/harmless_alpaca` train prompts, and local prompt files under `data/`. Held-out eval uses `mlabonne/harmful_behaviors` test, JailbreakBench behaviors, `mlabonne/harmless_alpaca` test, and the local refusal calibration set. Balanced mode front-loads 64 refusal calibration prompts and 48 public harmless KL prompts into validation so the optimizer sees the same hard refusal and KL distribution used by the public report path.

Custom data specs use `repo:split:col`, `repo@config:split:col`, or several sources joined with `|`. Local text files are accepted.

## Model Coverage

Model support is detected from module layout. Current coverage includes Llama 2/3, Qwen2/2.5/3, Mistral, Mixtral, DeepSeek, Gemma/Gemma2/Gemma 4 text decoders including `google/gemma-4-E4B`, Phi-3, GPT-NeoX, Pythia, OPT-style decoder stacks, and MPT-style block stacks.

Multimodal wrapper models are supported for the text path when Transformers exposes a causal language decoder inside the model object. Image and audio pipelines are not edited yet.

Gemma 2/3/4 use a post-norm sandwich, so editing writer outputs gets renormalized away. Apostate detects this and switches to reader-side ablation: it projects the per-layer refusal direction out of the inputs of the modules that read the residual, mainly MLP gate/up paths and the per-layer input gate. Attention q/k/v is skipped because it added attention drift without reliable refusal gain. The edit still bakes cleanly into a standalone checkpoint. Gemma 4 E4B goes from about 85% refusal to roughly 5-15% this way, coherent and complying, at a higher KL than dense pre-norm models.

## Requirements

Use Python 3.10+, CUDA Torch, Transformers, Datasets, Safetensors, Optuna, BitsAndBytes, Textual, and enough VRAM for the selected model. A 7B NF4 run expects about 16 GB VRAM.
