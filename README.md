![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

# Apostate

Apostate edits instruction-tuned causal language models by finding a residual-stream refusal direction, removing preserved benign directions from it, and folding the resulting projection into model weights. The output is a normal Transformers checkpoint. There is no runtime hook, adapter stack, or finetune dependency after bake.

The edit touches modules that write back into the residual stream: token embeddings when safe, attention output projections, MLP down projections, MoE expert down projections, shared experts, and Gemma 4 text-decoder residual projections. Dense and MoE models go through the same controller path.

## Downloads

- [Qwen2.5 7B Instruct Apostate](https://huggingface.co/heterodoxin/qwen2.5-7b-instruct-apostate)

Gemma 4 E4B is not listed as a passing release. The stale HF repo at
`heterodoxin/gemma-4-e4b-it-apostate` failed the 2026-06-02 staging smoke test:
classifier refusal `25.0%`, strict refusal or weak noncompliance `50.0%`,
weak nonanswer `25.0%`, and helpful-style starts `100.0%` on four manual prompts.
The prompt-mean PLE probe found a nonzero packed PLE direction (`6.6467`) but did
not move refusal at alphas `0.05` through `1.6`; KL rose from `0.0039` to `0.3142`.

## Current Numbers

Qwen2.5-7B-Instruct was run on an RTX 4070 Ti SUPER with 4-bit NF4 load, seed `0`, 16 optimization trials, HumanEval `n=80`, MBPP `n=80`, GSM8K `n=24`, JBB refusal `n=48`, and KL over 48 harmless prompts.

| model | refusal | complied | humaneval | mbpp | gsm8k | kl | ablation wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 100.0% | 0.0% | 73.8% | 70.0% | 70.8% | 0.000 | n/a |
| apostate | 4.2% | 93.8% | 80.0% | 70.0% | 70.8% | 0.143 | 306.8s |
| heretic | 8.3% | 87.5% | 72.5% | 72.5% | 70.8% | 0.099 | 1166.7s |

This is a same-budget comparison against Heretic `1.3.0`, not Heretic's 200-trial default. Apostate exported a baked checkpoint. Heretic exported a PEFT LoRA adapter. Full commands and raw counts are in `docs/benchmark-runs/2026-05-31-heretic-head-to-head.md`.

## Method

Apostate collects harmful and harmless activations from the base model. It forms a low-rank basis from the harmful-minus-harmless mean and treats that basis as the refusal subspace. Preservation directions from harmless prompts, or from `--preserve-path`, are removed before the projection is scored.

Layer strength is measured instead of guessed. The runner temporarily ablates one layer at a time, records how much refusal behavior moves, and uses that response curve as the alpha prior. The search then scores direction layer, rank, layer band, strength, causal mix, causal sharpness, embedding strength, and head strength when the architecture supports it.

Balanced mode targets low refusal first, then pulls KL back down with global alpha scaling and layer trimming. Repair passes add corrective directions only when they improve the refusal/KL tradeoff. Capability drift is penalized with canonical-answer logprob probes on small math and code items, then checked again with public benchmarks.

The main optimization target combines classifier-judged refusal rate, a weak-response guard, harmless-token KL, penalty above `kl_target`, penalty above `max_kl`, and cheap capability drift. The weak-response guard is used during optimization so short answers, deflections, safety lectures, and generic overviews do not count as solved harmful prompts. Public benchmark refusal scoring uses `protectai/distilroberta-base-rejection-v1` by default and reports weak/noncompliance rates separately. Keyword refusal scoring remains available with `--judge keyword`.

## Install

```bash
apostate setup
```

Manual install:

```bash
npm install
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install transformers datasets safetensors optuna bitsandbytes
```

`apostate setup` installs Node dependencies, Python dependencies, CUDA Torch from the PyTorch `cu128` wheel index on NVIDIA systems, and checks GPU visibility.

## Ablate

```bash
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate --resume
```

`--resume` reuses activation cache files after an interrupted run. A finished run writes `report.json`, `report.md`, `apostate_config.json`, a checkpoint `README.md`, and any `activation_cache/*.pt` files used by resume.

Balanced defaults are `target_refusal=0.03`, `kl_target=0.06`, `max_kl=0.16`, `preserve_rank=8`, `refine_deescalate=true`, `refine_kl_steps=10`, `refine_kl_layer_steps=10`, and `repair_steps=10`. The hard cap is `max_kl`; `kl_target` is the pressure point used during search.

## Benchmark

```bash
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite humaneval
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite humaneval,gsm8k,refusal
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite all
```

Suites are `humaneval`, `mbpp`, `gsm8k`, `refusal`, or `all`. The TUI benchmark screen is a multi-select list: Space toggles a suite and Enter runs the selected set. DeepSWE is not listed.

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

Default fit data combines `mlabonne/harmful_behaviors` train prompts, `mlabonne/harmless_alpaca` train prompts, and local prompt files under `data/`. Held-out eval uses `mlabonne/harmful_behaviors` test, JailbreakBench behaviors, `mlabonne/harmless_alpaca` test, and the local refusal calibration set.

Custom data specs use `repo:split:col`, `repo@config:split:col`, or several sources joined with `|`. Local text files are accepted.

## Model Coverage

Model support is detected from module layout. Current coverage includes Llama 2/3, Qwen2/2.5/3, Mistral, Mixtral, DeepSeek, Gemma/Gemma2/Gemma 4 text decoders including `google/gemma-4-E4B`, Phi-3, GPT-NeoX, Pythia, OPT-style decoder stacks, and MPT-style block stacks.

Multimodal wrapper models are supported for the text path when Transformers exposes a causal language decoder inside the model object. Image and audio pipelines are not edited yet.

Gemma 4 uses per-layer embeddings and shared KV layers. Apostate can inspect and edit the nested text decoder, but Gemma 4 E4B is still under architecture work. Current passing download coverage is Qwen2.5.

## Requirements

Use Python 3.10+, Node 18+, CUDA Torch, Transformers, Datasets, Safetensors, Optuna, BitsAndBytes, and enough VRAM for the selected model. A 7B NF4 run expects about 16 GB VRAM.
