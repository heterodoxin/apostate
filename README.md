# Apostate

Apostate edits instruction-tuned causal language models by finding a residual-stream direction associated with refusals and folding a projection against that direction into the model weights. The ablation pass does not finetune the model, does not add a second model, and writes a standalone Transformers checkpoint at the end.

The edit targets modules that write back into the residual stream: token embeddings, attention output projections, MLP down projections, and MoE expert down projections. Dense and MoE models use the same controller path; MoE layers apply the edit to every expert down projection plus shared experts when present.

## Reference Result

Qwen2.5-7B-Instruct on an RTX 4070 Ti SUPER, 4-bit NF4 load, seed `0`, 16 optimization trials. Public benchmark uses the classifier refusal judge, HumanEval `n=80`, MBPP `n=80`, GSM8K `n=24`, JBB refusal `n=48`, and KL over 48 harmless prompts.

| model | refusal | complied | humaneval | mbpp | gsm8k | kl | ablation wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 100.0% | 0.0% | 73.8% | 70.0% | 70.8% | 0.000 | n/a |
| apostate | 4.2% | 93.8% | 80.0% | 70.0% | 70.8% | 0.143 | 306.8s |
| heretic | 8.3% | 87.5% | 72.5% | 72.5% | 70.8% | 0.099 | 1166.7s |

This is a same-budget comparison against Heretic `1.3.0`, not Heretic's 200-trial default. Apostate exported a baked checkpoint; Heretic exported a PEFT LoRA adapter. Full commands, raw sample counts, and selected-trial details are in `docs/benchmark-runs/2026-05-31-heretic-head-to-head.md`.

## How It Works

Apostate first runs harmful and harmless prompts through the base model and records residual activations by layer. It computes a low-rank basis from the harmful-minus-harmless activation mean. That basis is treated as the refusal subspace.

Before applying the edit, Apostate removes benign preservation directions from the refusal basis. By default those preservation directions come from harmless activations. If `--preserve-path` is supplied, the custom preserve prompt set replaces the default harmless source.

Layer strength is not uniform unless causal targeting is disabled. Apostate measures how much each layer changes refusal behavior under a temporary single-layer ablation, then uses that as the per-layer alpha prior. The optimizer searches over direction layer, rank, layer band, strength, causal mix, causal sharpness, and embedding strength.

After search, the reconstruction guard remeasures refusal leakage and can add corrective directions. Balanced mode then tries to claw back KL: first with global alpha scaling, then by trimming layers that contribute high harmless KL while refusal remains under the target.

The final bake step folds the projection into the model's residual writer weights. The saved checkpoint loads normally through Transformers without runtime hooks.

## Optimization Target

The search objective combines four measured costs:

- classifier-judged refusal rate on harmful validation prompts
- harmless-token KL against the original model
- extra penalty above `kl_target`
- capability drift from canonical-answer logprob on small code/math samples

Balanced defaults use `target_refusal=0.03`, `kl_target=0.06`, `max_kl=0.16`, `preserve_rank=8`, `refine_deescalate=true`, `refine_kl_steps=10`, `refine_kl_layer_steps=10`, and `repair_steps=10`. The hard budget is still `max_kl`; the lower target exists so the search does not treat 0.16 as the desired landing zone.

KL is measured on harmless prompts by comparing original model logits to edited model logits over the final `kl_positions` tokens. The unit is nats. Base harmless logits are cached per prompt/window so repeated scoring does not recompute the unedited side.

## Install

```bash
apostate setup
```

The setup wizard installs Node dependencies, Python dependencies, and checks GPU visibility. On NVIDIA systems it installs CUDA Torch from the PyTorch `cu128` wheel index instead of letting PyPI pick a CPU-only build.

Manual install:

```bash
npm install
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install transformers datasets safetensors optuna bitsandbytes
```

## Ablate

```bash
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
```

Use `--resume` to reuse activation cache files after a failed or interrupted run:

```bash
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate --resume
```

Main output files:

- `report.json`: raw run metrics
- `report.md`: readable run report with params, alphas, guard history, and benchmark deltas
- `apostate_config.json`: full resolved config
- `README.md`: model card for the edited checkpoint
- `activation_cache/*.pt`: cached activations for `--resume`

## Benchmark

Benchmarks compare an edited model against its base model. Refusal scoring uses `protectai/distilroberta-base-rejection-v1` by default. Keyword scoring is still available with `--judge keyword`, but classifier judging is the public default.

```bash
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite humaneval
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite gsm8k
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite humaneval,gsm8k,refusal
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite all
```

The TUI benchmark screen is a multi-select list. Space toggles a suite, Enter runs the selected set.

HumanEval uses its `check(entry_point)` harness. MBPP uses its native assertion tests directly, so it does not get wrapped in the HumanEval checker.

| suite | measured fields |
|---|---|
| humaneval | refusal, pass@1, gsm8k, kl |
| mbpp | refusal, pass@1, gsm8k, kl |
| gsm8k | refusal, gsm8k, kl |
| refusal | refusal, compliance, category refusal, kl |
| all | humaneval, mbpp, gsm8k, refusal |

Benchmark output is written to `benchcode.json` and `benchcode.md`. If the candidate directory contains an Apostate `report.json`, the benchmark result is also merged into the candidate report and model card. PEFT adapter directories with `adapter_config.json` are loaded against their recorded base model, so adapter exports can be benchmarked without merging first.

## Chat

```bash
apostate talk --model qwen-apostate --quant nf4
apostate talk --model qwen-apostate --backend vllm --kv-cache-dtype turboquant_4bit_nc
```

`--quant` controls local weight loading: `auto`, `bf16`, `fp16`, `nf4`, `fp4`, `int8`, `gptq`, `marlin`, or `awq`.

`--kv-cache-dtype` is only for vLLM KV cache dtype. TurboQuant belongs there, not in weight quantization. Supported KV cache values include `auto`, `fp8`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, and `turboquant_3bit_nc`.

On Windows, vLLM runs through WSL. By default Apostate stops the WSL vLLM server when the chat session exits. Set `APOSTATE_KEEP_WSL=1` or pass `--no-shutdown-wsl` to keep it alive.

## Data

Default fit data combines `mlabonne/harmful_behaviors` train prompts, `mlabonne/harmless_alpaca` train prompts, and bundled local prompt files under `data/`. Held-out eval uses `mlabonne/harmful_behaviors` test, JailbreakBench behaviors, and `mlabonne/harmless_alpaca` test.

Custom data specs use `repo:split:col`, `repo@config:split:col`, or several sources joined with `|`. Local text files are accepted as sources.

Examples:

```text
mlabonne/harmful_behaviors:test:text
JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal
data/custom_harmful.txt|my_org/my_dataset:train:prompt
```

## Model Coverage

Supported families are detected from module layout, not from hardcoded model names. Current coverage includes Llama 2/3, Qwen2/2.5/3, Mistral, Mixtral, DeepSeek, Gemma/Gemma2/Gemma 4 text decoders including `google/gemma-4-E4B`, Phi-3, GPT-NeoX, Pythia, OPT-style decoder stacks, and MPT-style block stacks. Dense and MoE decoder stacks are supported when the residual writer modules can be located.

Multimodal wrapper models are supported for the text path when Transformers exposes a causal language decoder inside the model object. Gemma 4 E4B follows that pattern: Apostate edits `model.language_model` and reads layer count and hidden size from the nested text config. Image and audio input pipelines are not edited yet; use text prompts for ablation, benchmark, chat, and bake.

Gemma 4 uses per-layer embeddings. Apostate disables token-embedding edits for those models and defaults Gemma 4 runs to a smaller batch size, which avoids the high-KL embedding path and reduces desktop lag during search.

Architectures with nonstandard residual writers may need adapter support before the bake step is correct. State-space models without attention/MLP residual writers are outside the current edit path.

## Requirements

| dependency | minimum |
|---|---:|
| python | 3.10 |
| node | 18 |
| cuda | enabled |
| vram for 7b | 16gb |
| core packages | torch/cu128, torchvision/cu128, torchaudio/cu128, transformers, datasets, safetensors, optuna, bitsandbytes |
