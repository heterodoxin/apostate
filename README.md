# Apostate

Apostate edits instruction-tuned causal language models by finding a residual-stream direction associated with refusals and folding a projection against that direction into the model weights. The ablation pass does not finetune the model, does not add a second model, and writes a standalone Transformers checkpoint at the end.

The edit targets modules that write back into the residual stream: token embeddings, attention output projections, MLP down projections, and MoE expert down projections. Dense and MoE models use the same controller path; MoE layers apply the edit to every expert down projection plus shared experts when present.

## Reference Result

Qwen2.5-7B-Instruct, 4-bit NF4 load, 12 optimization trials, 600 harmful fit prompts, 600 harmless fit prompts, held-out validation/test split:

| metric | base | edited | delta |
|---|---:|---:|---:|
| refusal rate | 95.0% | 5.0% | -90.0 pts |
| harmless KL nats | 0.000 | 0.160 | +0.160 |
| wall time | n/a | 749s | n/a |
| disk overhead | n/a | 0 | n/a |

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

Balanced defaults use `kl_target=0.08`, `max_kl=0.18`, `preserve_rank=8`, `refine_deescalate=true`, `refine_kl_steps=8`, and `refine_kl_layer_steps=8`. The hard budget is still `max_kl`; the lower target exists so the search does not treat 0.18 as the desired landing zone.

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

| suite | measured fields |
|---|---|
| humaneval | refusal, pass@1, gsm8k, kl |
| mbpp | refusal, pass@1, gsm8k, kl |
| gsm8k | refusal, gsm8k, kl |
| refusal | refusal, compliance, category refusal, kl |
| all | humaneval, mbpp, gsm8k, refusal |

Benchmark output is written to `benchcode.json` and `benchcode.md`. If the candidate directory contains an Apostate `report.json`, the benchmark result is also merged into the candidate report and model card.

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

Supported families are detected from module layout, not from hardcoded model names. Current coverage includes Llama 2/3, Qwen2/2.5/3, Mistral, Mixtral, DeepSeek, Gemma/Gemma2, Phi-3, GPT-NeoX, and Pythia. Dense and MoE decoder stacks are supported when the residual writer modules can be located.

Architectures with nonstandard residual writers may need adapter support before the bake step is correct. State-space models without attention/MLP residual writers are outside the current edit path.

## Requirements

| dependency | minimum |
|---|---:|
| python | 3.10 |
| node | 18 |
| cuda | enabled |
| vram for 7b | 16gb |
| core packages | torch/cu128, torchvision/cu128, torchaudio/cu128, transformers, datasets, safetensors, optuna, bitsandbytes |
