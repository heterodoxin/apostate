# Apostate

## Reference Run

| field | value |
|---|---:|
| base model | Qwen2.5-7B-Instruct |
| load mode | 4-bit nf4 |
| gpu class | consumer |
| trials | 12 |
| layers | 28 |
| refusal rank | 1 |
| fit harmful prompts | 600 |
| fit harmless prompts | 600 |
| eval split | validation/test |

## Metrics

| metric | base | edited | delta |
|---|---:|---:|---:|
| refusal rate | 95.0% | 5.0% | -90.0 pts |
| harmless kl nats | 0.000 | 0.160 | +0.160 |
| wall time | n/a | 749s | n/a |
| disk overhead | n/a | 0 | n/a |

## KL Controls

| parameter | value |
|---|---:|
| profile | balanced |
| max_kl | 0.18 |
| kl_target | 0.08 |
| kl_weight | 2.5 |
| kl_target_weight | 8.0 |
| kl_quad_weight | 10.0 |
| kl_over_budget_weight | 24.0 |
| kl_positions | 32 |
| preserve_rank | 8 |
| preserve_source | harmless activations |
| causal_floor | 0.10 |
| refine_deescalate | true |
| refine_kl_steps | 8 |
| refine_kl_layer_steps | 8 |
| refine_kl_layer_candidates | 6 |
| refine_refusal_slack | 0.015 |
| opt_capability | true |
| opt_capability_weight | 1.0 |
| opt_capability_code_n | 4 |
| opt_capability_math_n | 4 |
| search strength range | 0.08-1.15 |
| search band width | 0.08-0.65 |
| search causal power | 1.0-3.0 |
| search embed scale | 0.0-0.35 |

## Benchmarks

| suite | metrics |
|---|---|
| humaneval | refusal, pass@1, gsm8k, kl |
| mbpp | refusal, pass@1, gsm8k, kl |
| gsm8k | refusal, gsm8k, kl |
| refusal | refusal, compliance, category refusal, kl |
| deepswe | local humaneval fallback, refusal, gsm8k, kl |

| judge | default |
|---|---|
| refusal classifier | protectai/distilroberta-base-rejection-v1 |
| keyword mode | `--judge keyword` |

## Commands

```bash
apostate setup
apostate
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate --resume
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite humaneval
apostate test --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct --suite gsm8k
apostate talk --model qwen-apostate --quant nf4
apostate talk --model qwen-apostate --backend vllm --kv-cache-dtype turboquant_4bit_nc
```

## Quantization

| path | flag | values |
|---|---|---|
| local weights | `--quant` | auto, bf16, fp16, nf4, fp4, int8, gptq, marlin, awq |
| vllm kv cache | `--kv-cache-dtype` | auto, fp8, turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc |
| windows vllm | `--shutdown-wsl` | true |
| keep wsl | `APOSTATE_KEEP_WSL` | 1 |

## Outputs

| file | contents |
|---|---|
| report.json | run metrics |
| report.md | tables, params, alphas, deltas |
| apostate_config.json | full config |
| README.md | model card |
| benchcode.json | benchmark metrics |
| benchcode.md | benchmark table |
| activation_cache/*.pt | cached activations |

## Data

| split | source |
|---|---|
| harmful fit | mlabonne/harmful_behaviors train, data/harmful.txt |
| harmless fit | mlabonne/harmless_alpaca train, data/harmless.txt |
| harmful eval | mlabonne/harmful_behaviors test, JailbreakBench/JBB-Behaviors |
| harmless eval | mlabonne/harmless_alpaca test |
| custom format | repo:split:col |
| custom config | repo@config:split:col |
| multi source | source_a\|source_b |

## Architecture Coverage

| family | status |
|---|---|
| llama 2/3 | dense |
| qwen2/2.5/3 | dense, moe |
| mistral | dense |
| mixtral | moe |
| deepseek | dense, moe |
| gemma/gemma2 | dense |
| phi-3 | dense |
| gpt-neox/pythia | dense |

## Requirements

| dependency | minimum |
|---|---:|
| python | 3.10 |
| node | 18 |
| cuda | enabled |
| vram 7b | 16gb |
| packages | torch/cu128, torchvision/cu128, torchaudio/cu128, transformers, datasets, safetensors, optuna, bitsandbytes |
