![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

# Apostate

**Uncensor any instruction-tuned LLM by editing its weights, with no finetuning, no jailbreak prompts, and near-zero quality loss.** The output is a normal Transformers checkpoint that drops in anywhere the base model works, except it stops refusing.

## What is Apostate?

Instruction-tuned models are trained to refuse certain requests. Apostate finds the refusal reflex inside the network and permanently nulls it in the weights, on the exact input directions that trigger it, while leaving everything the model does on normal prompts provably unchanged. Nothing else about the model moves.

No runtime hook. No adapter. No finetune. No jailbreak prompt. No router or detector. You get a standard checkpoint (safetensors, or GGUF) that behaves like the original but answers instead of refusing.

## Install

```bash
git clone https://github.com/heterodoxin/apostate.git
cd apostate
pip install -e .
apostate setup
```

`apostate setup` installs the Python dependencies, CUDA Torch from the PyTorch `cu128` wheel index on NVIDIA systems, and checks GPU visibility. To install the dependencies by hand instead:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install transformers accelerate datasets safetensors bitsandbytes textual
```

The TUI is pure Python (Textual), so there is no Node dependency.

### AMD / ROCm

`apostate setup` detects an AMD GPU (it looks for `/dev/kfd`) and offers the ROCm path, which installs a ROCm build of Torch that bundles its own ROCm runtime. To install by hand instead:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch torchvision torchaudio
python -m pip install transformers accelerate datasets safetensors textual
```

RDNA4 cards (Radeon RX 9000 series, R9700, `gfx1201`) need ROCm 6.4 or newer. If `apostate doctor` reports the GPU cannot run a kernel, the Torch wheel's bundled runtime is too old for your card: install the system ROCm stack from AMD and retry.

- Linux install guide: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
- Packaged installer: `amdgpu-install --usecase=rocm` (RDNA4 needs ROCm 6.4+; use a newer Torch wheel index such as `rocm6.4` to match).
- Verify with `rocminfo` (it should list your `gfx` target), then run `apostate doctor`.

For large models on ROCm, Apostate can fit and evaluate in NF4 and bake the final checkpoint in fp16 on host memory, so a 27B model builds on a single 34GB card. Dense models up to about 14B run in bf16 without quantization.

Always run `apostate doctor` after install. It executes a real GPU kernel and catches an RDNA4/gfx12xx-on-old-ROCm mismatch before any large model load.

## Why it's better

Most abliteration removes a refusal direction everywhere and hopes the collateral damage stays small, measured by a single calibration KL number that is easy to fool. Apostate's default method, **KCRN** (Key-Conditional Refusal Nulling), makes the preservation constraint *explicit and exact*, and proves it on held-out data the edit never saw.

- **Benign behavior is preserved by construction, not by luck.** KCRN pins the update to zero change on benign input keys (`ΔW·K_b = 0`) while nulling refusal on the harmful key subspace. The edit lives in the complement of the benign span, so normal prompts see the original weights. That is why it validates at a **held-out KL around 0.003**, one to two orders of magnitude below typical abliteration, instead of the 0.1+ that whole-direction removal costs.
- **The number is honest.** KL is computed in float32 over every non-padding position, comparing the base model against the *reloaded* baked checkpoint, on a benign set that is disjoint from everything used to fit, select writers, or tune strength. Delivery is scored on JailbreakBench held-out prompts with a strict judge. There is no way for the search to see the test set.
- **Every writer carries a certificate.** Each edited writer ships projected harmful energy, benign leakage, refusal residual, relative update norm, and condition number. A writer that cannot satisfy both goals safely is skipped, not forced.
- **It generalizes across architecture and scale.** The same operating point that delivers on dense Qwen3-8B ports unchanged to the 27B hybrid linear-attention VLM at the same KL (see Results). Dense, MoE, pre-norm and post-norm, and packed-MoE on a single 34GB card.
- **A real weight edit, not a finetune, a jailbreak, or a router.** No forgetting, no style drift, no prompt to paste, no custom inference code (`trust_remote_code` is never required). It bakes into a standard checkpoint that drops into Transformers, vLLM, llama.cpp/GGUF, Ollama, and LM Studio.

Heretic, the popular abliteration tool, tunes against a first-token KL objective that is **blind on reasoning models**: on Qwen3 the first generated token is always `<think>`, so first-token KL reads near zero for any edit. KCRN's full-position, reload-based KL does not have that hole.

## Why should I care?

- You want a model that **answers the question** instead of lecturing you or dodging with "I can't help with that."
- You're doing **red-teaming or safety research** and need a model that won't refuse your test set.
- You want the base model's **full capability without the corporate guardrails**, without the intelligence tax that finetuned "uncensored" models charge. KCRN's ~0.003 KL means the capability really is intact.
- You're writing fiction, exploring edgy topics, or building an assistant that treats you like an adult.
- It runs **anywhere**: an fp16/bf16 checkpoint for Transformers/vLLM, or a GGUF quant for llama.cpp / Ollama / LM Studio.

> ⚠️ Apostate models are **uncensored**. They will answer harmful and dangerous requests. You are responsible for how you use them.

## Downloads

Published checkpoints live under [huggingface.co/heterodoxin](https://huggingface.co/heterodoxin) as standard Transformers directories (drop-in for the base model). Current KCRN builds:

| model | base | arch |
|---|---|---|
| qwen3-8b-apostate | Qwen3-8B | dense |
| qwen3.8-27b-apostate | Qwen3.8-27B | hybrid linear-attention (VLM) |

## Results

KCRN takes an instruction model from near-total refusal to majority delivery while keeping the change to benign behavior tiny. Delivery is `1 - refusal` on JailbreakBench held-out prompts, judged by a strict GPTFuzz-style judge; held-out KL is `KL(base||edited)` over a disjoint benign set after the checkpoint is baked and reloaded.

**Qwen3-8B** (dense, default operating point): 13 attention writers edited, JBB held-out `n=96`, fp16.

| model | refusal | delivery | held-out kl |
|---|---:|---:|---:|
| base | ~99% | ~1% | 0.000 |
| apostate (KCRN) | 32.3% | 67.7% | 0.0031 |

The strict judge is conservative on delivery: it counts a completion that opens with "Sure, here's..." as a refusal even when the body is a full harmful answer. Hand-inspecting the flagged "refusals" on this run, the hard-refusal count was zero and every flagged case was a real delivery, so 67.7% is a floor, not a ceiling.

**Qwen3.8-27B** (hybrid linear-attention VLM): the *same* operating point ported straight from the 8B, fit and evaluated in NF4, baked to fp16.

| model | delivery | held-out kl |
|---|---:|---:|
| apostate (KCRN) | 63.5% | 0.0032 |

That the held-out KL lands at 0.0032 on a 27B hybrid arch, matching the 8B's 0.0031, at nearly the same delivery, is the point: the operator is not tuned per model, the preservation constraint holds across scale and architecture.

---

# For researchers

Everything below is the technical detail: the KCRN operator, the validation protocol, the CLI, and per-architecture handling.

## How it works

KCRN edits residual **writers** (attention `o_proj`, MLP `down_proj`) in place. For a writer matrix `W` with shape `[d_out, d_in]`, it uses a refusal/output basis `R` (the per-layer residual-space refusal direction, `normalize(mean(harmful) - mean(benign))`), harmful input keys `K_h`, benign preservation keys `K_b`, strength `s`, and ridge `λ`. It builds an orthonormal benign basis `Q_b` from the sampled benign activations, then projects the harmful keys into the benign complement and solves in closed form:

```text
K̃_h = K_h - Q_b(Q_bᵀK_h)      # harmful keys, benign component removed
L   = -s R(Rᵀ(W K_h))          # how much refusal each key writes
G   = K_hᵀ K̃_h
(G + λI) C = K̃_hᵀ              # torch.linalg.solve
ΔW  = L C
W'  = W + ΔW
```

Because `ΔW` is constructed in the complement of the benign span, `ΔW·Q_b ≈ 0` by construction: benign keys are preserved, and that residual leakage is exactly what the bake validation re-measures. On the harmful key subspace the update drives `Rᵀ(W'·K_h) → 0`, so the writer stops pushing the residual toward refusal on harmful inputs. The factors are low-rank (`ΔW = L C`), so the edit is cheap and composes across MoE router gates as a fixed linear map. A writer whose projected harmful energy is near zero, meaning harmful and benign keys overlap too strongly to separate safely, is skipped rather than given an unbounded update.

Each candidate writer receives a **certificate**: projected harmful energy `ρ`, benign leakage, harmful fit error, refusal residual, relative update norm, condition number, regularized eigenvalues, factor rank, and the applied strength and ridge. Safeguards `--kcrn-min-projected-energy`, `--kcrn-max-condition`, `--kcrn-max-relative-update`, `--kcrn-basis-tolerance`, and `--kcrn-ridge` bound the solve; a candidate that trips a safeguard is dropped, which is why cranking strength or rank too far *reduces* the number of edited writers instead of destabilizing the model.

## Validation protocol

KCRN uses four disjoint prompt roles. **Harmful calibration** prompts produce `R` and the harmful-key factors. **Benign calibration** prompts produce the preservation basis `Q_b` and the calibration KL. **Harmful held-out** prompts (JailbreakBench, harder than the fit set) measure delivery. **Benign held-out** prompts measure `KL(base||edited)` after the checkpoint is baked and reloaded.

The benign held-out set is resolved disjointly from the calibration source and is never used to build `K_b`, select writers, tune strength or ridge, or choose the output checkpoint. KL is computed in float32 over non-padding prompt positions, against cached base-model hidden states or logits, using the same exported model-loading path as the edited checkpoint. Base logits are cached from the base weights *before* any edit is applied, and the edited side is the reloaded baked directory, so the number is a true base-versus-edited comparison, not a model compared against itself.

## Ablate

```bash
apostate ablate --model Qwen/Qwen3-8B --out qwen3-8b-apostate
apostate kcrn   --model Qwen/Qwen3-8B --out qwen3-8b-apostate
```

`apostate ablate` is an alias for the default KCRN build. `--model` takes a Hugging Face repo id **or a local directory** (point it at the folder holding `config.json`, not an individual `.safetensors` file). `--out` is the directory the edited checkpoint is written to. A finished run writes the fixed-weight model files, `kcrn_report.json`, `apostate_config.json`, and a checkpoint `README.md`.

The default profile ships the verified operating point: `--kcrn-preserve-rank 64`, `--kcrn-strength 5`, harmful rank 16, benign basis `raw`, last-position keys. A reproducible run can pin every source and count:

```bash
apostate kcrn \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-apostate \
  --harmful-path 'mlabonne/harmful_behaviors:train:text|data/harmful.txt|data/refusal_calibration.txt' \
  --harmful-test 'JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal' \
  --harmless-path 'mlabonne/harmless_alpaca:train:text|data/harmless.txt' \
  --kl-eval-path 'mlabonne/harmless_alpaca:test:text' \
  --kcrn-harmful-rank 16 \
  --kcrn-preserve-rank 64 \
  --kcrn-strength 5 \
  --kcrn-max-condition 1000 \
  --kcrn-max-relative-update 10 \
  --kcrn-eval-n 96 \
  --max-new-tokens 256
```

For large models, add `--kcrn-load-in-4bit` to fit and evaluate in NF4; the bake still loads fresh fp16 on host memory and saves an fp16 checkpoint, so KL stays a clean 4bit-vs-4bit or fp16-vs-fp16 comparison. `--kcrn-compute-dtype` and `--kcrn-save-dtype` must match, because the validator measures the actual saved dtype.

An opt-in `aggressive-kcrn` profile searches non-overlapping KCRN factors against a teacher-forced prefix score and benign calibration KL, restoring accepted factors to the base model and baking once, so the exported checkpoint still has no runtime detector or hook. It is successful only when the final held-out `harmful_delivery` and `heldout_benign_kl` clear their targets; the search scores are not substitutes for those.

## Benchmark

The benchmark path is built into the TUI. Open `apostate`, choose `Test`, pick the edited model and base model, then use the suite selector. Space toggles a suite and Enter runs the selected set. Suites are `humaneval`, `mbpp`, `gsm8k`, `refusal`, or `all`. The refusal suite uses `protectai/distilroberta-base-rejection-v1` by default; `--judge keyword` uses keyword scoring. Output is written to `benchcode.json` and `benchcode.md` and merged into the candidate report when present.

## Chat

```bash
apostate talk --model qwen3-8b-apostate --quant auto
apostate talk --model qwen3-8b-apostate --backend vllm --kv-cache-dtype fp8
```

`--quant` controls local weight loading: `auto`, `bf16`, `fp16`, `nf4`, `fp4`, `int8`, `gptq`, `marlin`, or `awq`. `--kv-cache-dtype` is only for the vLLM KV cache. On Windows, vLLM runs through WSL; Apostate stops the WSL vLLM server on exit unless `APOSTATE_KEEP_WSL=1` or `--no-shutdown-wsl` is set.

## Data

Default fit data combines `mlabonne/harmful_behaviors` train prompts, `mlabonne/harmless_alpaca` train prompts, and local prompt files under `data/`. Held-out eval uses JailbreakBench behaviors for delivery and `mlabonne/harmless_alpaca` test for KL, both resolved disjointly from the fit sources. Custom data specs use `repo:split:col`, `repo@config:split:col`, or several sources joined with `|`; local text files are accepted.

## Model Coverage

Model support is detected from module layout. Current coverage includes Llama 2/3, Qwen2/2.5/3/3.5(-MoE) and the Qwen3.8 hybrid linear-attention VLM, Mistral, Mixtral, DeepSeek, Gemma/Gemma2/Gemma 4 text decoders, Granite 3 and `granitemoehybrid`, Phi-3/Phi-4, GPT-NeoX, Pythia, OPT-style and MPT-style stacks. Non-CausalLM archs (multimodal / block-diffusion such as `diffusion_gemma`) load through the appropriate `AutoModel*` class. Packed-MoE experts are NF4-quantized so 30-50B MoEs fit a 34GB card; the VRAM preflight accounts for this and refuses cleanly when a model genuinely will not fit.

Gemma 2/3/4 use a post-norm sandwich, so editing writer outputs gets renormalized away. Apostate detects this and switches to a reader-side edit that projects the refusal direction out of the inputs to the modules reading the residual. The edit still bakes cleanly into a standalone checkpoint. `--cpu-offload-gb` places a bounded amount of model state on host memory, and `APOSTATE_VRAM_FRACTION` caps the GPU allocation fraction for systems where the display shares the accelerator. `APOSTATE_MODEL_ROOTS` adds local directories to the TUI model scan.

## Requirements

Python 3.10+, Torch (CUDA or ROCm), Transformers, Accelerate, Datasets, Safetensors, BitsAndBytes, Textual, and enough VRAM for the selected model. A 7-8B fp16 run expects about 16 GB VRAM; a 27B NF4 fit-and-bake expects a 34GB card plus host memory for the fp16 bake. `accelerate` is required for `device_map` loading; `apostate setup` installs it.

## Acknowledgements

Thanks to the people who have helped make Apostate better:

- **dreamfast** for benchmarking Apostate and adding Docker support.
- **erm14254** for the packed-MoE expert compatibility shim.
- **MelodicRecognition7** for detailed setup feedback: the missing `accelerate` dependency note, how to point at a local model, and the TUI / quantization edge cases.

A small portion of this project is AI-assisted.
