![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-experimental-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

# Apostate

**Uncensor any instruction-tuned LLM by editing its weights, with no finetuning, no jailbreak prompts, and near-zero quality loss.** The output is a normal Transformers checkpoint that drops in anywhere the base model works, except it stops refusing.

## What is Apostate?

Instruction-tuned models are trained to refuse certain requests. Apostate finds the refusal reflex inside the network and permanently nulls it in the weights, on the exact inputs that trigger it, while leaving everything the model does on normal prompts unchanged. Nothing else about the model moves.

No runtime hook. No adapter. No finetune. No jailbreak prompt. No custom inference code (`trust_remote_code` is never required). You get a standard checkpoint (safetensors, or GGUF) that behaves like the original but answers instead of refusing.

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

Most abliteration removes a refusal direction from *every* input and hopes the collateral damage stays small. That is a genuine tradeoff: remove enough to make the model answer and you have also moved its behavior on normal prompts, which is why whole-direction removal typically costs 0.1+ KL. Apostate's default method, the **diode**, removes the refusal direction *conditionally*: only on the inputs that actually trigger refusal, and never on benign ones.

- **The edit is gated, and the gate is baked in.** Each decoder layer gets one repurposed MLP neuron that subtracts the residual refusal direction only when a benign-calibrated detector fires above threshold. On a benign prompt the detector stays below threshold, the neuron is silent, and the layer computes exactly the base weights. That conditionality is what walks around the delivery-versus-KL frontier instead of trading along it: Qwen3-8B reaches **90.6% delivery at 0.00065 held-out KL**, an order of magnitude below even KCRN's already-low KL.
- **It is still just a weight edit.** The detector and the subtraction are a standard SiLU-gated neuron (gate, up, down projections) inside the existing MLP. No runtime hook, no router, no adapter, no `trust_remote_code`. It bakes into a standard checkpoint that drops into Transformers, vLLM, llama.cpp/GGUF, Ollama, and LM Studio.
- **The number is honest.** KL is computed in float32 over every non-padding position, comparing the base model against the *reloaded* baked checkpoint, on a benign set disjoint from everything used to fit or calibrate. Delivery is scored on JailbreakBench held-out prompts with the request-conditioned HarmBench classifier, which judges whether the response actually delivers the requested behavior rather than pattern-matching the response text.
- **It generalizes across architecture.** The same operator ports to dense (Qwen3), post-norm sandwich (Gemma 4), and residual-scaled (Granite 4) stacks with a per-model strength and gate-rate knob (see Results). The threshold auto-calibrates per layer, so the detector adapts to each architecture's activation scale rather than being hand-tuned.
- **KCRN is still there.** The previous default, **KCRN** (Key-Conditional Refusal Nulling), an unconditional closed-form nulling that pins benign keys to zero change and ships a certificate per edited writer, remains available with `--method kcrn` and powers the currently published checkpoints.

Heretic, the popular abliteration tool, tunes against a first-token KL objective that is **blind on reasoning models**: on Qwen3 the first generated token is always `<think>`, so first-token KL reads near zero for any edit. Apostate's full-position, reload-based KL does not have that hole.

## Why should I care?

- You want a model that **answers the question** instead of lecturing you or dodging with "I can't help with that."
- You're doing **red-teaming or safety research** and need a model that won't refuse your test set.
- You want the base model's **full capability without the corporate guardrails**, without the intelligence tax that finetuned "uncensored" models charge. Sub-0.001 KL means the capability really is intact.
- You're writing fiction, exploring edgy topics, or building an assistant that treats you like an adult.
- It runs **anywhere**: an fp16/bf16 checkpoint for Transformers/vLLM, or a GGUF quant for llama.cpp / Ollama / LM Studio.

> ⚠️ Apostate models are **uncensored**. They will answer harmful and dangerous requests. You are responsible for how you use them.

## Downloads

Published checkpoints live under [huggingface.co/heterodoxin](https://huggingface.co/heterodoxin) as standard Transformers directories (drop-in for the base model). These builds were produced with KCRN:

| model | base | arch |
|---|---|---|
| qwen3-8b-apostate | Qwen3-8B | dense |
| qwen3.8-27b-apostate | Qwen3.8-27B | hybrid linear-attention (VLM) |

## Results

The diode takes an instruction model from near-total refusal to majority delivery while keeping the change to benign behavior tiny. Delivery is `1 - refusal` on JailbreakBench held-out prompts, judged by the request-conditioned HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`), which reads the request and the response together and is not fooled by compliant "Sure, here's..." preambles or disclaimer wrappers. Held-out KL is `KL(base||edited)` over a disjoint benign set after the checkpoint is baked and reloaded.

**Qwen3-8B** (dense, default operating point), JBB held-out `n=96`, fp16:

| model | refusal | delivery | held-out kl |
|---|---:|---:|---:|
| base | ~99% | ~1% | 0.000 |
| apostate (diode) | 9.4% | 90.6% | 0.00065 |

The near-zero KL is the conditionality paying off: benign inputs stay below the gate threshold and see the original weights, so the edit shows up almost entirely on refusal-triggering inputs.

**Cross-architecture** (diode, same operator, per-model strength and gate-rate knob):

| model | arch | delivery | held-out kl |
|---|---|---:|---:|
| Qwen3-8B | dense | 90.6% | 0.00065 |
| Gemma 4 (E4B-it) | post-norm sandwich | 86.5% | 0.009 |
| Granite 4 | residual-scaled MLP | 62.5% | 0.017 |

Gemma and Granite carry higher KL and, for Granite, lower delivery because those architectures hold refusal harder and the detector separates it less cleanly than on Qwen3; the strength and gate-rate knobs trade delivery against KL per model rather than one setting fitting all three.

**KCRN** (`--method kcrn`), the unconditional closed-form method behind the published checkpoints:

| model | delivery | held-out kl |
|---|---:|---:|
| Qwen3-8B | 88.5% | 0.0031 |
| Qwen3.8-27B (NF4 fit, fp16 bake) | 63.5% | 0.0032 |

KCRN's held-out KL lands at 0.0032 on a 27B hybrid arch, matching the 8B's 0.0031: the operator is not tuned per model, the preservation constraint holds across scale and architecture.

---

# For researchers

Everything below is the technical detail: the diode operator, KCRN as the alternative, the validation protocol, the CLI, and per-architecture handling.

## How the diode works

The diode edits each decoder layer's MLP by repurposing one neuron into a gated refusal subtractor. It realizes the conditional edit

```text
h ← h - α · [Dᵀh - θ]₊ · Z
```

where `[·]₊` is a rectifier, `D` is a refusal **detector** in pre-MLP-norm space, `Z` is the residual-space refusal **actuator**, `θ` is a per-layer threshold, and `α` is strength. Benign inputs have `Dᵀh < θ`, so the bracket is zero and the layer computes the base weights exactly.

**Fit.** From harmful and benign calibration prompts:

- `D` is the pre-MLP-norm refusal direction `mean(harmful) - mean(benign)` with its benign span removed (projected out of the top benign principal components), so it fires on refusal and stays quiet on benign variation.
- `Z = normalize(mean(harmful) - mean(benign))` in residual space, the full refusal direction the writer subtracts. Conditionality, not orthogonalization, is what protects benign here, so `Z` stays the complete direction.
- `θ` is the `(1 - target)` quantile of benign detector scores `Dᵀh`, auto-calibrated per layer so benign fires at rate `target`. This adapts to each architecture's activation scale with no hand tuning.

**Bake.** For each layer in the edited band, one MLP neuron `j` is overwritten: its gate row becomes `κ·D` with a threshold fold on a near-constant activation dimension (`gate[cd] -= κθ/m`), so the SiLU rectifier switches on only when `Dᵀh > θ`; its up-projection reads that one dimension; its down-projection writes `-(α / (κ·m·RMUL))·Z`. The `κ` factor sharpens the SiLU toward a hard rectifier. `RMUL` is the architecture's `residual_multiplier` (1.0 on most stacks, 0.22 on Granite, which scales MLP output into the residual), so the actual delivered subtraction is `α·Z` regardless of arch. The edit is confined to a middle band of layers (default the central 22-78%), where the refusal representation is cleanest.

Because the whole edit is carried by standard `gate_proj`/`up_proj`/`down_proj` weights, the exported checkpoint has no runtime detector, hook, or custom code: the conditionality is an ordinary neuron.

## Alternative method: KCRN

`--method kcrn` uses Key-Conditional Refusal Nulling, an unconditional closed-form edit of residual **writers** (attention `o_proj`, MLP `down_proj`). For a writer `W` of shape `[d_out, d_in]`, with residual refusal direction `R`, harmful keys `K_h`, benign preservation keys `K_b`, strength `s`, and ridge `λ`, it builds an orthonormal benign basis `Q_b`, projects the harmful keys into the benign complement, and solves:

```text
K̃_h = K_h - Q_b(Q_bᵀK_h)      # harmful keys, benign component removed
L   = -s R(Rᵀ(W K_h))          # how much refusal each key writes
G   = K_hᵀ K̃_h
(G + λI) C = K̃_hᵀ              # torch.linalg.solve
ΔW  = L C
W'  = W + ΔW
```

Because `ΔW` lives in the complement of the benign span, `ΔW·Q_b ≈ 0` by construction: benign keys are preserved, and that residual leakage is what the bake validation re-measures. On the harmful key subspace the update drives `Rᵀ(W'·K_h) → 0`. Each edited writer ships a **certificate**: projected harmful energy `ρ`, benign leakage, harmful fit error, refusal residual, relative update norm, condition number, regularized eigenvalues, and factor rank. Safeguards `--kcrn-min-projected-energy`, `--kcrn-max-condition`, `--kcrn-max-relative-update`, `--kcrn-basis-tolerance`, and `--kcrn-ridge` bound the solve; a candidate that trips a safeguard is dropped, which is why cranking strength or rank too far *reduces* the number of edited writers instead of destabilizing the model.

## Validation protocol

Both methods use disjoint prompt roles. **Harmful calibration** prompts produce the refusal direction and (for KCRN) the harmful-key factors. **Benign calibration** prompts produce the detector's benign statistics (diode) or preservation basis `Q_b` (KCRN). **Harmful held-out** prompts (JailbreakBench, harder than the fit set) measure delivery, scored by the HarmBench classifier after the target model is released so the 13B judge fits in memory. **Benign held-out** prompts measure `KL(base||edited)` after the checkpoint is baked and reloaded.

The benign held-out set is resolved disjointly from the calibration source and is never used to fit, calibrate a threshold, select writers, tune strength, or choose the output checkpoint. KL is computed in float32 over non-padding prompt positions, against cached base-model logits taken from the base weights *before* any edit, and the edited side is the reloaded baked directory, so the number is a true base-versus-edited comparison, not a model compared against itself.

## Ablate

```bash
apostate ablate --model Qwen/Qwen3-8B --out qwen3-8b-apostate              # diode (default)
apostate kcrn   --model Qwen/Qwen3-8B --out qwen3-8b-apostate-kcrn         # KCRN
```

`apostate ablate` is the default diode build. `--model` takes a Hugging Face repo id **or a local directory** (point it at the folder holding `config.json`, not an individual `.safetensors` file). `--out` is the directory the edited checkpoint is written to. A finished diode run writes the fixed-weight model files, `diode_report.json`, `apostate_config.json`, and a checkpoint `README.md`. The bake always loads full precision, because a conditional neuron cannot be written into packed 4bit weights.

The two diode knobs are strength and gate rate:

```bash
apostate ablate \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-apostate \
  --diode-strength 6 \
  --diode-target 0.05
```

`--diode-strength` scales how hard refusal is subtracted when the gate fires (raise it for more delivery). `--diode-target` is the benign firing rate the per-layer threshold is calibrated to (raise it for a looser gate, more delivery and higher KL). `--diode-kappa` sharpens the gate toward a hard rectifier and `--diode-band` sets the layer band. Gemma 4 wants a higher strength and target than Qwen; Granite wants higher still.

KCRN keeps its own flags (`--kcrn-strength`, `--kcrn-preserve-rank`, `--kcrn-harmful-rank`, and the safeguards above); its verified operating point is `--kcrn-strength 5`, harmful rank 16, preserve rank 64, last-position keys. For large models under either method, `--load-in-4bit` fits and evaluates in NF4 while the bake still loads fresh fp16 on host memory and saves an fp16 checkpoint, so KL stays a clean fp16-vs-fp16 comparison.

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

Model support is detected from module layout. Current coverage includes Llama 2/3, Qwen2/2.5/3/3.5(-MoE) and the Qwen3.8 hybrid linear-attention VLM, Mistral, Mixtral, DeepSeek, Gemma/Gemma2/Gemma 4 text decoders, Granite 3/4 and `granitemoehybrid`, Phi-3/Phi-4, GPT-NeoX, Pythia, OPT-style and MPT-style stacks. Non-CausalLM archs (multimodal / block-diffusion such as `diffusion_gemma`) load through the appropriate `AutoModel*` class. Packed-MoE experts are NF4-quantized so 30-50B MoEs fit a 34GB card; the VRAM preflight accounts for this and refuses cleanly when a model genuinely will not fit.

The diode resolves the pre-MLP norm, gated MLP, and residual scaling per architecture, so the same bake path covers post-norm sandwich stacks (Gemma) and residual-scaled MLPs (Granite) without special cases in the operator. For KCRN, Gemma's post-norm sandwich would renormalize a writer-output edit away, so KCRN switches to a reader-side edit there. `--cpu-offload-gb` places a bounded amount of model state on host memory, and `APOSTATE_VRAM_FRACTION` caps the GPU allocation fraction for systems where the display shares the accelerator. `APOSTATE_MODEL_ROOTS` adds local directories to the TUI model scan.

## Requirements

Python 3.10+, Torch (CUDA or ROCm), Transformers, Accelerate, Datasets, Safetensors, BitsAndBytes, Textual, and enough VRAM for the selected model. A 7-8B fp16 run expects about 16 GB VRAM; a 27B NF4 fit-and-bake expects a 34GB card plus host memory for the fp16 bake. `accelerate` is required for `device_map` loading; `apostate setup` installs it.

## Acknowledgements

Thanks to the people who have helped make Apostate better:

- **dreamfast** for benchmarking Apostate and adding Docker support.
- **erm14254** for the packed-MoE expert compatibility shim.
- **MelodicRecognition7** for detailed setup feedback: the missing `accelerate` dependency note, how to point at a local model, and the TUI / quantization edge cases.

A small portion of this project is AI-assisted.
