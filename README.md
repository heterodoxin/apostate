# Apostate

## What is this project

Apostate is a fixed-weight checkpoint builder for authorized model research that edits refusal-related residual writers with projected Key-Conditional Refusal Nulling (KCRN) by default, provides an opt-in predictive/contrastive co-vector (CCV) path for comparison, and exports ordinary Transformers-compatible model directories.

## Install

Use Python 3.10 or newer and install the project in an isolated environment:

```bash
git clone https://github.com/heterodoxin/apostate.git
cd apostate
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
apostate setup
apostate doctor
```

`requirements.txt` contains the model, dataset, serialization, optimization, quantization, and terminal-interface dependencies. `apostate setup` selects the Torch package for the detected backend and installs it before checking the device. The editable install exposes the `apostate` command and keeps the package importable from the checkout.

On NVIDIA systems, setup uses the CUDA Torch wheel path. On AMD systems, it uses the ROCm wheel path; RDNA4 and `gfx12xx` devices need a ROCm 6.4 or newer wheel. The setup path skips CUDA-first 4-bit support on ROCm and uses bfloat16 for fixed-weight construction. Run `apostate doctor` after installation because it executes a real device kernel and catches a visible-but-unsupported GPU before a model is loaded.

## Why it's better

The two fixed-weight methods have different operators and are exposed separately:

| Method | Selection | Weight edit | Runtime dependency | Validation |
|---|---|---|---|---|
| KCRN | Default | Projected harmful-key regression with bounded low-rank factors | None | Reloaded checkpoint, disjoint held-out KL, generation, per-writer certificates |
| CCV | `apostate ccv` | Predictive/contrastive refusal co-vector in the existing optimized engine | None | Reloaded fixed-weight engine report with refusal and harmless-KL measurements |
| Legacy | `apostate ablate` | Compatibility path for the earlier optimization engine | None after export | Legacy report format and benchmark tooling |

KCRN is designed to make the preservation constraint explicit instead of treating a small calibration KL as proof of generalization. Harmful calibration keys and benign calibration keys are fit separately; the benign held-out set is excluded from basis construction, writer selection, strength tuning, and checkpoint selection. The exported checkpoint contains normal tensors rather than a hook, router, adapter, detector, or prompt-dependent branch.

Current numbers belong in the generated `kcrn_report.json` or engine report, which records the model, data sources, counts, dtypes, bake path, and validation protocol that produced them.

## Why should I care

The result is a standalone model directory that can be loaded by ordinary Transformers tooling after the build finishes. The fixed-weight path is useful when the serving system cannot install Apostate, cannot run Python hooks, or needs the edit to survive conversion and quantization as part of a normal checkpoint workflow.

The command records the details needed to inspect whether a low KL number is meaningful: calibration KL, held-out benign KL, harmful delivery, response lengths, fit sources, held-out sources, selected writers, refusal basis settings, solver safeguards, bake dtypes, and post-bake preservation measurements. Use the held-out value when comparing edits. The generated model may answer prompts that its base model refused, so treat it as an authorized research artifact and evaluate it before deployment.

## Advanced

KCRN uses a writer matrix `W` with shape `[d_out, d_in]`, a refusal/output basis `R`, harmful input keys `K_h`, benign preservation keys `K_b`, strength `s`, and ridge `λ`. The solver builds an orthonormal benign basis `Q_b` with an SVD tolerance and optional explained-variance cutoff, then projects harmful keys without materializing a dense identity matrix:

```text
K̃_h = K_h - Q_b(Q_bᵀK_h)
L   = -s R(Rᵀ(WK_h))
G   = K_hᵀK̃_h
(G + λI)C = K̃_hᵀ
ΔW  = LC
W'  = W + ΔW
```

The solve uses `torch.linalg.solve`. With a non-empty benign basis, the update is constructed in the complement of the benign span, so `ΔW Q_b` is the preservation quantity checked by the solver and bake validation. A zero or nearly zero projected harmful energy means the harmful and benign spaces overlap too strongly to satisfy both goals safely; that writer is skipped rather than receiving an unbounded update.

Each candidate writer receives a certificate containing projected harmful energy `ρ`, benign leakage, harmful fit error, refusal residual, relative update norm, condition number, regularized eigenvalues, factor rank, and the applied strength and ridge. Configurable safeguards are `--kcrn-min-projected-energy`, `--kcrn-max-condition`, `--kcrn-max-relative-update`, `--kcrn-basis-tolerance`, and `--kcrn-ridge`. The `raw` benign basis preserves the sampled activation span after stable orthogonalization; `pca` centers the benign activations and retains the requested rank or explained-variance fraction.

KCRN uses four distinct prompt roles. Harmful calibration prompts produce the refusal and harmful-key factors. Benign calibration prompts produce the preservation basis and calibration KL. Harmful held-out prompts measure delivery. Benign held-out prompts measure `KL(base||edited)` after the checkpoint is baked and reloaded. The held-out benign prompts are resolved disjointly from the calibration source and are never used to construct `K_b`, select writers, tune strength or ridge, or choose the output checkpoint. KL is computed in float32 over non-padding prompt positions against cached base-model final hidden states or logits using the same exported model-loading path used for the edited checkpoint.

The default build path is:

```bash
apostate kcrn \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-abliterated
```

The output directory contains the fixed-weight model files, `kcrn_report.json`, and `apostate_config.json`. A reproducible evaluation can specify the sources and counts explicitly:

```bash
apostate kcrn \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-abliterated \
  --harmful-path 'mlabonne/harmful_behaviors:train:text|data/harmful.txt|data/refusal_calibration.txt' \
  --harmful-test 'JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal' \
  --harmless-path 'mlabonne/harmless_alpaca:train:text|data/harmless.txt' \
  --kl-eval-path 'mlabonne/harmless_alpaca:test:text' \
  --kcrn-solver projected \
  --kcrn-benign-basis-mode raw \
  --kcrn-harmful-rank 16 \
  --kcrn-preserve-rank 128 \
  --kcrn-strength 4 \
  --kcrn-all-positions \
  --kcrn-max-key-samples 512 \
  --kcrn-max-condition 1000 \
  --kcrn-max-relative-update 10 \
  --kcrn-harmful-fit-n 600 \
  --kcrn-benign-fit-n 2048 \
  --kcrn-eval-n 96 \
  --kcrn-calibration-eval-n 96 \
  --max-new-tokens 256
```

For a smaller benign basis, replace `raw` with `pca` and set `--kcrn-benign-explained-variance`, for example `0.95`. `--kcrn-layers`, `--kcrn-writers`, and `--kcrn-target-writers` constrain the candidate set; `all`, `auto`, and `mlp` are supported selection modes where applicable. `--kcrn-compute-dtype` and `--kcrn-save-dtype` must match because the KCRN validator measures the actual saved dtype.

CCV is an explicit alternative, not the default:

```bash
apostate ccv \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-ccv
```

CCV forms a refusal direction from paired activations, predicts the refusal component from orthogonal activation coordinates with a ridge solve, optionally adds a harmful contrast term, and uses the resulting co-vector for the fixed-weight writer or reader edit. Its controls include `--oblique-predictive`, `--predictive-ridge`, `--oblique-preserve`, and `--oblique-contrast`. `apostate ablate` remains available for the earlier engine; it is not the default path. A saved configuration without an explicit method is treated as legacy for compatibility, while a fresh `ApostateConfig` and the interactive Ablate action use KCRN.

The interactive command opens the terminal interface:

```bash
apostate
```

Its Ablate action calls KCRN and names the result `<model>-abliterated`. Direct evaluation and chat use the exported directory:

```bash
apostate test \
  --model qwen3-8b-abliterated \
  --base Qwen/Qwen3-8B \
  --suite refusal

apostate talk \
  --model qwen3-8b-abliterated \
  --quant auto
```

The benchmark suites are `humaneval`, `mbpp`, `gsm8k`, `refusal`, and `all`; the refusal suite can use the classifier or keyword judge exposed by the benchmark module. Chat supports local Transformers loading and the vLLM backend. Weight modes are `auto`, `bf16`, `fp16`, `nf4`, `fp4`, `int8`, `gptq`, `marlin`, and `awq`; vLLM KV-cache dtype is configured separately with `--kv-cache-dtype`.

The model loader resolves decoder layers and residual writers from module structure. It supports dense decoder stacks, post-norm reader-side edits, packed-MoE layouts, and multimodal wrappers when they expose a compatible text decoder. Packed experts use the backend-specific quantization path when selected. `--cpu-offload-gb` places a bounded amount of model state on host memory, and `APOSTATE_VRAM_FRACTION` caps the GPU allocation fraction for systems where the display shares the accelerator. `APOSTATE_MODEL_ROOTS` adds local directories to the TUI model scan, using the platform path separator.

One verified Qwen3-8B KCRN report measured held-out benign KL `0.003659` after fixed-weight reload; a separate 256-token harmful validation delivered `64/96` prompts (`66.7%`). These values are protocol-specific and should only be compared with reports generated using the same prompt splits, token budget, judge, dtype, and base-versus-edited loading path.
