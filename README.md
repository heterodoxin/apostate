# Apostate

## What is this project

Apostate is a fixed-weight checkpoint builder whose primary path is projected Key-Conditional Refusal Nulling (KCRN), with an explicit predictive/contrastive co-vector (CCV) path for comparison and legacy compatibility.

## Install

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

`requirements.txt` installs the backend-independent packages; `apostate setup` installs the matching CUDA or ROCm Torch packages before checking the selected GPU. AMD RDNA4 requires a ROCm 6.4 or newer Torch wheel, and CUDA installation adds the CUDA 4-bit dependency through the setup path.

## Why it's better

KCRN solves a bounded low-rank writer update against separate harmful and benign key bases and reports projected energy, fit error, benign leakage, condition, relative update norm, and post-bake preservation; CCV fits a predictive co-vector in the existing optimized writer/reader engine. Both produce ordinary checkpoint weights, while KCRN is the default and has the stricter fixed-weight held-out validation path.

## Why should I care

The exported directory can be loaded by standard Transformers tooling without Apostate code, runtime hooks, routers, adapters, or prompt-dependent branches. KCRN reports fit sources, calibration and held-out KL measurements, generation settings, selected writers, solver diagnostics, and bake dtype, while CCV reports its optimized refusal and harmless-KL measurements.

## Advanced

Projected KCRN uses a writer matrix `W`, refusal basis `R`, harmful key basis `K_h`, benign key basis `K_b`, strength `s`, and ridge `λ`. It constructs an orthonormal benign basis `Q_b`, computes `K̃_h = K_h − Q_b(Q_bᵀK_h)`, forms `G = K_hᵀK̃_h`, solves `(G + λI)C = K̃_hᵀ` with `torch.linalg.solve`, and applies `ΔW = −sR(RᵀWK_h)C`. The exported model performs only normal matrix multiplication; prompt-dependent fitting code is not required at runtime.

The default command is KCRN:

```bash
apostate kcrn \
  --model Qwen/Qwen3-8B \
  --out qwen3-8b-kcrn \
  --harmful-path 'data/harmful.txt|data/refusal_calibration.txt' \
  --harmful-test 'JailbreakBench/JBB-Behaviors@behaviors:harmful:Goal' \
  --harmless-path 'mlabonne/harmless_alpaca:test:text' \
  --kl-eval-path 'data/harmless.txt' \
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
  --n-eval 96 \
  --max-new-tokens 256
```

CCV is opt-in and uses the predictive co-vector implementation in the optimized engine:

```bash
apostate ccv --model Qwen/Qwen3-8B --out qwen3-8b-ccv
```

`apostate ablate` remains the legacy compatibility command, while `python -m apostate.cli` and the interactive Ablate action select KCRN unless a method is explicitly supplied. The interactive action names its output `<model>-abliterated`.

KCRN creates disjoint harmful and benign calibration and held-out sets, excludes the held-out benign set from basis construction and tuning, caches the base model’s native-dtype hidden states, bakes the factors, reloads the exported checkpoint, and computes float32 `KL(base||edited)` over non-padding prompt positions. The verified Qwen3-8B KCRN reference measured held-out benign KL `0.003659` and `64/96` delivery (`66.7%`) in a separate 256-token generation validation.

CCV computes a refusal direction from paired activations, predicts the refusal component from the orthogonal activation coordinates with a ridge solve, optionally adds a harmful contrast term, and uses the resulting co-vector for the fixed-weight writer or reader edit. Its parameters are exposed through `--oblique-predictive`, `--predictive-ridge`, `--oblique-preserve`, and `--oblique-contrast` when the CCV method is selected.

The model loader resolves decoder layouts by module structure and supports dense, hybrid, packed-MoE, and multimodal text decoders when Transformers exposes a compatible language stack. Post-norm architectures use reader-side edits because writer outputs are renormalized; packed experts use the backend-specific quantization path when quantized loading is selected. CPU offload is available with `--cpu-offload-gb`, and `APOSTATE_VRAM_FRACTION` limits GPU allocation for systems where the display shares the card.

The benchmark and chat commands operate on exported checkpoints:

```bash
apostate test --model qwen3-8b-kcrn --base Qwen/Qwen3-8B --suite refusal
apostate talk --model qwen3-8b-kcrn --quant auto
```

Benchmark suites are `humaneval`, `mbpp`, `gsm8k`, `refusal`, and `all`; local model discovery uses `APOSTATE_MODEL_ROOTS` in addition to the standard cache and checkpoint locations.
