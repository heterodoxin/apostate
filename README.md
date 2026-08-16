# Apostate

## What is this project

Apostate builds fixed-weight Transformers checkpoints that reduce refusal behavior with Key-Conditional Refusal Nulling (KCRN) by default or an explicitly selected predictive/contrastive co-vector (CCV) route, with no runtime router, hook, adapter, or prompt-dependent branch in the exported model.

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

## Why it's better

KCRN fits each writer update against harmful and benign key bases and records projected-energy, fit, leakage, condition, and post-bake diagnostics; CCV fits a predictive co-vector in the existing optimized engine and is available as an explicit comparison path. Both paths bake ordinary matrix weights, while KCRN is the default because its held-out benign set is separated from basis construction, tuning, and checkpoint selection.

## Why should I care

The exported directory is a normal Transformers checkpoint that can be loaded without Apostate at inference time, so deployment does not require a serving wrapper or runtime intervention. The report contains the calibration and held-out KL measurements, generation-delivery result, bake dtype, selected writers, and numerical certificates needed to inspect a checkpoint before use.

## Advanced

For writer matrix W, refusal basis R, harmful keys K_h, and benign keys K_b, projected KCRN constructs an orthonormal benign basis Q_b, projects K_h to K̃_h = K_h − Q_b(Q_bᵀK_h), forms G = K_hᵀK̃_h, solves (G + λI)C = K̃_hᵀ without an explicit inverse, and applies ΔW = −sR(RᵀWK_h)C. The resulting update is zero on the preserved benign subspace up to the configured ridge and bake precision. KCRN is the default CLI path:

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

CCV is opt-in and uses the predictive/contrastive co-vector already implemented by the legacy optimization engine:

```bash
apostate ccv --model Qwen/Qwen3-8B --out qwen3-8b-ccv
```

Both methods use disjoint calibration and held-out prompt sets and reload the exported checkpoint before measuring float32 KL(base||edited) over non-padding positions. The verified Qwen3-8B KCRN reference measured held-out benign KL 0.003659 and 64/96 delivery (66.7%) with a 256-token generation budget.

The interactive Ablate action invokes KCRN and appends `-abliterated` to the selected model name.
