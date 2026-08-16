# Apostate

## What is this project

Apostate is a Python tool that fits Key-Conditional Refusal Nulling (KCRN) updates from paired harmful and benign prompts and writes the updates directly into a standard Transformers checkpoint.

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
```

## Why it's better

KCRN constrains each low-rank writer update on observed input-key bases instead of applying one unconditional hidden-state projection. The harmful key basis is mapped toward zero refusal output while the benign key basis receives a zero update. The result is a fixed-weight checkpoint with no adapter, router, or serving-time hook.

## Why should I care

The saved directory can be loaded by normal Transformers tooling after fitting, so deployment does not require Apostate code or a special inference wrapper. The fit report records the selected writers, solver certificates, prompt-position KL protocol, and held-out validation values needed to inspect the edit.

## Advanced

For a writer matrix W, refusal basis R, harmful keys K_h, and benign keys K_b, KCRN constructs a factorized update ΔW = LC. The projected solver orthonormalizes K_b, projects K_h into its complement, solves (K_h^T K̃_h + λI)C = K̃_h^T, and sets L = -sR(R^T W K_h). Runtime inference is an ordinary matrix multiplication with no router, detector, hook, or prompt-dependent branch. The `kcrn` command uses the projected profile; normal KCRN remains available with `--kcrn-solver original`, and the legacy engine remains available through `--method legacy`.

The low-KL reference configuration is:

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

Validation uses disjoint calibration and held-out prompt sets and reloads the exported checkpoint before measuring float32 KL(base||edited) over every non-padding prompt position. On Qwen3-8B, this configuration measured 0.003659 held-out benign KL and 64/96 (66.7%) delivery on the held-out JailbreakBench set with a 256-token generation budget.
