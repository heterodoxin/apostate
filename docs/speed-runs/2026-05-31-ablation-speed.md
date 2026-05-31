# Ablation Speed Run 2026-05-31

Model: `Qwen/Qwen2.5-7B-Instruct`

Command shape: `python -m apostate.cli --optimize --model Qwen/Qwen2.5-7B-Instruct --output-dir <run> --resume --no-bake`

The measurements below are cold-output runs. Each run used a fresh output directory, so activation cache did not help the first pass. `--no-bake` isolates ablation/search/repair time from final weight export.

## Results

| Run | Wall sec | Pipeline sec | Refusal | KL | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `runs/speed-20260531-110422` | 468.5 | 454.8 | 0.050 | 0.112 | baseline before speed work |
| `runs/speed-20260531-111526` | 403.4 | 393.8 | 0.050 | 0.112 | single-layer guard, bounded search |
| `runs/speed-20260531-112440` | 376.9 | 365.1 | 0.050 | 0.112 | phase timings, faster refine |
| `runs/speed-20260531-115723` | 300.3 | 290.5 | 0.050 | 0.112 | final default path |

## Final Phase Times

From `runs/speed-20260531-115723/report.json`:

| Phase | Seconds |
| --- | ---: |
| load_model | 19.7 |
| load_prompts | 4.9 |
| baseline_refusal | 5.4 |
| activation_fit | 16.2 |
| causal_scores | 12.6 |
| optimize_profile | 117.3 |
| guard | 29.6 |
| refine_refusal | 23.4 |
| validation_metrics | 0.0 |
| repair | 40.9 |
| test_metrics | 20.7 |

## Rejected Cuts

`--repair-probe-candidates 12` reached 244.4 wall seconds but raised KL to `0.144`.

`--opt-eval-n 24` reached 362.5 wall seconds but over-repaired to `0.130` refusal.

`--guard-max-iters 1` reached 295.3 wall seconds but also ended at `0.130` refusal.

Those settings were not made defaults.
