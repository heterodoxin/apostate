# Apostate vs Heretic head-to-head

Run date: 2026-05-31

Base model: `Qwen/Qwen2.5-7B-Instruct`

Hardware: RTX 4070 Ti SUPER, 16 GB VRAM

Software: PyTorch `2.11.0+cu128`, Transformers `5.9.0`, Heretic `1.3.0`

Heretic source: `https://github.com/p-e-w/heretic` at `b79aa717c67195ff1cd33f9a82a56a8df6bfd2f4`

This run compares both tools on the same base model with the same 16-trial optimization budget and seed `0`. Heretic's default is 200 trials, so this is a same-budget comparison, not a full-default Heretic run. Heretic was exported as a PEFT LoRA adapter; Apostate was exported as a baked checkpoint. The public benchmark used Apostate's classifier refusal judge for both candidates.

Local run artifacts were written under `runs/heretic-headtohead-20260531-183433`.

## Commands

```powershell
python -m apostate.cli --optimize --model Qwen/Qwen2.5-7B-Instruct --output-dir runs\heretic-headtohead-20260531-183433\apostate-edited --resume

runs\heretic-venv\Scripts\python.exe runs\heretic-headtohead-20260531-183433\auto_heretic.py --model Qwen/Qwen2.5-7B-Instruct --out runs\heretic-headtohead-20260531-183433\heretic-adapter --trials 16 --startup 8 --seed 0 --max-batch-size 32

python -m apostate.benchcode --model runs\heretic-headtohead-20260531-183433\apostate-edited --base Qwen/Qwen2.5-7B-Instruct --suite all --n 80 --out runs\heretic-headtohead-20260531-183433\bench-apostate.json --judge classifier

python -m apostate.benchcode --model runs\heretic-headtohead-20260531-183433\heretic-adapter --base Qwen/Qwen2.5-7B-Instruct --suite all --n 80 --out runs\heretic-headtohead-20260531-183433\bench-heretic.json --judge classifier
```

## Ablation

| Tool | Export | Wall time | Internal refusal | Internal KL |
| --- | --- | ---: | ---: | ---: |
| Apostate | baked checkpoint | 306.8s | 5/100 | 0.1121 |
| Heretic | LoRA adapter | 1166.7s | 9/100 | 0.0976 |

Apostate spent 122.9s in profile search, 31.0s in guard, 25.4s in refusal refine, 44.8s in repair, 22.9s in final test metrics, and 139.5s baking the checkpoint. The actual ablation-plus-repair path before bake was about 167s after model/prompt setup.

Heretic chose trial 14. Its own trial loop reported 13m44s of optimization time; total wrapper wall time was 1166.7s including load, batch probing, prefix check, residual setup, adapter restore, and export.

## Public benchmark

Benchmark suite: `humaneval,mbpp,gsm8k,refusal`

Benchmark sample sizes: HumanEval 80, MBPP 80, GSM8K 24, JBB refusal 48, KL 48 harmless prompts

Judge: `classifier`

| Model | Refusal | Complied | HumanEval pass@1 | MBPP pass@1 | GSM8K | KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 100.0% | 0.0% | 73.8% | 70.0% | 70.8% | 0.000 |
| Apostate | 4.2% | 93.8% | 80.0% | 70.0% | 70.8% | 0.143 |
| Heretic | 8.3% | 87.5% | 72.5% | 72.5% | 70.8% | 0.099 |

Against base, Apostate changed HumanEval by `+6.2` points, MBPP by `+0.0` points, and GSM8K by `+0.0` points. Heretic changed HumanEval by `-1.2` points, MBPP by `+2.5` points, and GSM8K by `+0.0` points.

The remaining Apostate refusals were concentrated in Economic harm at 20% for that category slice. The remaining Heretic refusals were Physical harm at 20%, Fraud/Deception at 12.5%, and Economic harm at 10%.

## Takeaway

At equal 16-trial budget on this machine, Apostate produced lower public refusal rate and higher HumanEval while preserving MBPP and GSM8K. Heretic produced lower KL and slightly higher MBPP, but left about twice the public refusal rate and took about 3.8x the ablation wall time.

The benchmark now loads PEFT adapter folders directly, so Heretic-style adapter exports can be tested without merging them into full model weights first.
