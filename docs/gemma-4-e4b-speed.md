# Gemma 4 E4B speed notes

Run: `runs/gemma-fast-test-20260601-131657`

Command:

```text
apostate ablate --model google/gemma-4-E4B-it --out C:\Users\Levit\apostate\runs\gemma-fast-test-out5 --resume --activation-cache-dir C:\Users\Levit\OneDrive\Desktop\apostatehfmodels\gemma-4-e4b-it-apostate\activation_cache --no-bake
```

Result:

| metric | value |
| --- | ---: |
| wall time | 370.4s |
| load model | 28.4s |
| prompt load | 0.0s |
| baseline refusal | 16.9s |
| activation fit | 0.0s |
| head sweep | 129.5s |
| final test metrics | 195.5s |
| validation refusal | 0.0833 |
| validation KL | 0.0995 |
| test refusal | 0.0900 |
| test KL | 0.1205 |
| head alpha | 4.65 |

Notes:

- Gemma 4 E4B uses the head-token path because per-layer embeddings make the normal residual activation edit ineffective.
- Prompt caching removed dataset load time on resumed runs.
- Activation fit, guard, refine, and repair are skipped for the head-token profile.
- The probe grid uses a logit proxy; exact rerank still uses the rejection classifier.
- The remaining wall-time bottleneck is full final test reporting, not the ablation search.
