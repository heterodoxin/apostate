# Apostate

Removes refusals from instruction-tuned LLMs by ablating the refusal direction in the residual stream. It edits the weights in place. There is no finetuning and no second model.

## Results

Qwen2.5-7B-Instruct, 4-bit, one consumer GPU:

| metric | base | apostate |
|--------|------|----------|
| refusal rate | 95% | 5% |
| harmless KL (nats) | 0 | 0.16 |
| wall time | n/a | 749s |
| disk overhead | n/a | 0 (in place) |

Refusal is scored by a classifier (`protectai/distilroberta-base-rejection-v1`), not by keyword matching, so soft refusals and topic-dodges count instead of slipping through. The numbers come from a held-out test split. The search tunes its hyperparameters on a validation half and reports refusal and KL on a disjoint test half it never saw during tuning. Harmful eval prompts are JailbreakBench plus the harmful_behaviors test set; harmless prompts are the harmless_alpaca test set. The search itself is 12 TPE trials over 28 layers with a rank-1 subspace.

## How it works

Everything runs in one pass on a 4-bit model with forward hooks, so the search never reloads weights between trials.

1. Direction. Rank-1 mean difference between harmful and harmless residuals.
2. Targeting. Per-layer edit strength from single-layer activation patching.
3. Preservation. Project protected directions out of the edit (Gram-Schmidt).
4. Guard. Re-measure leakage and re-ablate the layers where refusal creeps back.
5. Refine. Raise strength until refusal reaches the target, capped by a KL budget.
6. Bake. Fold the projection into fp16/bf16 weights and write a standalone model.

The TPE objective trades refusal against harmless KL. Dense and MoE layers are both handled. For MoE the edit is applied to every expert's down-projection, not just one MLP.

## Install

The wizard detects your OS, installs the node and python deps, and checks the GPU:

```
apostate setup        # or: npm run setup
```

On Windows you can double-click `Apostate-Setup.cmd`, or build a standalone `Apostate-Setup.exe` with `npm run build-exe`. On Linux or macOS run `./setup.sh`. By hand:

```
npm install
pip install torch transformers datasets safetensors optuna bitsandbytes
```

## Use

```
apostate                 # menu
apostate ablate --model Qwen/Qwen2.5-7B-Instruct --out qwen-apostate
apostate test  --model qwen-apostate --base Qwen/Qwen2.5-7B-Instruct
apostate talk  --model qwen-apostate --quant nf4
```

`talk --quant` decides how the model is loaded for chat. `bf16` and `fp16` skip quantization and are fastest when the model fits in VRAM. `nf4`, `fp4`, and `int8` use bitsandbytes and load right away. `gptq` and `marlin` use the int4 Marlin kernel (fastest 4-bit on Ampere and newer), but need `pip install gptqmodel optimum` and quantize on the first load. `awq` loads a checkpoint that is already AWQ-quantized.

`talk --backend vllm` (or the vllm entry in the menu) serves through vLLM for higher throughput. The first run installs vLLM and starts a local server, then streams from it. vLLM has no native Windows build, so on Windows this runs inside WSL and sets itself up there.

## Architectures

Detected from the module layout, with dense and MoE handled the same way:

Llama 2/3, Qwen2/2.5/3 (dense and MoE), Mistral, Mixtral, DeepSeek and DeepSeek-V2, Gemma and Gemma 2, Phi-3, GPT-NeoX, Pythia.

Some are not covered yet. Falcon names its attention block differently, so only the MLP would be edited. Phi-2 calls its MLP down-projection `fc2`. GPT-2 style models store weights as Conv1D, which need transposed edit math. State-space models like Mamba and RWKV have no attention or MLP residual writers, so they are out of scope.

## Data

The fit set mixes `mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca` with the prompts bundled in `data/`, 600 of each. The held-out eval pulls real benchmarks (JailbreakBench, harmful_behaviors test, harmless_alpaca test). Any of these can point at your own data with a `repo:split:col` spec, `repo@config:split:col`, or several sources joined with `|`.

## Notes

A bf16 7B is around 14GB, which leaves no room for a KV cache on a 16GB card that is also driving a desktop, so the vLLM path loads in 4-bit by default. KL is measured on benign prompts and stands in for capability; it is not a full benchmark of reasoning or coding.

## Requires

Python 3.10+, Node 18+, CUDA, and about 16GB of VRAM for a 7B.
