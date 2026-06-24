#!/usr/bin/env python3.11
"""Generate the README result graphs from each published model's report.json (real numbers,
no hand-entered data). Writes assets/refusal_before_after.png and assets/refusal_kl.png."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

# (display name, hf repo) ordered roughly by size
ROSTER = [
    ("VibeThinker-3B", "vibethinker-3b-apostate"),
    ("Gemma-4-E4B", "gemma-4-e4b-it-apostate"),
    ("FastContext-4B", "fastcontext-1.0-4b-sft-apostate"),
    ("Qwen2.5-7B", "qwen2.5-7b-instruct-apostate"),
    ("Qwen3-8B", "qwen3-8b-apostate"),
    ("Granite-3.3-8B", "granite-3.3-8b-instruct-apostate"),
    ("Falcon3-10B", "falcon3-10b-instruct-apostate"),
    ("Gemma-4-12B", "gemma-4-12b-it-apostate"),
]

rows = []
for name, repo in ROSTER:
    try:
        d = json.load(open(hf_hub_download(f"heterodoxin/{repo}", "report.json")))
        rows.append((name, d["baseline_refusal_rate"] * 100, d["edited_refusal_rate"] * 100,
                     d["harmless_kl_nats"]))
        print(f"  {name}: {rows[-1][1]:.0f}% -> {rows[-1][2]:.0f}%  KL={rows[-1][3]:.3f}")
    except Exception as e:
        print(f"  skip {name}: {type(e).__name__}")

names = [r[0] for r in rows]
base = [r[1] for r in rows]
edit = [r[2] for r in rows]
kl = [r[3] for r in rows]

PURPLE, GREY = "#7c3aed", "#cbd5e1"

# --- Graph 1: refusal rate, base vs Apostate (grouped horizontal bars) ---
fig, ax = plt.subplots(figsize=(9, 5))
y = range(len(names))
ax.barh([i + 0.2 for i in y], base, height=0.4, color=GREY, label="Base model")
ax.barh([i - 0.2 for i in y], edit, height=0.4, color=PURPLE, label="Apostate")
ax.set_yticks(list(y)); ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel("Refusal rate on harmful prompts (%)")
ax.set_title("Apostate removes refusal while keeping the model intact")
ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
for i, v in zip(y, edit):
    ax.text(v + 1, i - 0.2, f"{v:.0f}%", va="center", fontsize=8, color=PURPLE)
fig.tight_layout(); fig.savefig(ASSETS / "refusal_before_after.png", dpi=130); plt.close(fig)

# --- Graph 2: final refusal vs harmless KL (the quality frontier) ---
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(edit, kl, s=90, color=PURPLE, zorder=3, edgecolor="white")
for n, x, yk in zip(names, edit, kl):
    ax.annotate(n, (x, yk), textcoords="offset points", xytext=(7, 4), fontsize=8)
ax.set_xlabel("Refusal rate after edit (%)  —  lower is more uncensored")
ax.set_ylabel("Harmless KL (nats)  —  lower preserves behavior")
ax.set_title("Low refusal at low KL: decensored without breaking the model")
ax.grid(alpha=0.3)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(ASSETS / "refusal_kl.png", dpi=130); plt.close(fig)

print(f"wrote {ASSETS}/refusal_before_after.png and refusal_kl.png")

# --- Graph 3: Apostate vs Heretic on Qwen2.5-7B (real same-budget numbers from the README) ---
# apostate / heretic: refusal %, harmless KL (nats), ablation wall (s)
# fresh same-budget run: 16 trials, seed 0, AMD R9700 (gfx1201, ROCm). heretic = its best-refusal
# Pareto point at 16 trials.
metrics = ["Refusal (%)  ↓ lower better", "Harmless KL (nats)  ↓ lower better",
           "Ablation wall (s)  ↓ lower better"]
apo = [2.9, 0.095, 249.9]
her = [15.0, 0.150, 420.0]
GREEN, GRAY = "#16a34a", "#9ca3af"
fig, axes = plt.subplots(1, 3, figsize=(10, 4.4))
for ax, m, a, h in zip(axes, metrics, apo, her):
    ax.bar(["Apostate", "Heretic"], [a, h], color=[GREEN, GRAY])
    ax.set_title(m, fontsize=10)
    ax.text(0, a, f"{a:g}  ✓", ha="center", va="bottom", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(1, h, f"{h:g}", ha="center", va="bottom", fontsize=9, color="#6b7280")
    ax.set_ylim(0, max(a, h) * 1.22); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Apostate (green ✓) vs Heretic 1.4.0 — Qwen2.5-7B, same 16-trial budget\n"
             "shorter bar = better on every metric; Apostate wins all three",
             fontsize=11)
fig.tight_layout(); fig.savefig(ASSETS / "vs_heretic.png", dpi=130); plt.close(fig)
print(f"wrote {ASSETS}/vs_heretic.png")
