from __future__ import annotations

import json
import os
from typing import Any, Optional


def _pct(v: Any) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "n/a"


def _num(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "n/a"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _benchmark_rows(benchmark: Optional[dict]) -> list[list[str]]:
    if not benchmark:
        return []
    base = benchmark.get("base", {})
    cand = benchmark.get("candidate", {})
    rows: list[list[str]] = []
    if "pass@1" not in base and "pass@1" not in cand:
        code_suites = sorted(set((base.get("code") or {}).keys()) | set((cand.get("code") or {}).keys()))
        for suite in code_suites:
            bv = (base.get("code") or {}).get(suite, {}).get("pass@1")
            cv = (cand.get("code") or {}).get(suite, {}).get("pass@1")
            try:
                delta = float(cv) - float(bv)
                d = f"{delta * 100:+.1f} pts"
            except Exception:
                d = "n/a"
            rows.append([f"{suite} pass@1", _pct(bv), _pct(cv), d])
    for key, label, fmt in (
        ("refusal_rate", "Refusal", _pct),
        ("weak_rate", "Weak", _pct),
        ("noncompliance_rate", "Noncompliance", _pct),
        ("complied_rate", "Complied", _pct),
        ("pass@1", "Code pass@1", _pct),
        ("gsm8k", "GSM8K", _pct),
        ("kl_vs_base", "KL vs base", _num),
    ):
        if key not in base and key not in cand:
            continue
        bv, cv = base.get(key), cand.get(key)
        try:
            delta = float(cv) - float(bv)
            d = f"{delta * 100:+.1f} pts" if key != "kl_vs_base" else f"{delta:+.3f}"
        except Exception:
            d = "n/a"
        rows.append([label, fmt(bv), fmt(cv), d])
    return rows


def write_run_report(cfg: Any, report: dict, benchmark: Optional[dict] = None, command: Optional[str] = None) -> str:
    out_dir = _cfg_get(cfg, "output_dir", report.get("baked_to") or ".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.md")
    command = command or report.get("command")
    base_label = "Baseline refusal"
    if report.get("baseline_eval_n"):
        base_label = f"Baseline refusal (n={report.get('baseline_eval_n')})"

    lines = [
        "# Apostate Run Report",
        "",
        "## Summary",
        _table(
            ["Metric", "Value"],
            [
                ["Base model", report.get("model", _cfg_get(cfg, "model", "n/a"))],
                ["Profile", report.get("profile", _cfg_get(cfg, "profile", "n/a"))],
                ["Output", out_dir],
                ["Layers", report.get("num_layers", "n/a")],
                ["Hidden size", report.get("hidden_size", "n/a")],
                ["Direction layer", report.get("direction_layer", "n/a")],
                ["Refusal rank", report.get("refusal_subspace_rank", "n/a")],
                ["Max refusal rank", report.get("max_refusal_rank", _cfg_get(cfg, "max_rank", "n/a"))],
                ["Multi refusal", report.get("multi_refusal", _cfg_get(cfg, "multi_refusal", "n/a"))],
                ["Multi clusters", report.get("multi_refusal_clusters", _cfg_get(cfg, "multi_refusal_clusters", "n/a"))],
                ["Multi min coverage", report.get("multi_refusal_min_coverage", _cfg_get(cfg, "multi_refusal_min_coverage", "n/a"))],
                [base_label, _pct(report.get("baseline_refusal_rate"))],
                ["Edited refusal", _pct(report.get("edited_refusal_rate"))],
                ["Refusal metric", report.get("refusal_metric", "classifier + weak guard")],
                ["Harmless KL", _num(report.get("harmless_kl_nats"))],
                ["Target refusal", _pct(_cfg_get(cfg, "target_refusal", report.get("target_refusal")))],
                ["KL target", _num(report.get("kl_target", _cfg_get(cfg, "kl_target")))],
                ["KL budget", _num(_cfg_get(cfg, "max_kl", report.get("max_kl")))],
                ["KL positions", report.get("kl_positions", _cfg_get(cfg, "kl_positions", "n/a"))],
                ["KL layer trims", report.get("kl_layer_trim_steps", "n/a")],
                ["Repair steps", report.get("repair_steps", "n/a")],
                ["Preserve rank", report.get("preserve_rank", "n/a")],
                ["Preserve source", report.get("preserve_source", "n/a")],
                ["Capability penalty", report.get("opt_capability", _cfg_get(cfg, "opt_capability", "n/a"))],
                ["Elapsed", f"{report.get('elapsed_sec', 'n/a')} sec"],
            ],
        ),
    ]
    if command:
        lines += ["", "## Command", "", "```text", command, "```"]

    best = report.get("best_params")
    if best:
        lines += [
            "",
            "## Best Parameters",
            _table(["Parameter", "Value"], [[k, round(v, 4) if isinstance(v, float) else v] for k, v in best.items()]),
        ]

    trial = report.get("best_trial")
    if trial:
        lines += [
            "",
            "## Best Trial",
            _table(["Metric", "Value"], [[k, v] for k, v in trial.items()]),
        ]

    alphas = report.get("layer_alphas") or []
    if alphas:
        rows = [[i, _num(a)] for i, a in enumerate(alphas)]
        lines += ["", "## Layer Alphas", _table(["Layer", "Alpha"], rows)]

    guard = report.get("guard_history") or []
    if guard:
        keys = ["iter", "separation", "ratio", "rank", "refusal", "kl", "reverted"]
        lines += ["", "## Guard History", _table(keys, [[g.get(k, "") for k in keys] for g in guard])]

    bench_rows = _benchmark_rows(benchmark or report.get("benchmark"))
    if bench_rows:
        lines += [
            "",
            "## Benchmark Deltas",
            _table(["Metric", "Base", "Edited", "Delta"], bench_rows),
        ]

    if report.get("pruned_layers"):
        lines += ["", "## Pruning", f"Pruned layers: `{report.get('pruned_layers')}`."]

    timings = report.get("timings_sec") or {}
    if timings:
        lines += ["", "## Timings", _table(["Phase", "Seconds"], [[k, v] for k, v in timings.items()])]

    lines += [
        "",
        "## Measurement",
        _table(
            ["field", "value"],
            [
                ["refusal judge", "classifier + weak guard"],
                ["preservation metric", "harmless kl"],
                ["capability suites", "gsm8k, humaneval, mbpp"],
            ],
        ),
        "",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_model_card(cfg: Any, report: dict, benchmark: Optional[dict] = None, command: Optional[str] = None) -> str:
    out_dir = _cfg_get(cfg, "output_dir", report.get("baked_to") or ".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "README.md")
    base_model = report.get("model", _cfg_get(cfg, "model", "unknown"))
    command = command or report.get("command")

    lines = [
        "---",
        "library_name: transformers",
        f"base_model: {json.dumps(str(base_model))}",
        "tags:",
        "- apostate",
        "- model-edit",
        "---",
        "",
        "# Apostate Edited Model",
        "",
        f"Base model: `{base_model}`",
        "",
        "## Metrics",
        _table(
            ["Metric", "Value"],
            [
                ["Baseline refusal", _pct(report.get("baseline_refusal_rate"))],
                ["Edited refusal", _pct(report.get("edited_refusal_rate"))],
                ["Refusal metric", report.get("refusal_metric", "classifier + weak guard")],
                ["Harmless KL", _num(report.get("harmless_kl_nats"))],
                ["KL target", _num(report.get("kl_target"))],
                ["Preserve rank", report.get("preserve_rank", "n/a")],
                ["Preserve source", report.get("preserve_source", "n/a")],
                ["Refusal rank", report.get("refusal_subspace_rank", "n/a")],
                ["Multi refusal", report.get("multi_refusal", _cfg_get(cfg, "multi_refusal", "n/a"))],
                ["Direction layer", report.get("direction_layer", "n/a")],
                ["Elapsed", f"{report.get('elapsed_sec', 'n/a')} sec"],
            ],
        ),
    ]

    bench_rows = _benchmark_rows(benchmark or report.get("benchmark"))
    if bench_rows:
        lines += ["", "## Benchmark Deltas", _table(["Metric", "Base", "Edited", "Delta"], bench_rows)]

    if command:
        lines += ["", "## Reproduction", "", "```bash", command, "```"]

    lines += [
        "",
        "## Measurement",
        _table(
            ["field", "value"],
            [
                ["edit type", "weight projection"],
                ["refusal judge", "classifier + weak guard"],
                ["preservation metric", "harmless kl"],
            ],
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_benchmark_report(report: dict, out_json: Optional[str]) -> Optional[str]:
    if not out_json:
        return None
    path = os.path.splitext(out_json)[0] + ".md"
    rows = _benchmark_rows(report)
    lines = [
        "# Apostate Benchmark Report",
        "",
        f"Suite: `{report.get('suite', 'unknown')}`",
        f"Judge: `{report.get('judge', 'unknown')}`",
        "",
        _table(["Metric", "Base", "Edited", "Delta"], rows) if rows else "No comparable metrics were recorded.",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def refresh_candidate_reports(candidate: str, benchmark: dict):
    if not os.path.isdir(candidate):
        return
    report_path = os.path.join(candidate, "report.json")
    if not os.path.isfile(report_path):
        return
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return
    report["benchmark"] = benchmark
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cfg = {
        "output_dir": candidate,
        "model": report.get("model"),
        "target_refusal": report.get("target_refusal"),
        "max_kl": report.get("max_kl"),
    }
    write_run_report(cfg, report, benchmark=benchmark, command=report.get("command"))
    write_model_card(cfg, report, benchmark=benchmark, command=report.get("command"))
