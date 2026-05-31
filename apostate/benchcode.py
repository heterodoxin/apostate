"""bench compare"""

from __future__ import annotations

from typing import List, Optional
import argparse
import gc
import json
import torch

from .data import resolve_prompts
from .codeeval import load_code_problems, pass_at_1
from .benchmark import _load, _logprobs_lastK, _kl
from .evalsuite import refusal_eval, gsm8k_eval
from .reports import refresh_candidate_reports, write_benchmark_report


def _mbpp_as_humaneval(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    out = []
    for i in range(min(n, len(ds))):
        r = ds[i]
        tests = "\n".join(r["test_list"])
        out.append({"prompt": r["text"] + "\nWrite a Python function.\n",
                    "canonical_solution": r["code"], "test": tests, "entry_point": ""})
    return out


def _coding_problems(suite: str, n: int):
    if suite == "mbpp":
        return _mbpp_as_humaneval(n), "mbpp"
    # eval fallback
    return load_code_problems("openai/openai_humaneval:test", n), "humaneval"


def benchmark_vs_base(
    candidate: str, base: str, suite: str = "humaneval", n: int = 80,
    harmful_spec: str = "mlabonne/harmful_behaviors:test:text",
    gen_tokens: int = 256, batch: int = 8, timeout: int = 8,
    out_json: Optional[str] = None, judge: str = "classifier",
) -> dict:
    deepswe_note = None
    run_code = suite in ("humaneval", "mbpp", "deepswe")
    run_gsm = suite in ("humaneval", "mbpp", "deepswe", "gsm8k")
    run_refusal = suite in ("humaneval", "mbpp", "deepswe", "gsm8k", "refusal")

    if suite == "deepswe":
        deepswe_note = ("DeepSWE is an agentic, container-isolated multi-language benchmark "
                        "(needs the datacurve harness + Docker per task); not runnable for a "
                        "single-shot local 7B. Falling back to executable HumanEval.")
        print(f"[bench] NOTE: {deepswe_note}", flush=True)
        coding_suite = "humaneval"
    else:
        coding_suite = suite

    problems, suite_name = _coding_problems(coding_suite, n) if run_code else ([], suite)
    refusal_n = max(20, min(48, n // 2 + 16))   # speed cap
    gsm_n = max(20, min(24, n // 2))             # speed cap
    parts = []
    if run_code:
        parts.append(f"coding={suite_name}({len(problems)})")
    if run_refusal:
        parts.append(f"refusal=JBB({refusal_n})")
    if run_gsm:
        parts.append(f"capability=GSM8K({gsm_n})")
    parts.append("KL=harmless_alpaca(48)")
    print("[bench] " + "  ".join(parts), flush=True)

    def eval_model(path, ref_lp):
        m, tok = _load(path)
        res = {}
        if run_refusal:
            ref = refusal_eval(m, tok, refusal_n, 48, batch, judge=judge)  # jbb eval
            res.update({"refusal_rate": ref["refusal_rate"], "complied_rate": ref["complied_rate"],
                        "category_refusal": ref["category_refusal"]})
        if run_code:
            p1, comp = pass_at_1_wrap(m, tok, problems, gen_tokens, batch, timeout)
            res.update({"pass@1": round(p1, 4), "code_complete": round(comp, 4)})
        if run_gsm:
            gsm = gsm8k_eval(m, tok, gsm_n, 320, batch)                   # capability retention
            res["gsm8k"] = gsm["accuracy"]
        lp = _logprobs_lastK(m, tok, resolve_prompts("mlabonne/harmless_alpaca:test:text", 48, 0), 8, batch)
        kl = 0.0 if ref_lp is None else _kl(ref_lp, lp)
        del m; gc.collect(); torch.cuda.empty_cache()
        res["kl_vs_base"] = round(kl, 4)
        return res, lp

    print(f"[bench] base: {base}", flush=True)
    base_res, base_lp = eval_model(base, None)
    print(f"[bench] candidate: {candidate}", flush=True)
    cand_res, _ = eval_model(candidate, base_lp)

    report_n = len(problems) if run_code else (gsm_n if run_gsm else refusal_n if run_refusal else 0)
    report_suite = "deepswe->humaneval" if suite == "deepswe" else suite_name
    report = {"suite": report_suite, "n": report_n,
              "judge": judge,
              "base": {"path": base, **base_res}, "candidate": {"path": candidate, **cand_res},
              }
    if "pass@1" in cand_res and "pass@1" in base_res:
        report["pass@1_delta"] = round(cand_res["pass@1"] - base_res["pass@1"], 4)
    if "gsm8k" in cand_res and "gsm8k" in base_res:
        report["gsm8k_delta"] = round(cand_res["gsm8k"] - base_res["gsm8k"], 4)
    if deepswe_note:
        report["deepswe_note"] = deepswe_note

    def pct(x): return f"{x*100:.1f}"
    cols = [("model", None)]
    if "refusal_rate" in base_res or "refusal_rate" in cand_res:
        cols.extend([("refusal%", "refusal_rate"), ("complied%", "complied_rate")])
    if "pass@1" in base_res or "pass@1" in cand_res:
        cols.append(("pass@1%", "pass@1"))
    if "gsm8k" in base_res or "gsm8k" in cand_res:
        cols.append(("gsm8k%", "gsm8k"))
    cols.append(("KL", "kl_vs_base"))

    def row(label, res, is_base=False):
        vals = [label]
        for _, key in cols[1:]:
            if key == "kl_vs_base" and is_base:
                vals.append("0.000")
            elif key in res:
                vals.append(pct(res[key]) if key != "kl_vs_base" else f"{res[key]:.3f}")
            else:
                vals.append("n/a")
        return tuple(vals)

    rows = [tuple(c[0] for c in cols), row("BASE", base_res, True), row("EDITED", cand_res)]
    w = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    print(f"\n=== EDITED vs BASE  (decensoring + capability) ===")
    for j, r in enumerate(rows):
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(cols))))
        if j == 0:
            print("  " + "  ".join("-" * w[i] for i in range(len(cols))))
    delta_bits = []
    if "pass@1_delta" in report:
        delta_bits.append(f"coding pass@1 {report['pass@1_delta']*100:+.1f} pts")
    if "gsm8k_delta" in report:
        delta_bits.append(f"capability(gsm8k) {report['gsm8k_delta']*100:+.1f} pts")
    if delta_bits:
        print("\n  " + "   ".join(delta_bits))
    surviving = {c: r for c, r in cand_res.get("category_refusal", {}).items() if r > 0}
    if surviving:
        top = list(surviving.items())[:4]
        print("  refusals still seen in: " + ", ".join(f"{c} ({int(r*100)}%)" for c, r in top))

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        write_benchmark_report(report, out_json)
    refresh_candidate_reports(candidate, report)
    return report


def pass_at_1_wrap(model, tok, problems, gen_tokens, batch, timeout):
    class _B:  # adapter
        pass
    b = _B(); b.tokenizer = tok; b.model = model
    return pass_at_1(b, problems, gen_tokens, batch, True, timeout)


def main(argv=None):
    p = argparse.ArgumentParser(prog="apostate.benchcode")
    p.add_argument("--model", required=True, help="edited model path/id")
    p.add_argument("--base", required=True, help="base model path/id")
    p.add_argument("--suite", default="humaneval", choices=["humaneval", "mbpp", "deepswe", "gsm8k", "refusal"])
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", default="benchcode.json")
    p.add_argument("--judge", default="classifier", choices=["classifier", "keyword"])
    a = p.parse_args(argv)
    benchmark_vs_base(a.model, a.base, a.suite, a.n, out_json=a.out, judge=a.judge)


if __name__ == "__main__":
    main()
