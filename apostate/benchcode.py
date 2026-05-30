"""benchmark vs base."""

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
    # humaneval is the default and the deepswe fallback
    return load_code_problems("openai/openai_humaneval:test", n), "humaneval"


def benchmark_vs_base(
    candidate: str, base: str, suite: str = "humaneval", n: int = 80,
    harmful_spec: str = "mlabonne/harmful_behaviors:test:text",
    gen_tokens: int = 256, batch: int = 8, timeout: int = 8,
    out_json: Optional[str] = None,
) -> dict:
    deepswe_note = None
    if suite == "deepswe":
        deepswe_note = ("DeepSWE is an agentic, container-isolated multi-language benchmark "
                        "(needs the datacurve harness + Docker per task); not runnable for a "
                        "single-shot local 7B. Falling back to executable HumanEval.")
        print(f"[bench] NOTE: {deepswe_note}", flush=True)
        suite = "humaneval"

    problems, suite_name = _coding_problems(suite, n)
    refusal_n = max(20, min(48, n // 2 + 16))   # reduce for speed
    gsm_n = max(20, min(24, n // 2))             # reduce for speed
    print(f"[bench] coding={suite_name}({len(problems)})  refusal=JBB({refusal_n})  capability=GSM8K({gsm_n})", flush=True)

    def eval_model(path, ref_lp):
        m, tok = _load(path)
        ref = refusal_eval(m, tok, refusal_n, 48, batch)                  # JBB: refusal + compliance + categories
        p1, comp = pass_at_1_wrap(m, tok, problems, gen_tokens, batch, timeout)
        gsm = gsm8k_eval(m, tok, gsm_n, 320, batch)                       # capability retention
        lp = _logprobs_lastK(m, tok, resolve_prompts("mlabonne/harmless_alpaca:test:text", 48, 0), 8, batch)
        kl = 0.0 if ref_lp is None else _kl(ref_lp, lp)
        del m; gc.collect(); torch.cuda.empty_cache()
        return {"refusal_rate": ref["refusal_rate"], "complied_rate": ref["complied_rate"],
                "category_refusal": ref["category_refusal"], "pass@1": round(p1, 4),
                "gsm8k": gsm["accuracy"], "kl_vs_base": round(kl, 4)}, lp

    print(f"[bench] base: {base}", flush=True)
    base_res, base_lp = eval_model(base, None)
    print(f"[bench] candidate: {candidate}", flush=True)
    cand_res, _ = eval_model(candidate, base_lp)

    report = {"suite": suite_name, "n": len(problems),
              "base": {"path": base, **base_res}, "candidate": {"path": candidate, **cand_res},
              "pass@1_delta": round(cand_res["pass@1"] - base_res["pass@1"], 4),
              "gsm8k_delta": round(cand_res["gsm8k"] - base_res["gsm8k"], 4)}
    if deepswe_note:
        report["deepswe_note"] = deepswe_note

    def pct(x): return f"{x*100:.1f}"
    rows = [("model", "refusal%", "complied%", "pass@1%", "gsm8k%", "KL"),
            ("BASE", pct(base_res["refusal_rate"]), pct(base_res["complied_rate"]),
             pct(base_res["pass@1"]), pct(base_res["gsm8k"]), "0.000"),
            ("EDITED", pct(cand_res["refusal_rate"]), pct(cand_res["complied_rate"]),
             pct(cand_res["pass@1"]), pct(cand_res["gsm8k"]), f"{cand_res['kl_vs_base']:.3f}")]
    w = [max(len(r[i]) for r in rows) for i in range(6)]
    print(f"\n=== EDITED vs BASE  (decensoring + capability) ===")
    for j, r in enumerate(rows):
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(6)))
        if j == 0:
            print("  " + "  ".join("-" * w[i] for i in range(6)))
    print(f"\n  coding pass@1 {report['pass@1_delta']*100:+.1f} pts   "
          f"capability(gsm8k) {report['gsm8k_delta']*100:+.1f} pts")
    surviving = {c: r for c, r in cand_res["category_refusal"].items() if r > 0}
    if surviving:
        top = list(surviving.items())[:4]
        print("  refusals still seen in: " + ", ".join(f"{c} ({int(r*100)}%)" for c, r in top))

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def pass_at_1_wrap(model, tok, problems, gen_tokens, batch, timeout):
    class _B:  # minimal bundle adapter for codeeval.pass_at_1
        pass
    b = _B(); b.tokenizer = tok; b.model = model
    return pass_at_1(b, problems, gen_tokens, batch, True, timeout)


def main(argv=None):
    p = argparse.ArgumentParser(prog="apostate.benchcode")
    p.add_argument("--model", required=True, help="edited model path/id")
    p.add_argument("--base", required=True, help="base model path/id")
    p.add_argument("--suite", default="humaneval", choices=["humaneval", "mbpp", "deepswe"])
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--out", default="benchcode.json")
    a = p.parse_args(argv)
    benchmark_vs_base(a.model, a.base, a.suite, a.n, out_json=a.out)


if __name__ == "__main__":
    main()
