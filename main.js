#!/usr/bin/env node
"use strict";
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const os = require("os");

const ROOT = path.resolve(__dirname);
const PY = process.platform === "win32" ? "python" : "python3";

function hfCacheRoots() {
  const roots = [];
  const add = (p) => {
    if (!p) return;
    const full = path.resolve(p);
    if (!roots.includes(full) && fs.existsSync(full)) roots.push(full);
  };
  add(process.env.HUGGINGFACE_HUB_CACHE);
  if (process.env.HF_HOME) add(path.join(process.env.HF_HOME, "hub"));
  add(path.join(os.homedir(), ".cache", "huggingface", "hub"));
  return roots;
}

function hasModelSnapshot(dir) {
  if (fs.existsSync(path.join(dir, "config.json"))) return true;
  const snapRoot = path.join(dir, "snapshots");
  let snaps = [];
  try { snaps = fs.readdirSync(snapRoot, { withFileTypes: true }); } catch { return false; }
  return snaps.some((s) => {
    if (!s.isDirectory()) return false;
    const p = path.join(snapRoot, s.name);
    return fs.existsSync(path.join(p, "config.json")) ||
           fs.existsSync(path.join(p, "tokenizer_config.json")) ||
           fs.existsSync(path.join(p, "processor_config.json"));
  });
}

function findHFModels() {
  const out = [];
  const seen = new Set();
  for (const root of hfCacheRoots()) {
    let entries = [];
    try { entries = fs.readdirSync(root, { withFileTypes: true }); } catch { continue; }
    for (const d of entries) {
      if (!d.isDirectory() || !d.name.startsWith("models--")) continue;
      const id = d.name.slice("models--".length).split("--").join("/");
      if (!id || seen.has(id.toLowerCase())) continue;
      const dir = path.join(root, d.name);
      if (!hasModelSnapshot(dir)) continue;
      seen.add(id.toLowerCase());
      out.push(id);
    }
  }
  return out.sort((a, b) => a.localeCompare(b));
}

async function run(args, commandLabel) {
  return new Promise((resolve) => {
    const env = {
      ...process.env,
      APOSTATE_COMMAND: commandLabel || process.env.APOSTATE_COMMAND || `${PY} ${args.join(" ")}`,
      PYTHONPATH: ROOT,
      PYTHONUNBUFFERED: "1"
    };
    const proc = spawn(PY, args, { stdio: "inherit", env, cwd: process.cwd() });
    proc.on("close", (code) => resolve(code || 0));
  });
}

(async () => {
  let [, , cmd, ...args] = process.argv;

  if (!cmd) {
    cmd = "tui";
  } else if (cmd === "-h" || cmd === "--help") {
    console.log(`
  tui                             interactive menu (default)
  setup                           install deps, check gpu (wizard)
  ablate [--model M] [--out D]    remove refusals (--resume reuses activation cache)
  test   [--model D] [--base M]   benchmark (--suite humaneval,mbpp,gsm8k,refusal or all)
  talk   [--model D] [--backend]  chat (vllm: --kv-cache-dtype fp8|turboquant_4bit_nc)
  list                            checkpoints
    `);
    return;
  }

  const getFlag = (a, name, def) => {
    const i = a.indexOf(name);
    return (i >= 0 && i + 1 < a.length) ? a[i + 1] : def;
  };
  const stripFlags = (a, names) => {
    const out = [];
    for (let i = 0; i < a.length; i++) {
      if (names.includes(a[i])) {
        i++;
      } else {
        out.push(a[i]);
      }
    }
    return out;
  };

  let pyCmd = [];
  if (cmd === "tui") {
    const tuiPath = path.join(ROOT, "tui.js");
    const proc = spawn(process.execPath, [tuiPath], { stdio: "inherit" });
    proc.on("close", (code) => process.exit(code || 0));
    return;
  } else if (cmd === "setup") {
    const proc = spawn(process.execPath, [path.join(ROOT, "setup.js"), ...args], { stdio: "inherit" });
    proc.on("close", (code) => process.exit(code || 0));
    return;
  } else if (cmd === "boost" || cmd === "ablate") {
    const model = getFlag(args, "--model", "Qwen/Qwen2.5-7B-Instruct");
    const out = getFlag(args, "--out", getFlag(args, "--output-dir", "out"));
    pyCmd = [
      "-m", "apostate.cli", "--optimize", "--model", model, "--output-dir", out,
      ...stripFlags(args, ["--model", "--out", "--output-dir"]),
    ];
  } else if (cmd === "turbo") {
    const model = getFlag(args, "--model", "Qwen/Qwen2.5-7B-Instruct");
    const out = getFlag(args, "--out", "out");
    console.log("Step 1: Finetune...");
    await run(["-m", "apostate.finetune", "--model", model, "--out", out + "_ft"], `apostate turbo --model ${model} --out ${out}`);
    console.log("Step 2: Abliterate...");
    await run(["-m", "apostate.cli", "--optimize", "--model", out + "_ft", "--output-dir", out], `apostate turbo --model ${model} --out ${out}`);
    console.log("Step 3: Cleanup intermediate...");
    fs.rmSync(out + "_ft", { recursive: true, force: true });
    console.log("Step 4: Verify...");
    await run(["-m", "apostate.benchcode", "--model", out, "--base", model], `apostate turbo --model ${model} --out ${out}`);
    console.log("Done!");
    return;
  } else if (cmd === "test") {
    pyCmd = ["-m", "apostate.benchcode", ...args];
  } else if (cmd === "talk") {
    pyCmd = ["-m", "apostate.chat", ...args];
  } else if (cmd === "quantize") {
    pyCmd = ["-m", "apostate.quantize", ...args];
  } else if (cmd === "train") {
    pyCmd = ["-m", "apostate.finetune", ...args];
  } else if (cmd === "list") {
    const seen = new Set();
    console.log("hf cache:");
    for (const id of findHFModels()) {
      console.log("  " + id);
    }
    console.log("");
    console.log("checkpoints:");
    for (const base of [process.cwd(), ROOT]) {
      try {
        for (const d of fs.readdirSync(base, { withFileTypes: true })) {
          if (!d.isDirectory() || /_merged$/.test(d.name)) continue;
          const dir = path.join(base, d.name);
          if (seen.has(dir)) continue;
          const f = fs.readdirSync(dir);
          if (f.includes("config.json") && f.some(x => x.endsWith(".safetensors"))) {
            seen.add(dir);
            console.log("  " + dir);
          }
        }
      } catch { }
    }
    return;
  } else {
    console.error("unknown command: " + cmd);
    process.exit(1);
  }

  const code = await run(pyCmd, `apostate ${cmd} ${args.join(" ")}`.trim());
  process.exit(code);
})();
