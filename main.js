#!/usr/bin/env node
"use strict";
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const ROOT = path.resolve(__dirname);
const PY = process.platform === "win32" ? "python" : "python3";

// run module
async function run(args) {
  return new Promise((resolve) => {
    const env = {
      ...process.env,
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
  ablate [--model M] [--out D]    remove refusals
  test   [--model D] [--base M]   benchmark
  talk   [--model D] [--backend]  chat (backend: local | vllm)
  list                            checkpoints
    `);
    return;
  }

  // read flag
  const getFlag = (a, name, def) => {
    const i = a.indexOf(name);
    return (i >= 0 && i + 1 < a.length) ? a[i + 1] : def;
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
    pyCmd = ["-m", "apostate.cli", "--optimize", "--model", model, "--output-dir", out];
  } else if (cmd === "turbo") {
    const model = getFlag(args, "--model", "Qwen/Qwen2.5-7B-Instruct");
    const out = getFlag(args, "--out", "out");
    console.log("Step 1: Finetune...");
    await run(["-m", "apostate.finetune", "--model", model, "--out", out + "_ft"]);
    console.log("Step 2: Abliterate...");
    await run(["-m", "apostate.cli", "--optimize", "--model", out + "_ft", "--output-dir", out]);
    console.log("Step 3: Cleanup intermediate...");
    fs.rmSync(out + "_ft", { recursive: true, force: true });
    console.log("Step 4: Verify...");
    await run(["-m", "apostate.benchcode", "--model", out, "--base", model]);
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

  const code = await run(pyCmd);
  process.exit(code);
})();
