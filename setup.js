#!/usr/bin/env node
"use strict";
// setup wizard: autodetect os, install deps, check gpu

const { spawnSync } = require("child_process");
const readline = require("readline");
const os = require("os");
const path = require("path");

const ROOT = __dirname;
const isWin = process.platform === "win32";
const PY = isWin ? "python" : "python3";

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
const ask = (q) => new Promise((r) => rl.question(q, r));
const yes = (a) => !["n", "no"].includes((a || "").trim().toLowerCase());

function run(cmd, args, opts) {
  console.log("  $ " + cmd + " " + args.join(" "));
  return spawnSync(cmd, args, { stdio: "inherit", cwd: ROOT, ...opts }).status === 0;
}
function out(cmd, args) {
  const r = spawnSync(cmd, args, { encoding: "utf8", cwd: ROOT });
  return ((r.stdout || "") + (r.stderr || "")).trim();
}
function have(cmd) {
  const r = spawnSync(isWin ? "where" : "which", [cmd], { encoding: "utf8" });
  return r.status === 0;
}

const PY_DEPS = ["torch", "transformers", "datasets", "safetensors", "optuna", "bitsandbytes"];

(async () => {
  console.log("\n=== Apostate setup ===");
  console.log(`os: ${process.platform} ${os.arch()} | node: ${process.version}\n`);

  // prerequisites
  for (const [bin, why] of [["node", "required"], [PY, "required"]]) {
    console.log(`${have(bin) ? "ok " : "MISSING"}  ${bin} (${why})`);
  }
  if (!have(PY)) {
    console.log(`\ninstall Python 3.10+ first, then re-run.\n`);
    rl.close();
    return;
  }

  // node deps
  console.log("\n[1/3] node deps ...");
  run("npm", ["install", "--omit=dev"]);

  // python deps
  if (yes(await ask(`\n[2/3] install python deps (${PY_DEPS.join(" ")})? [Y/n] `))) {
    run(PY, ["-m", "pip", "install", "-U", "--quiet", ...PY_DEPS]);
  }

  // gpu check
  console.log("\n[3/3] gpu ...");
  console.log("  " + out(PY, ["-c",
    "import torch;print('cuda', torch.cuda.is_available(),"
    + "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'))"]));

  console.log("\nready.");
  console.log("  apostate                      # menu");
  console.log("  apostate ablate --model <hf>  # decensor");
  if (isWin) {
    console.log("\nfast vLLM serving on Windows runs through WSL and self-installs:");
    console.log("  apostate talk --model <dir> --backend vllm");
  }
  console.log("");
  rl.close();
})();
