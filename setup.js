#!/usr/bin/env node
"use strict";
// setup wizard

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
function toWsl(p) {
  p = p.replace(/\\/g, "/");
  if (/^[A-Za-z]:/.test(p)) p = "/mnt/" + p[0].toLowerCase() + p.slice(2);
  return p;
}
const optedYes = (a) => ["y", "yes"].includes((a || "").trim().toLowerCase());

const CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128";
const TORCH_DEPS = ["torch", "torchvision", "torchaudio"];
const PY_DEPS = [
  "transformers",
  "datasets",
  "safetensors",
  "optuna",
  "bitsandbytes",
  "numpy==2.2.6",
  "fsspec==2026.2.0",
];

(async () => {
  console.log("\n=== Apostate setup ===");
  console.log(`os: ${process.platform} ${os.arch()} | node: ${process.version}\n`);

  // prereqs
  for (const [bin, why] of [["node", "required"], [PY, "required"]]) {
    console.log(`${have(bin) ? "ok " : "MISSING"}  ${bin} (${why})`);
  }
  if (!have(PY)) {
    console.log(`\ninstall Python 3.10+ first, then re-run.\n`);
    rl.close();
    return;
  }

  // node deps
  console.log("\n[1/4] node deps ...");
  run("npm", ["install", "--omit=dev"]);

  // python deps
  const useCudaTorch = have("nvidia-smi");
  const torchLabel = useCudaTorch ? "cu128 torch" : "cpu torch";
  if (yes(await ask(`\n[2/4] install python deps (${torchLabel}, ${PY_DEPS.join(" ")})? [Y/n] `))) {
    const torchArgs = useCudaTorch
      ? ["-m", "pip", "install", "-U", "--force-reinstall", "--quiet", "--index-url", CUDA_TORCH_INDEX, ...TORCH_DEPS]
      : ["-m", "pip", "install", "-U", "--force-reinstall", "--quiet", ...TORCH_DEPS];
    if (!run(PY, torchArgs) && useCudaTorch) {
      console.log("  cuda torch install failed. Check python, driver, and wheel index.");
    }
    run(PY, ["-m", "pip", "install", "-U", "--quiet", ...PY_DEPS]);
  }

  // gpu check
  console.log("\n[3/4] gpu ...");
  console.log("  " + out(PY, ["-c",
    "import torch;print('torch', torch.__version__, 'cuda_build', torch.version.cuda, 'cuda', torch.cuda.is_available(),"
    + "(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only'))"]));

  // fast serving
  console.log("\n[4/4] fast serving (vLLM, optional) ...");
  const q = isWin
    ? "  set up vLLM via WSL now? (installs uv+vLLM in WSL, several GB) [y/N] "
    : "  install vLLM for fast serving? (several GB) [y/N] ";
  if (optedYes(await ask(q))) {
    if (isWin) {
      if (spawnSync("wsl", ["-e", "echo", "ok"], { stdio: "ignore" }).status === 0) {
        const script = toWsl(path.join(ROOT, "apostate", "vllm_serve.sh"));
        run("wsl", ["-u", "root", "bash", script, "setup"]);
      } else {
        console.log("  WSL not ready. Install once (admin PowerShell): wsl --install  then reboot and re-run setup.");
      }
    } else {
      run(PY, ["-m", "pip", "install", "-q", "vllm"]);
    }
  }

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
