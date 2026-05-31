// tui harness
"use strict";
const pty = require("node-pty");
const path = require("path");
const { Terminal } = require("@xterm/headless");

const COLS = 120, ROWS = 40;
const steps = (process.argv[2] || "").split(",").filter(Boolean); // step list

const termEmu = new Terminal({ cols: COLS, rows: ROWS, allowProposedApi: true });

const child = pty.spawn(process.platform === "win32" ? "node.exe" : "node",
  [path.join(__dirname, "tui.js")],
  { name: "xterm-256color", cols: COLS, rows: ROWS, cwd: __dirname, env: process.env });

child.onData((d) => termEmu.write(d));

const KEYS = {
  enter: "\r", esc: "\x1b", up: "\x1b[A", down: "\x1b[B",
  space: " ",
  "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "0": "0",
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function render(label) {
  const buf = termEmu.buffer.active;
  const lines = [];
  for (let y = 0; y < ROWS; y++) {
    const line = buf.getLine(y);
    lines.push(line ? line.translateToString(true) : "");
  }
  // trim blanks
  while (lines.length && !lines[lines.length - 1].trim()) lines.pop();
  console.log(`\n===== ${label} =====`);
  console.log(lines.join("\n"));
}

(async () => {
  await sleep(1500);
  render("AFTER BOOT");
  for (const s of steps) {
    if (s.startsWith("wait")) {
      const ms = parseInt(s.slice(4)) || 1500;
      await sleep(ms);
      render(`AFTER wait ${ms}`);
      continue;
    }
    child.write(KEYS[s] ?? s);
    await sleep(1300);
    render(`AFTER key '${s}'`);
  }
  await sleep(200);
  child.kill();
  process.exit(0);
})();
