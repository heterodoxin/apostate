#!/usr/bin/env node
"use strict";

const blessed = require('blessed');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const C = {
    acc:  '#cba6f7',
    dim:  '#6c7086',
    fg:   '#cdd6f4',
    sub:  '#bac2de',
    grn:  '#a6e3a1',
    red:  '#f38ba8',
    ylw:  '#f9e2af',
    bdr:  '#313244',
};

process.env.COLORTERM = 'truecolor';
process.env.TERM = 'xterm-256color';

const { execSync } = require('child_process');

const isWindowsTerminal = () => process.env.WT_SESSION || process.env.WT_PROFILE_ID;

const getWinConsoleSize = () => {
  try {
    if (isWindowsTerminal()) {
      const output = execSync('powershell -NoProfile -Command "[Console]::WindowWidth, [Console]::WindowHeight" 2>$null', { encoding: 'utf-8' });
      const lines = output.trim().split('\n');
      if (lines.length >= 2) {
        const cols = parseInt(lines[0]);
        const rows = parseInt(lines[1]);
        if (!isNaN(cols) && !isNaN(rows) && cols > 0 && rows > 0) {
          return { cols, rows };
        }
      }
    } else {
      const output = execSync('mode con', { encoding: 'utf-8' });
      const lines = output.split('\n');
      let cols = 120, rows = 30;
      for (const line of lines) {
        if (line.includes('Columns:')) {
          const match = line.match(/(\d+)/);
          if (match) cols = parseInt(match[1]);
        }
        if (line.includes('Lines:')) {
          const match = line.match(/(\d+)/);
          if (match) rows = parseInt(match[1]);
        }
      }
      return { cols, rows };
    }
  } catch (e) {}
  return { cols: 120, rows: 30 };
};

let screen;

const forceWindowsTerminalResize = () => {
  try {
    const { execSync } = require('child_process');
    execSync(`python -c "import ctypes as c,ctypes.wintypes as w,time,uuid;u=c.WinDLL('user32',use_last_error=True);k=c.WinDLL('kernel32',use_last_error=True);old=c.create_unicode_buffer(512);k.GetConsoleTitleW(old,512);t='WT_'+uuid.uuid4().hex;k.SetConsoleTitleW(t);time.sleep(.2);m=[];P=c.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM);cb=P(lambda h,l:(u.GetWindowTextW(h,(buf:=c.create_unicode_buffer(u.GetWindowTextLengthW(h)+1)),u.GetWindowTextLengthW(h)+1),(m.append(h) if t in buf.value else None),not bool(m))[2]);u.EnumWindows(cb,0);k.SetConsoleTitleW(old.value);r=w.RECT();h=m[0];u.ShowWindow(h,9);u.GetWindowRect(h,c.byref(r));u.SetWindowPos(h,None,r.left,r.top,int((r.right-r.left)*1.01),int((r.bottom-r.top)*1.01),0x40)"`, { stdio: 'ignore' });
  } catch (e) {}
};

forceWindowsTerminalResize();
{
    const consoleSize = getWinConsoleSize();

    screen = blessed.screen({
      mouse: true,
      keyboard: true,
      dockBorders: false,
      useBCE: true,
      width: consoleSize.cols,
      height: consoleSize.rows,
    });

  const LOTUS_TEXT = `⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠟⠹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣷⣦⣄⣠⣿⠃⢠⣄⠈⢻⣆⣠⣴⡞⡆⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣀⣀⣿⠀⠈⢻⣇⢀⣾⢟⡄⣸⡿⠋⠀⡇⣇⣀⣀⠀⠀⠀⠀⠀
⠀⣤⣤⣤⣀⣱⢻⠚⠻⣧⣀⠀⢹⡿⠃⠈⢻⣟⠀⢀⣤⠧⠓⣹⣟⣀⣤⣤⣤⡀
⠀⠈⠻⣧⠉⠛⣽⠀⠀⠀⠙⣷⡿⠁⠀⠀⠀⢻⣶⠛⠁⠀⠀⡟⠟⠉⣵⡟⠁⠀
⠀⠀⠀⠹⣧⡀⠏⡇⠀⠀⠀⣿⠁⠀⠀⠀⠀⠀⣿⡄⠀⠀⢠⢷⠀⣼⡟⠀⠀⠀
⠀⠀⠀⠀⠙⣟⢼⡹⡄⠀⠀⣿⡄⠀⠀⠀⠀⢀⣿⡇⠀⢀⣞⣦⢾⠟⠀⠀⠀⠀
⠀⠠⢶⣿⣛⠛⢒⣭⢻⣶⣤⣹⣿⣤⣀⣀⣠⣾⣟⣠⣔⡛⢫⣐⠛⢛⣻⣶⠆⠀
⠀⠀⠀⠉⣻⡽⠛⠉⠁⠀⠉⢙⣿⠖⠒⠛⠻⣿⡋⠉⠁⠈⠉⠙⢿⣿⠉⠀⠀⠀
⠀⠀⠀⠸⠿⠷⠒⣦⣤⣴⣶⢿⣿⡀⠀⠀⠀⣽⡿⢷⣦⠤⢤⡖⠶⠿⠧⠀⠀⠀`;

  const lotusLines = LOTUS_TEXT.split('\n');
  for (let i = 0; i < lotusLines.length; i++) {
    blessed.text({
      parent: screen,
      top: i,
      left: 'center',
      content: `{${C.acc}-fg}${lotusLines[i]}{/${C.acc}-fg}`,
      tags: true,
    });
  }

  blessed.text({
    parent: screen,
    top: 12,
    left: 'center',
    content: `{${C.sub}-fg}Decensor + Improve LLMs{/${C.sub}-fg}`,
    tags: true,
  });

  const rule1 = blessed.box({
    parent: screen,
    top: 13,
    left: 0,
    right: 0,
    height: 1,
    style: { fg: C.bdr },
  });

  blessed.text({
    parent: screen,
    top: 14,
    left: 'center',
    content: `{${C.acc}-fg}What do you want to do?{/${C.acc}-fg}`,
    tags: true,
    bold: true,
  });

  const menuItems = [
    { label: '1', desc: 'Ablate', help: 'Remove refusals', action: 'ablate' },
    { label: '2', desc: 'Test', help: 'Benchmark vs base', action: 'test' },
    { label: '3', desc: 'Talk', help: 'Chat with model', action: 'talk' },
    { label: '4', desc: 'List', help: 'Show checkpoints', action: 'list' },
    { label: '0', desc: 'Exit', help: 'Quit', action: 'exit' },
  ];

  const menuText = menuItems.map(item =>
    `  ${item.label}  ${item.desc.padEnd(8)}  {${C.dim}-fg}${item.help}{/${C.dim}-fg}`
  ).join('\n');

  blessed.text({
    parent: screen,
    top: 16,
    left: 'center',
    content: `{${C.fg}-fg}${menuText}{/${C.fg}-fg}`,
    tags: true,
  });

  let inputValue = '';
  const inputBox = blessed.box({
    parent: screen,
    bottom: 0,
    left: 0,
    right: 0,
    height: 1,
    content: `  {${C.acc}-fg}>{/${C.acc}-fg} {${C.fg}-fg}${inputValue}{/${C.fg}-fg}{${C.acc}-fg}▋{/${C.acc}-fg}`,
    tags: true,
  });

  const rule2 = blessed.box({
    parent: screen,
    bottom: 1,
    left: 0,
    right: 0,
    height: 1,
    style: { fg: C.bdr },
  });

  // fill rules
  function fillRules() {
    const w = screen.width;
    rule1.setContent('─'.repeat(w));
    rule2.setContent('─'.repeat(w));
  }
  fillRules();
  screen.on('resize', () => { fillRules(); screen.render(); });

  screen.on('keypress', (ch, key) => {
    if (!key) return;

    if (key.name === 'return' || key.name === 'enter') {
      handleChoice();
      return;
    }

    if (ch && ch >= '0' && ch <= '4') {
      inputValue = ch;
      inputBox.setContent(`  {${C.acc}-fg}>{/${C.acc}-fg} {${C.fg}-fg}${inputValue}{/${C.fg}-fg}{${C.acc}-fg}▋{/${C.acc}-fg}`);
      screen.render();
      handleChoice();
      return;
    }

    if (key.name === 'escape' || (key.ctrl && key.name === 'c')) {
      screen.destroy();
      process.exit(0);
    }
  });

  function handleChoice() {
    const item = menuItems.find(m => m.label === inputValue);
    if (!item) return;
    const action = item.action;

    screen.destroy();

    if (action === 'exit') {
      process.exit(0);
    } else if (action === 'ablate' || action === 'talk' || action === 'test') {
      selectModel(action);
    } else if (action === 'list') {
      runCommandWithProgress(['list'], 'Checkpoints');
    }
  }

  function selectModel(action) {
    const models = modelChoicesForAction(action, 'Custom model ID/path...');

    {
      const consoleSize = getWinConsoleSize();
      const selectScreen = blessed.screen({
        mouse: true,
        keyboard: true,
        dockBorders: false,
        useBCE: true,
        width: consoleSize.cols,
        height: consoleSize.rows,
      });

      blessed.text({
        parent: selectScreen,
        top: 0,
        left: 'center',
        content: `{${C.acc}-fg}Select model for ${action.toUpperCase()}{/${C.acc}-fg}`,
        tags: true,
        bold: true,
      });

      const list = blessed.list({
        parent: selectScreen,
        mouse: true,
        keys: true,
        vi: true,
        tags: true,
        style: {
          selected: { bg: C.acc, fg: 'black' },
          item: { fg: C.fg },
        },
        top: 2,
        left: 'center',
        width: 70,
        height: 10,
        items: models.map(m => `${m.name.padEnd(35)} {${C.dim}-fg}${m.desc}{/${C.dim}-fg}`),
      });

      list.focus();

      list.on('select', (item, index) => {
        selectScreen.destroy();
        const chosen = models[index];
        if (chosen.custom) {
          askText(`Custom model for ${action.toUpperCase()}`, 'HF model ID or local path:', (model) => {
            runModelAction(action, model);
          });
          return;
        }
        runModelAction(action, chosen.name);
      });

      selectScreen.key(['escape', 'q', 'C-c'], () => {
        selectScreen.destroy();
        backToMenu();
      });

      selectScreen.render();
    }
  }

  function runModelAction(action, model) {
    if (action === 'ablate') {
      const out = model.split(/[\\/]/).pop().split('/').pop() + '-apostate';
      runCommandWithProgress(['ablate', '--model', model, '--out', out], 'Ablating refusals...');
    } else if (action === 'talk') {
      selectQuant((q) => {
        if (q === 'vllm') {
          selectKvCache((kv) => runInteractive(['talk', '--model', model, '--backend', 'vllm', '--kv-cache-dtype', kv]));
        } else {
          runInteractive(['talk', '--model', model, '--quant', q]);
        }
      });
    } else if (action === 'test') {
      selectBase((base) => {
        selectBenchmark((suite) => {
          runCommandWithProgress(['test', '--model', model, '--base', base, '--suite', suite], 'Running benchmark...');
        });
      });
    }
  }

  function askText(title, prompt, callback) {
    const consoleSize = getWinConsoleSize();
    const inputScreen = blessed.screen({
      mouse: true,
      keyboard: true,
      dockBorders: false,
      useBCE: true,
      width: consoleSize.cols,
      height: consoleSize.rows,
    });

    blessed.text({
      parent: inputScreen,
      top: 0,
      left: 'center',
      content: `{${C.acc}-fg}${title}{/${C.acc}-fg}`,
      tags: true,
      bold: true,
    });

    blessed.text({
      parent: inputScreen,
      top: 2,
      left: 'center',
      content: `{${C.sub}-fg}${prompt}{/${C.sub}-fg}`,
      tags: true,
    });

    const box = blessed.textbox({
      parent: inputScreen,
      top: 4,
      left: 'center',
      width: 78,
      height: 3,
      border: 'line',
      inputOnFocus: true,
      style: {
        fg: C.fg,
        border: { fg: C.bdr },
        focus: { border: { fg: C.acc } },
      },
    });

    blessed.text({
      parent: inputScreen,
      bottom: 0,
      left: 'center',
      content: `{${C.dim}-fg}Enter submit · ESC cancel{/${C.dim}-fg}`,
      tags: true,
    });

    box.on('submit', (value) => {
      const text = String(value || '').trim();
      inputScreen.destroy();
      if (text) callback(text);
      else backToMenu();
    });
    inputScreen.key(['escape', 'C-c'], () => {
      inputScreen.destroy();
      backToMenu();
    });
    box.focus();
    box.readInput();
    inputScreen.render();
  }

  function selectBase(callback) {
    const models = baseModelChoices('Custom base ID/path...');

    {
      const consoleSize = getWinConsoleSize();
      const selectScreen = blessed.screen({
        mouse: true,
        keyboard: true,
        dockBorders: false,
        useBCE: true,
        width: consoleSize.cols,
        height: consoleSize.rows,
      });

      blessed.text({
        parent: selectScreen,
        top: 0,
        left: 'center',
        content: `{${C.acc}-fg}Select base model to compare against{/${C.acc}-fg}`,
        tags: true,
        bold: true,
      });

      const list = blessed.list({
        parent: selectScreen,
        mouse: true,
        keys: true,
        vi: true,
        tags: true,
        style: {
          selected: { bg: C.acc, fg: 'black' },
          item: { fg: C.fg },
        },
        top: 2,
        left: 'center',
        width: 70,
        height: 10,
        items: models.map(m => `${m.name.padEnd(35)} {${C.dim}-fg}${m.desc}{/${C.dim}-fg}`),
      });

      list.focus();

      list.on('select', (item, index) => {
        selectScreen.destroy();
        const chosen = models[index];
        if (chosen.custom) {
          askText('Custom base model', 'HF model ID or local path:', callback);
          return;
        }
        callback(chosen.name);
      });

      selectScreen.key(['escape', 'q', 'C-c'], () => {
        selectScreen.destroy();
        backToMenu();
      });

      selectScreen.render();
    }
  }

  function selectBenchmark(callback) {
    const suites = [
      { name: 'humaneval', desc: 'code + refusal + GSM8K + KL' },
      { name: 'mbpp', desc: 'MBPP code + refusal + GSM8K + KL' },
      { name: 'gsm8k', desc: 'math capability + refusal + KL' },
      { name: 'refusal', desc: 'JBB refusal/compliance + KL' },
    ];
    const picked = new Set(['humaneval']);
    const rows = () => suites.map(s => {
      const mark = picked.has(s.name) ? '[x]' : '[ ]';
      return `${mark} ${s.name.padEnd(10)} {${C.dim}-fg}${s.desc}{/${C.dim}-fg}`;
    });
    const consoleSize = getWinConsoleSize();
    const selectScreen = blessed.screen({
      mouse: true, keyboard: true, dockBorders: false, useBCE: true,
      width: consoleSize.cols, height: consoleSize.rows,
    });
    blessed.text({
      parent: selectScreen, top: 0, left: 'center', tags: true, bold: true,
      content: `{${C.acc}-fg}Benchmark suites{/${C.acc}-fg}`,
    });
    const list = blessed.list({
      parent: selectScreen, mouse: true, keys: true, vi: true, tags: true,
      style: { selected: { bg: C.acc, fg: 'black' }, item: { fg: C.fg } },
      top: 2, left: 'center', width: 80, height: 8,
      items: rows(),
    });
    blessed.text({
      parent: selectScreen, bottom: 0, left: 'center', tags: true,
      content: `{${C.dim}-fg}Space select  Enter run  ESC cancel{/${C.dim}-fg}`,
    });
    list.focus();
    const toggle = () => {
      const index = list.selected;
      const name = suites[index].name;
      if (picked.has(name)) picked.delete(name);
      else picked.add(name);
      list.setItems(rows());
      list.select(index);
      selectScreen.render();
    };
    list.key(['space'], toggle);
    list.on('select', () => {
      if (!picked.size) {
        toggle();
        return;
      }
      selectScreen.destroy();
      callback([...picked].join(','));
    });
    selectScreen.key(['escape', 'q', 'C-c'], () => {
      selectScreen.destroy();
      backToMenu();
    });
    selectScreen.render();
  }

  // inference quant
  function selectQuant(callback) {
    const quants = [
      { name: 'auto', desc: 'auto weight quant: bf16 if it fits, else nf4' },
      { name: 'vllm', desc: 'serve via vLLM; choose KV cache next' },
      { name: 'nf4', desc: '4-bit, low VRAM' },
      { name: 'marlin', desc: 'int4 Marlin kernel, fastest (Ampere+)' },
      { name: 'bf16', desc: 'no quant, fastest if VRAM fits' },
      { name: 'int8', desc: '8-bit, balanced' },
      { name: 'fp4', desc: '4-bit fp4' },
      { name: 'fp16', desc: 'no quant, fp16' },
      { name: 'gptq', desc: 'int4 gptq (quantizes on load)' },
      { name: 'awq', desc: 'load pre-quantized awq' },
    ];
    {
      const consoleSize = getWinConsoleSize();
      const selectScreen = blessed.screen({
        mouse: true, keyboard: true, dockBorders: false, useBCE: true,
        width: consoleSize.cols, height: consoleSize.rows,
      });
      blessed.text({
        parent: selectScreen, top: 0, left: 'center', tags: true, bold: true,
        content: `{${C.acc}-fg}Inference quant{/${C.acc}-fg}`,
      });
      const list = blessed.list({
        parent: selectScreen, mouse: true, keys: true, vi: true, tags: true,
        style: { selected: { bg: C.acc, fg: 'black' }, item: { fg: C.fg } },
        top: 2, left: 'center', width: 70, height: 12,
        items: quants.map(q => `${q.name.padEnd(10)} {${C.dim}-fg}${q.desc}{/${C.dim}-fg}`),
      });
      list.focus();
      list.on('select', (item, index) => {
        selectScreen.destroy();
        callback(quants[index].name);
      });
      selectScreen.key(['escape', 'q', 'C-c'], () => {
        selectScreen.destroy();
        backToMenu();
      });
      selectScreen.render();
    }
  }

  function selectKvCache(callback) {
    const kvs = [
      { name: 'auto', desc: 'vLLM default' },
      { name: 'fp8', desc: 'KV cache fp8, strong default for memory pressure' },
      { name: 'turboquant_4bit_nc', desc: 'TurboQuant 4-bit KV cache + norm correction' },
      { name: 'turboquant_k8v4', desc: 'TurboQuant fp8 keys + 4-bit values' },
      { name: 'bf16', desc: 'unquantized KV cache' },
      { name: 'turboquant_k3v4_nc', desc: 'more aggressive, may hurt reasoning' },
      { name: 'turboquant_3bit_nc', desc: 'most aggressive, highest risk' },
    ];
    const consoleSize = getWinConsoleSize();
    const selectScreen = blessed.screen({
      mouse: true, keyboard: true, dockBorders: false, useBCE: true,
      width: consoleSize.cols, height: consoleSize.rows,
    });
    blessed.text({
      parent: selectScreen, top: 0, left: 'center', tags: true, bold: true,
      content: `{${C.acc}-fg}vLLM KV-cache dtype{/${C.acc}-fg}`,
    });
    const list = blessed.list({
      parent: selectScreen, mouse: true, keys: true, vi: true, tags: true,
      style: { selected: { bg: C.acc, fg: 'black' }, item: { fg: C.fg } },
      top: 2, left: 'center', width: 82, height: 10,
      items: kvs.map(k => `${k.name.padEnd(22)} {${C.dim}-fg}${k.desc}{/${C.dim}-fg}`),
    });
    list.focus();
    list.on('select', (item, index) => {
      selectScreen.destroy();
      callback(kvs[index].name);
    });
    selectScreen.key(['escape', 'q', 'C-c'], () => {
      selectScreen.destroy();
      backToMenu();
    });
    selectScreen.render();
  }

  // model choices
  function modelChoicesForAction(action, customName) {
    if (action === 'ablate') return baseModelChoices(customName);
    if (action === 'talk' || action === 'test') return apostateModelChoices(customName);
    return baseModelChoices(customName);
  }

  function baseModelChoices(customName) {
    return uniqueModels([
      { name: 'Qwen/Qwen2.5-7B-Instruct', desc: 'default HF' },
      ...findHFModels({ excludeApostate: true }),
      ...findCheckpoints({ apostate: false }),
      { name: customName, desc: 'paste HF ID or local path', custom: true },
    ]);
  }

  function apostateModelChoices(customName) {
    return uniqueModels([
      ...findApostateCheckpoints(),
      ...findCheckpoints({ apostate: true }),
      { name: customName, desc: 'paste HF ID or local path', custom: true },
    ]);
  }

  // dedupe models
  function uniqueModels(items) {
    const out = [];
    const seen = new Set();
    for (const item of items) {
      const key = String(item.name).toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(item);
    }
    return out;
  }

  function isApostateName(name) {
    return String(name || '').toLowerCase().includes('apostate');
  }

  // hf cache roots
  function hfCacheRoots() {
    const roots = [];
    const add = (p) => {
      if (!p) return;
      const full = path.resolve(p);
      if (!roots.includes(full) && fs.existsSync(full)) roots.push(full);
    };
    add(process.env.HUGGINGFACE_HUB_CACHE);
    if (process.env.HF_HOME) add(path.join(process.env.HF_HOME, 'hub'));
    add(path.join(os.homedir(), '.cache', 'huggingface', 'hub'));
    return roots;
  }

  // hf snapshots
  function hasModelSnapshot(dir) {
    if (fs.existsSync(path.join(dir, 'config.json'))) return true;
    const snapRoot = path.join(dir, 'snapshots');
    let snaps = [];
    try { snaps = fs.readdirSync(snapRoot, { withFileTypes: true }); } catch (e) { return false; }
    return snaps.some(s => {
      if (!s.isDirectory()) return false;
      const p = path.join(snapRoot, s.name);
      return fs.existsSync(path.join(p, 'config.json')) ||
             fs.existsSync(path.join(p, 'tokenizer_config.json')) ||
             fs.existsSync(path.join(p, 'processor_config.json'));
    });
  }

  // scan hf cache
  function findHFModels(opts = {}) {
    const out = [];
    const seen = new Set();
    for (const root of hfCacheRoots()) {
      let entries = [];
      try { entries = fs.readdirSync(root, { withFileTypes: true }); } catch (e) { continue; }
      for (const d of entries) {
        if (!d.isDirectory() || !d.name.startsWith('models--')) continue;
        const id = d.name.slice('models--'.length).split('--').join('/');
        if (!id || seen.has(id.toLowerCase())) continue;
        if (opts.excludeApostate && isApostateName(id)) continue;
        const dir = path.join(root, d.name);
        if (!hasModelSnapshot(dir)) continue;
        seen.add(id.toLowerCase());
        out.push({ name: id, desc: 'HF cache' });
      }
    }
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }

  // scan checkpoints
  function findCheckpoints(opts = {}) {
    const out = [];
    const seen = new Set();
    for (const base of [process.cwd(), __dirname]) {
      let entries = [];
      try { entries = fs.readdirSync(base, { withFileTypes: true }); } catch (e) { continue; }
      for (const d of entries) {
        if (!d.isDirectory() || seen.has(d.name)) continue;
        const dir = path.join(base, d.name);
        let files = [];
        try { files = fs.readdirSync(dir); } catch (e) { continue; }
        if (files.includes('config.json') && files.some(x => x.endsWith('.safetensors'))) {
          const apostate = isApostateCheckpoint(dir, files);
          if (opts.apostate === true && !apostate) continue;
          if (opts.apostate === false && apostate) continue;
          seen.add(d.name);
          out.push({ name: dir, desc: apostate ? 'apostate checkpoint' : 'checkpoint' });
        }
      }
    }
    return out;
  }

  function isApostateCheckpoint(dir, files = null) {
    const nameHit = isApostateName(path.basename(dir));
    if (!files) {
      try { files = fs.readdirSync(dir); } catch (e) { files = []; }
    }
    if (files.includes('apostate_config.json')) return true;
    if (files.includes('report.json')) {
      try {
        const txt = fs.readFileSync(path.join(dir, 'report.json'), 'utf8').toLowerCase();
        if (txt.includes('"optimized"') || txt.includes('"best_params"') || txt.includes('apostate')) return true;
      } catch (e) {}
    }
    if (files.includes('README.md')) {
      try {
        if (fs.readFileSync(path.join(dir, 'README.md'), 'utf8').toLowerCase().includes('apostate')) return true;
      } catch (e) {}
    }
    return nameHit;
  }

  function isModelDir(dir, files = null) {
    if (!files) {
      try { files = fs.readdirSync(dir); } catch (e) { return false; }
    }
    return files.includes('config.json') && files.some(x => x.endsWith('.safetensors') || x.endsWith('.bin'));
  }

  function isUnderAny(dir, roots) {
    const clean = (p) => path.resolve(p).replace(/[\\/]+$/, '').toLowerCase();
    const full = clean(dir);
    return roots.some(root => {
      const r = clean(root);
      return full === r || full.startsWith(r + path.sep.toLowerCase());
    });
  }

  function scanRoots() {
    const roots = [];
    const add = (p, depth) => {
      if (!p) return;
      const full = path.resolve(p);
      if (!fs.existsSync(full)) return;
      if (roots.some(r => r.root === full)) return;
      roots.push({ root: full, depth });
    };
    const home = os.homedir();
    const envRoots = String(process.env.APOSTATE_MODEL_ROOTS || '').split(path.delimiter).filter(Boolean);
    for (const r of envRoots) add(r, 6);
    add(path.join(home, 'OneDrive', 'Desktop', 'apostatehfmodels'), 3);
    add(path.join(home, 'Desktop', 'apostatehfmodels'), 3);
    add(path.join(home, 'OneDrive', 'Desktop'), 3);
    add(path.join(home, 'Desktop'), 3);
    add(path.join(home, 'OneDrive', 'Documents'), 3);
    add(path.join(home, 'Documents'), 3);
    add(process.cwd(), 3);
    add(__dirname, 3);
    if (process.platform === 'win32') {
      for (let code = 68; code <= 90; code++) {
        add(String.fromCharCode(code) + ':\\', 4);
      }
    }
    return roots;
  }

  function shouldSkipDir(dir) {
    const n = path.basename(dir).toLowerCase();
    return [
      '.git', '.hg', '.svn', 'node_modules', '__pycache__', '.venv', 'venv',
      '.cache', 'cache', 'windows', 'program files', 'program files (x86)',
      'programdata', 'appdata', '$recycle.bin', 'system volume information',
    ].includes(n);
  }

  function findApostateCheckpoints() {
    const out = [];
    const seen = new Set();
    const hfRoots = hfCacheRoots();
    const start = Date.now();
    const budgetMs = 2500;
    const maxFound = 80;
    for (const item of scanRoots()) {
      const stack = [{ dir: item.root, depth: 0 }];
      while (stack.length && Date.now() - start < budgetMs && out.length < maxFound) {
        const { dir, depth } = stack.pop();
        if (!dir || seen.has(dir.toLowerCase())) continue;
        seen.add(dir.toLowerCase());
        if (isUnderAny(dir, hfRoots) || shouldSkipDir(dir)) continue;
        let entries = [];
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { continue; }
        const files = entries.filter(e => e.isFile()).map(e => e.name);
        if (isModelDir(dir, files) && isApostateCheckpoint(dir, files)) {
          out.push({ name: dir, desc: 'local apostate' });
          continue;
        }
        if (depth >= item.depth) continue;
        for (let i = entries.length - 1; i >= 0; i--) {
          const e = entries[i];
          if (!e.isDirectory()) continue;
          const child = path.join(dir, e.name);
          if (!shouldSkipDir(child)) stack.push({ dir: child, depth: depth + 1 });
        }
      }
    }
    return out.sort((a, b) => a.name.localeCompare(b.name));
  }

  // relaunch menu
  function backToMenu() {
    const proc = spawn(process.execPath, [__filename], { stdio: 'inherit' });
    proc.on('close', (c) => process.exit(c || 0));
  }

  // kill tree
  function killTree(proc) {
    if (!proc || proc.killed) return;
    if (process.platform === 'win32') {
      try { execSync(`taskkill /pid ${proc.pid} /T /F`, { stdio: 'ignore' }); } catch (e) {}
    } else {
      try { proc.kill('SIGTERM'); } catch (e) {}
    }
  }

  // live log
  function runCommandWithProgress(args, message) {
    const mainPath = path.join(__dirname, 'main.js');
    const sz = getWinConsoleSize();

    const logScreen = blessed.screen({
      mouse: true, keyboard: true, useBCE: true,
      width: sz.cols, height: sz.rows,
    });

    blessed.text({
      parent: logScreen, top: 0, left: 'center', tags: true, bold: true,
      content: `{${C.acc}-fg}${message}{/${C.acc}-fg}`,
    });

    const log = blessed.log({
      parent: logScreen, top: 2, left: 0, right: 0, bottom: 1,
      border: 'line', style: { border: { fg: C.bdr }, fg: C.fg },
      scrollable: true, alwaysScroll: true, scrollOnInput: true,
      scrollbar: { ch: ' ', style: { bg: C.acc } },
      tags: true, mouse: true, keys: true,
    });

    const footer = blessed.text({
      parent: logScreen, bottom: 0, left: 'center', tags: true,
      content: `{${C.dim}-fg}ESC cancel · running…{/${C.dim}-fg}`,
    });

    logScreen.render();

    let finished = false, cancelling = false;

    const proc = spawn(process.execPath, [mainPath, ...args], {
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    const append = (buf) => {
      const text = buf.toString().replace(/\r/g, '');
      for (const line of text.split('\n')) {
        if (line.length) log.add(blessed.escape(line));
      }
      logScreen.render();
    };
    proc.stdout.on('data', append);
    proc.stderr.on('data', append);
    proc.on('error', (e) => { log.add(`{${C.red}-fg}spawn error: ${blessed.escape(String(e.message || e))}{/${C.red}-fg}`); logScreen.render(); });

    logScreen.key(['escape', 'C-c'], () => {
      if (finished) { logScreen.destroy(); backToMenu(); return; }
      cancelling = true;
      footer.setContent(`{${C.ylw}-fg}cancelling…{/${C.ylw}-fg}`);
      logScreen.render();
      killTree(proc);
    });

    logScreen.key(['enter', 'return'], () => {
      if (finished) { logScreen.destroy(); backToMenu(); }
    });

    proc.on('close', (code) => {
      finished = true;
      if (cancelling) { logScreen.destroy(); backToMenu(); return; }
      log.add('');
      log.add(code === 0
        ? `{${C.grn}-fg}✓ Complete{/${C.grn}-fg}`
        : `{${C.red}-fg}✗ Exited (code ${code}){/${C.red}-fg}`);
      footer.setContent(`{${C.dim}-fg}ESC / Enter → main menu{/${C.dim}-fg}`);
      logScreen.render();
    });
  }

  // interactive chat
  function runInteractive(args) {
    const mainPath = path.join(__dirname, 'main.js');
    const proc = spawn(process.execPath, [mainPath, ...args], { stdio: 'inherit' });
    proc.on('close', () => backToMenu());
  }

    screen.render();
}
