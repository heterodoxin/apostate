# textual tui: pick an action and a model, then run the matching cli command.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from textual.app import App, ComposeResult
from textual.containers import Center, Middle, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Rule, Static

from . import discover

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

LOTUS = (
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠟⠹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⠀⠀⠀⣷⣦⣄⣠⣿⠃⢠⣄⠈⢻⣆⣠⣴⡞⡆⠀⠀⠀⠀⠀⠀⠀\n"
    "⠀⠀⠀⠀⠀⢀⣀⣀⣿⠀⠈⢻⣇⢀⣾⢟⡄⣸⡿⠋⠀⡇⣇⣀⣀⠀⠀⠀⠀⠀\n"
    "⠀⣤⣤⣤⣀⣱⢻⠚⠻⣧⣀⠀⢹⡿⠃⠈⢻⣟⠀⢀⣤⠧⠓⣹⣟⣀⣤⣤⣤⡀\n"
    "⠀⠈⠻⣧⠉⠛⣽⠀⠀⠀⠙⣷⡿⠁⠀⠀⠀⢻⣶⠛⠁⠀⠀⡟⠟⠉⣵⡟⠁⠀\n"
    "⠀⠀⠀⠹⣧⡀⠏⡇⠀⠀⠀⣿⠁⠀⠀⠀⠀⠀⣿⡄⠀⠀⢠⢷⠀⣼⡟⠀⠀⠀\n"
    "⠀⠀⠀⠀⠙⣟⢼⡹⡄⠀⠀⣿⡄⠀⠀⠀⠀⢀⣿⡇⠀⢀⣞⣦⢾⠟⠀⠀⠀⠀\n"
    "⠀⠠⢶⣿⣛⠛⢒⣭⢻⣶⣤⣹⣿⣤⣀⣀⣠⣾⣟⣠⣔⡛⢫⣐⠛⢛⣻⣶⠆⠀\n"
    "⠀⠀⠀⠉⣻⡽⠛⠉⠁⠀⠉⢙⣿⠖⠒⠛⠻⣿⡋⠉⠁⠈⠉⠙⢿⣿⠉⠀⠀⠀\n"
    "⠀⠀⠀⠸⠿⠷⠒⣦⣤⣴⣶⢿⣿⡀⠀⠀⠀⣽⡿⢷⣦⠤⢤⡖⠶⠿⠧⠀⠀⠀"
)

ACTIONS = [
    ("ablate", "Ablate   remove refusals"),
    ("test", "Test     benchmark vs base"),
    ("talk", "Talk     chat with model"),
    ("list", "List     show checkpoints"),
    ("exit", "Exit     quit"),
]
QUANTS = ["auto", "vllm", "nf4", "bf16", "fp16", "int8", "fp4", "gptq", "marlin", "awq"]
SUITES = [
    ("humaneval", "code pass@1 + refusal + GSM8K + KL"),
    ("mbpp", "MBPP code + refusal + GSM8K + KL"),
    ("gsm8k", "math capability + refusal + KL"),
    ("refusal", "JBB refusal/compliance + KL"),
]
KV_CACHE = [
    ("auto", "no KV-cache quant"),
    ("turboquant_4bit_nc", "TurboQuant 4-bit KV + norm correction"),
    ("turboquant_k8v4", "TurboQuant fp8 keys + 4-bit values"),
    ("turboquant_k3v4_nc", "more aggressive, may hurt reasoning"),
    ("turboquant_3bit_nc", "most aggressive, highest risk"),
]
CUSTOM = "… custom id / path"

CSS = """
Screen { align: center middle; background: black; }
#lotus { color: #cba6f7; content-align: center middle; background: black; }
#title { color: #bac2de; content-align: center middle; background: black; }
Rule.-horizontal { color: #6c7086; width: 60; height: 1; margin: 0; background: black; }
ListView { width: 60; height: auto; max-height: 16; background: black; }
ListView > ListItem { background: black; color: #cdd6f4; }
ListView > ListItem.-highlight { background: #e6dcff; color: #1e1e2e; }
ListView:focus > ListItem.-highlight { background: #e6dcff; color: #1e1e2e; }
#prompt { color: #cba6f7; background: black; }
Input { width: 60; background: black; border: tall #313244; }
.hint { color: #6c7086; background: black; }
"""


class Pick(ModalScreen[Optional[str]]):
    BINDINGS = [("escape", "dismiss", "back")]

    def __init__(self, prompt: str, options: List[str], allow_custom: bool = False):
        super().__init__()
        self.prompt = prompt
        self.options = options + ([CUSTOM] if allow_custom else [])

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(o)) for o in self.options))
            with Center():
                yield Rule()
            with Center():
                yield Label("enter select   esc back", classes="hint")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()  # don't let the modal's selection bubble up to the menu handler
        choice = self.options[event.list_view.index]
        if choice == CUSTOM:
            self.app.push_screen(AskText("model id or path"), self._custom)
        else:
            self.dismiss(choice)

    def _custom(self, value: Optional[str]) -> None:
        if value:
            self.dismiss(value)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class AskText(ModalScreen[Optional[str]]):
    BINDINGS = [("escape", "dismiss", "back")]

    def __init__(self, prompt: str):
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Input(placeholder="Qwen/Qwen2.5-7B-Instruct")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_dismiss(self) -> None:
        self.dismiss(None)


class MultiPick(ModalScreen[Optional[str]]):
    BINDINGS = [("escape", "dismiss", "back"), ("space", "toggle", "toggle")]

    def __init__(self, prompt: str, options: List[tuple], default: set):
        super().__init__()
        self.prompt = prompt
        self.opts = options          # (name, desc)
        self.picked = set(default)

    def _rows(self) -> List[str]:
        return [f"[{'x' if n in self.picked else ' '}] {n:<10} {d}" for n, d in self.opts]

    def compose(self) -> ComposeResult:
        with Middle():
            with Center():
                yield Label(self.prompt, id="prompt")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(r)) for r in self._rows()), id="multi")
            with Center():
                yield Rule()
            with Center():
                yield Label("space select   enter run   esc back", classes="hint")

    def _refresh(self) -> None:
        lv = self.query_one("#multi", ListView)
        idx = lv.index
        lv.clear()
        for r in self._rows():
            lv.append(ListItem(Label(r)))
        lv.index = idx

    def action_toggle(self) -> None:
        lv = self.query_one("#multi", ListView)
        name = self.opts[lv.index][0]
        self.picked.discard(name) if name in self.picked else self.picked.add(name)
        self._refresh()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if not self.picked:                 # enter with nothing picked = toggle current
            self.action_toggle()
            return
        self.dismiss(",".join(n for n, _ in self.opts if n in self.picked))

    def action_dismiss(self) -> None:
        self.dismiss(None)


class Apostate(App):
    CSS = CSS
    TITLE = "apostate"
    BINDINGS = [("escape", "quit", "quit"), ("q", "quit", "quit")]

    def compose(self) -> ComposeResult:
        with Vertical():
            with Center():
                yield Static(LOTUS, id="lotus")
            with Center():
                yield Static("decensor + improve llms", id="title")
            with Center():
                yield Rule()
            with Center():
                yield ListView(*(ListItem(Label(text)) for _, text in ACTIONS), id="menu")

    def on_mount(self) -> None:
        # bases = unablated hf models; apostate = baked checkpoints found anywhere on disk.
        # full drive scan runs in the background; talk/test await it before opening.
        self.base_models: List[str] = [DEFAULT_MODEL, *discover.hf_models()]
        self.apostate_models: List[str] = discover.apostate_checkpoints()
        self._scan = self.run_worker(self._scan_drive, thread=True, exclusive=True)

    def _scan_drive(self) -> None:
        seen = {str(m).lower() for m in self.apostate_models}
        for m in discover.scan_apostate():
            if str(m).lower() not in seen:
                self.apostate_models.append(m)
                seen.add(str(m).lower())

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "menu":  # ignore selections from modal pickers
            return
        action = ACTIONS[event.list_view.index][0]
        if action == "exit":
            self.exit()
        elif action == "list":
            self.run_cli(["list"])
        elif action == "ablate":
            # only bases to ablate, never an already-ablated model
            self.push_screen(Pick("base model to ablate", self.base_models, allow_custom=True),
                             self._do_ablate)
        elif action == "talk":
            await self._await_scan()
            self.push_screen(Pick("apostate model", self.apostate_models, allow_custom=True),
                             self._pick_quant)
        elif action == "test":
            await self._await_scan()
            self.push_screen(Pick("apostate model to benchmark", self.apostate_models, allow_custom=True),
                             self._pick_base)

    async def _await_scan(self) -> None:
        # block briefly on first talk/test so the drive scan has the full list ready
        if getattr(self, "_scan", None) is not None and self._scan.is_running:
            try:
                await self._scan.wait()
            except Exception:
                pass

    def _do_ablate(self, model: Optional[str]) -> None:
        if model:
            out = model.split("/")[-1].split("\\")[-1] + "-apostate"
            self.run_cli(["ablate", "--model", model, "--out", out])

    def _pick_base(self, model: Optional[str]) -> None:
        if model:
            self.push_screen(Pick("base model to compare", self.base_models, allow_custom=True),
                             lambda base: base and self._pick_suite(model, base))

    def _pick_suite(self, model: str, base: str) -> None:
        self.push_screen(MultiPick("benchmark suites", SUITES, {"humaneval"}),
                         lambda s: s and self.run_cli(["test", "--model", model, "--base", base, "--suite", s]))

    def _pick_quant(self, model: Optional[str]) -> None:
        if model:
            self.push_screen(Pick("inference quant", QUANTS),
                             lambda q: q and self._talk(model, q))

    def _talk(self, model: str, quant: str) -> None:
        if quant == "vllm":
            # vllm can quantize the kv cache; let the user pick a dtype
            self.push_screen(Pick("KV-cache dtype", [n for n, _ in KV_CACHE]),
                             lambda kv: self.run_cli(["talk", "--model", model, "--backend", "vllm"]
                                                     + (["--kv-cache-dtype", kv] if kv and kv != "auto" else [])))
        else:
            self.run_cli(["talk", "--model", model, "--quant", quant])

    def run_cli(self, args: List[str]) -> None:
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
        with self.suspend():
            subprocess.run([sys.executable, "-m", "apostate", *args], env=env)
            input("\n[enter] back to menu ")


def run() -> int:
    Apostate().run()
    return 0


if __name__ == "__main__":
    run()
