"""textual tui: pick an action and a model, then run the matching cli command."""

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
    """list picker; returns the chosen value (or None on escape)."""

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
    """single text input; returns the string (or None on escape)."""

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
                yield ListView(*(ListItem(Label(text)) for _, text in ACTIONS))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        action = ACTIONS[event.list_view.index][0]
        if action == "exit":
            self.exit()
        elif action == "list":
            self.run_cli(["list"])
        elif action in ("ablate", "test", "talk"):
            self.push_screen(Pick(f"model for {action}", self._models(), allow_custom=True),
                             lambda m: self._after_model(action, m))

    def _models(self) -> List[str]:
        return [DEFAULT_MODEL, *discover.checkpoints(), *discover.hf_models()]

    def _after_model(self, action: str, model: Optional[str]) -> None:
        if not model:
            return
        if action == "ablate":
            out = model.split("/")[-1].split("\\")[-1] + "-apostate"
            self.run_cli(["ablate", "--model", model, "--out", out])
        elif action == "test":
            self.push_screen(Pick("base model to compare", self._models(), allow_custom=True),
                             lambda base: base and self.run_cli(["test", "--model", model, "--base", base]))
        elif action == "talk":
            self.push_screen(Pick("inference quant", QUANTS),
                             lambda q: q and self._talk(model, q))

    def _talk(self, model: str, quant: str) -> None:
        if quant == "vllm":
            self.run_cli(["talk", "--model", model, "--backend", "vllm"])
        else:
            self.run_cli(["talk", "--model", model, "--quant", quant])

    def run_cli(self, args: List[str]) -> None:
        """drop to the terminal, run `apostate <args>`, then return to the menu."""
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1")
        with self.suspend():
            subprocess.run([sys.executable, "-m", "apostate", *args], env=env)
            input("\n[enter] back to menu ")


def run() -> int:
    Apostate().run()
    return 0


if __name__ == "__main__":
    run()
