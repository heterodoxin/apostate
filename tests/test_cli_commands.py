"""The command surface: what `apostate <command>` dispatches to, and what `--help` promises.

These tests exercise `apostate.__main__`'s dispatcher, `apostate.cli`'s engine selection and the TUI's
invocation of it -- not any single method's math. Five of them used to sit in `tests/test_kcrn.py`,
between the KCRN formula tests, where a change to the command surface looked like a KCRN failure. The
one that was added there as a regression guard has been moved here too, because that is what it guards.

**The invariant worth pinning is that the help text and the dispatcher are two descriptions of one
surface, and they must agree.** A documented command whose dispatch branch disappears still prints in
`--help`; a dispatch branch whose help line disappears is invisible to whoever reads `--help`. Both are
caught here, from the one list below, which is the contract: `DOCUMENTED_COMMANDS`.

These tests are offline and GPU-free by construction -- they stub every collaborator that would load a
model, resolve a Hub id or touch a device -- so a red result here means the command surface changed, not
that a machine was unavailable.
"""

from __future__ import annotations

import pytest

#: The commands `apostate --help` documents. Adding a command means adding it here, which is the point:
#: the help text and the dispatcher cannot drift apart without one of these tests going red.
DOCUMENTED_COMMANDS = (
    "setup",
    "doctor",
    "ablate",
    "diode",
    "ccv",
    "kcrn",
    "ticv",
    "finetune",
    "test",
    "talk",
    "prepare-quant",
    "convert-tree",
    "quantize-gguf",
    "quantize",
    "list",
)


@pytest.fixture
def stubbed_dispatch(monkeypatch):
    """Record what the dispatcher launched, and stop anything from leaving the process."""
    import apostate.__main__ as main_module

    launched = []
    monkeypatch.setattr(
        main_module, "run_module", lambda args, label=None: launched.append((list(args), label)) or 0
    )
    monkeypatch.setattr(main_module, "_run_interactive", lambda: 0, raising=False)

    def record_inline(name):
        def stub(*args, **kwargs):
            del args, kwargs
            launched.append(([name], name))
            return 0

        return stub

    # `setup` and the bare-`apostate` menu are dispatched inline, so run_module never sees them.
    monkeypatch.setattr("apostate.setup_wizard.main", record_inline("setup"))
    monkeypatch.setattr("apostate.tui.run", record_inline("tui"))
    return launched


def test_help_documents_every_contract_command(capsys):
    """The guard this file inherited: a documented command must keep being discoverable."""
    import apostate.__main__ as main_module

    assert main_module.main(["--help"]) == 0

    help_text = capsys.readouterr().out
    for command in DOCUMENTED_COMMANDS:
        assert f"apostate {command}" in help_text, (
            f"{command!r} is in DOCUMENTED_COMMANDS but missing from --help; either restore its line in "
            "HELP or take it out of the contract deliberately"
        )


def test_every_contract_command_dispatches(stubbed_dispatch):
    """A documented command must reach a module, not fall through to `return 1`."""
    import apostate.__main__ as main_module

    for command in DOCUMENTED_COMMANDS:
        assert main_module.main([command]) == 0, f"{command} is documented but not dispatched"


def test_a_bare_invocation_still_reaches_the_menu(stubbed_dispatch):
    """`apostate` with no arguments is the interactive menu, not an error."""
    import apostate.__main__ as main_module

    assert main_module.main([]) == 0
    assert stubbed_dispatch == [(["tui"], "tui")]


@pytest.mark.parametrize(
    ("command", "method"),
    [("ablate", "diode"), ("diode", "diode"), ("boost", "legacy"), ("kcrn", "kcrn"), ("ccv", "ccv")],
)
def test_subcommands_select_their_engine(command, method, stubbed_dispatch):
    """Each build subcommand composes the engine it names. `ablate` is the diode: the default method."""
    import apostate.__main__ as main_module

    assert main_module.main([command, "--model", "base", "--out", "out"]) == 0

    argv, _label = stubbed_dispatch[0]
    assert argv[argv.index("--method") + 1] == method
    assert "--model" in argv and argv[argv.index("--model") + 1] == "base"


def test_subcommands_forward_the_flags_they_do_not_consume(stubbed_dispatch):
    """`--model`/`--out` are lifted into the label; everything else must survive into the engine argv."""
    import apostate.__main__ as main_module

    assert main_module.main(["ablate", "--model", "base", "--out", "out", "--resume"]) == 0

    argv, label = stubbed_dispatch[0]
    assert "--resume" in argv
    assert label is not None and "ablate" in label


def test_cli_defaults_to_the_diode_engine(monkeypatch):
    """`apostate.cli` with no `--method` bakes with the diode, which is the method the README documents.

    This test used to assert the KCRN engine and failed once the default moved: with `run_kcrn` stubbed
    it fell through to the diode path, which resolves its model over the network. Stubbing the diode
    entry point keeps the test offline and asserts the contract that actually holds.
    """
    import apostate.cli as cli

    seen = []
    monkeypatch.setattr("apostate.diode.fit_and_bake", lambda cfg: seen.append(cfg.method) or {"edited_layers": 0})
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: seen.append("kcrn"))

    cli.main(["--model", "base", "--output-dir", "out"])

    assert seen == ["diode"]


def test_cli_dispatches_ccv_to_legacy_engine_with_predictive_oblique(monkeypatch):
    import apostate.cli as cli

    calls = []
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: calls.append(("kcrn", cfg)))
    monkeypatch.setattr(cli, "run_legacy", lambda cfg, command=None: calls.append(("legacy", cfg)))

    cli.main(["--method", "ccv", "--model", "base", "--output-dir", "out"])

    assert len(calls) == 1
    route, cfg = calls[0]
    assert route == "legacy"
    assert cfg.method == "ccv"
    assert cfg.oblique_ablation is True
    assert cfg.oblique_predictive is True


def test_cli_can_override_old_config_method(monkeypatch, tmp_path):
    import apostate.cli as cli

    config_path = tmp_path / "old-config.json"
    config_path.write_text('{"model": "base"}\n', encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "run_kcrn", lambda cfg, command=None: calls.append(cfg.method))

    cli.main(["--config", str(config_path), "--method", "kcrn"])

    assert calls == ["kcrn"]


def test_tui_ablation_uses_the_default_engine(monkeypatch):
    """The TUI's ablate action runs `apostate ablate`, which is the diode, and names its output
    `-abliterated`. It used to call `kcrn` directly; the output naming is the part that must not move."""
    import apostate.tui as tui

    calls = []
    app = tui.Apostate.__new__(tui.Apostate)
    monkeypatch.setattr(app, "run_cli", lambda args: calls.append(args))

    app._do_ablate("/models/Qwen3-8B")

    assert calls == [["ablate", "--model", "/models/Qwen3-8B", "--out", "Qwen3-8B-abliterated"]]
