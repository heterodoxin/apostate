from __future__ import annotations

import argparse
import dataclasses
import os
import shlex
import sys

from .config import ApostateConfig
from .moe_compat import apply_model_bundle_patches

apply_model_bundle_patches()

from .engine import run as run_legacy
from .kcrn_runner import run as run_kcrn


def _add_config_args(parser: argparse.ArgumentParser):
    for f in dataclasses.fields(ApostateConfig):
        name = "--" + f.name.replace("_", "-")
        default = f.default
        if f.type == bool or isinstance(default, bool):
            parser.add_argument(name, dest=f.name, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int) and not isinstance(default, bool):
            parser.add_argument(name, dest=f.name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(name, dest=f.name, type=float, default=default)
        else:
            parser.add_argument(name, dest=f.name, type=str, default=default)


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="apostate",
        description="Build a fixed-weight KCRN checkpoint.",
    )
    parser.add_argument("--config", type=str, default=None, help="Load an ApostateConfig JSON; CLI flags override it.")
    _add_config_args(parser)
    args = parser.parse_args(raw_args)

    if args.config:
        cfg = ApostateConfig.from_json(args.config)
        for f in dataclasses.fields(ApostateConfig):
            val = getattr(args, f.name, None)
            flag = "--" + f.name.replace("_", "-")
            explicitly_set = any(item == flag or item.startswith(flag + "=") for item in raw_args)
            if val is not None and (explicitly_set or val != f.default):
                setattr(cfg, f.name, val)
    else:
        kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(ApostateConfig)}
        cfg = ApostateConfig(**kwargs)

    command = os.environ.get("APOSTATE_COMMAND") or " ".join(shlex.quote(x) for x in sys.argv)
    method = (cfg.method or "kcrn").strip().lower()
    if method == "kcrn":
        run_kcrn(cfg, command=command)
    elif method in ("legacy", "engine"):
        run_legacy(cfg, command=command)
    else:
        raise ValueError(f"unknown Apostate method {cfg.method!r}; use kcrn or legacy")


if __name__ == "__main__":
    main()
