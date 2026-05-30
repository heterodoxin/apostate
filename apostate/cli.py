"""Command-line interface for Apostate."""

from __future__ import annotations

import argparse
import dataclasses

from .config import ApostateConfig
from .engine import run


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
    parser = argparse.ArgumentParser(
        prog="apostate",
        description="Memory-efficient abliteration: subspace + preservation + causal + guard.",
    )
    parser.add_argument("--config", type=str, default=None, help="Load an ApostateConfig JSON; CLI flags override it.")
    _add_config_args(parser)
    args = parser.parse_args(argv)

    if args.config:
        cfg = ApostateConfig.from_json(args.config)
        # apply any explicitly-passed overrides
        for f in dataclasses.fields(ApostateConfig):
            val = getattr(args, f.name, None)
            if val is not None and val != f.default:
                setattr(cfg, f.name, val)
    else:
        kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(ApostateConfig)}
        cfg = ApostateConfig(**kwargs)

    run(cfg)


if __name__ == "__main__":
    main()
