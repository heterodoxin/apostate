#!/usr/bin/env bash
# setup wizard
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
exec python3 -m apostate setup "$@"
