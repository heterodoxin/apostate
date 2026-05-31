#!/usr/bin/env bash
# setup wizard (linux/mac)
cd "$(dirname "$0")"
exec node setup.js "$@"
