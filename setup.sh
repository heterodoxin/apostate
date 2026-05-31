#!/usr/bin/env bash
# setup wizard
cd "$(dirname "$0")"
exec node setup.js "$@"
