#!/usr/bin/env bash
# Brings up MediaMTX + backend + frontend together. See run_all.py for flags,
# e.g. `./start.sh --skip-mediamtx` if you already have MediaMTX running.
cd "$(dirname "$0")"
python3 run_all.py "$@"
