#!/bin/sh
# Bootstraps .venv on the first run, reinstalls only when requirements.txt
# changed (a copy is kept in .venv), then runs run.py. No cd: paths stay
# relative to the user's folder.
set -e

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv"
  python3 -m venv .venv
fi

REQ=.venv/.requirements.txt
if [ ! -f "$REQ" ] || ! cmp -s "$REQ" requirements.txt; then
  echo "installing requirements"
  .venv/bin/pip install --quiet -r requirements.txt
  cp requirements.txt "$REQ"
fi

exec .venv/bin/python run.py "$@"
