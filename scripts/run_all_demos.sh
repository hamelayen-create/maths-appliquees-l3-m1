#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for f in 0*/src/demo.py; do
  echo "======== $f ========"
  python3 "$f"
  echo
done
