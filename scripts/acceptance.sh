#!/usr/bin/env bash
# Released-artifact acceptance gate, shared by ci.yml and release.yml so the
# two can never drift. Installs the BUILT WHEEL into a clean venv and runs the
# documented commands from a NEUTRAL working directory, so the source tree can
# never shadow the installed package (cwd/script-dir on sys.path).
#
# Usage: scripts/acceptance.sh [venv-dir]   (run after `python -m build`)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$(mktemp -d)/venv}"

python -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$ROOT"/dist/*.whl

cd "$(mktemp -d)"

"$VENV/bin/mothrag" --version
"$VENV/bin/mothrag" --help
"$VENV/bin/mothrag" query "Where is the Eiffel Tower?" \
  --text "The Eiffel Tower is a wrought-iron lattice tower in Paris, France." \
  --embedder hash
# Guards run against the INSTALLED wheel (they take absolute paths, cwd stays neutral).
"$VENV/bin/python" "$ROOT/tests/test_entry_points.py"
"$VENV/bin/python" "$ROOT/tests/test_version_sync.py"

echo "acceptance: OK"
