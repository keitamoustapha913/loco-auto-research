#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -d .git ]]; then
  echo "Git already initialized. No history changed."
  exit 0
fi
git init
git add .
echo "Review 'git status --short' for secrets/dataset leaks before first commit."
echo "Then run: git -c user.name='Research Operator' -c user.email='operator@localhost' commit -m 'Initialize guarded LOCO research laboratory'"
