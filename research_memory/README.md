# Persistent research memory

`hypotheses.md`, `discoveries.md`, and `failed_approaches.md` are curated human-readable research notes; do not treat prose as trusted numerical evidence. Operational hypotheses, candidate run states, and events live in **SQLite**, created automatically at `${LOCO_LEDGER}` (default `${LOCO_RUNTIME}/controller/experiments.sqlite`).

The diagram originally placed `experiments.sqlite` here, but shipping an initialized database in a source repository is unsafe. No source-control commit should include the live ledger; operators can export redacted summaries into tracked Markdown and back up the SQLite DB outside Git. The dashboard uses the operational SQLite ledger read-only.
