import argparse
from pathlib import Path
from orchestrator.codex_runner import create_candidate_worktree
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hypothesis_id")
    parser.add_argument("--base-ref", default="HEAD")
    args = parser.parse_args()
    print(create_candidate_worktree(Path(__file__).resolve().parents[1], args.hypothesis_id, args.base_ref))
