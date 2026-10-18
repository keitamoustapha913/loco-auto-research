from __future__ import annotations
import argparse
import json
from pathlib import Path
from benchmark.data_guard import build_manifest, manifest_fingerprint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest = build_manifest(args.root)
    if not manifest["images"]:
        raise SystemExit("No good images found in category/{train,validation}/good folders")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"images": len(manifest["images"]), "fingerprint": manifest_fingerprint(manifest), "manifest": str(output)}))

if __name__ == "__main__":
    main()
