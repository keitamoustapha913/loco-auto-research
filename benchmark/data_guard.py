"""Trusted manifest validation. Real secrecy also requires independent OS permissions."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

CATEGORIES = ("breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors")
ALLOWED_SPLITS = ("train/good", "validation/good")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class DataAccessError(ValueError):
    pass


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_root(root: str | Path, sealed_root: str | Path | None = None) -> Path:
    root = Path(root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise DataAccessError("good dataset root must be a directory")
    forbidden = Path(sealed_root or os.getenv("LOCO_SEALED_ROOT", "/srv/loco-sealed")).expanduser().resolve()
    if is_relative_to(root, forbidden) or is_relative_to(forbidden, root):
        raise DataAccessError("good root overlaps sealed root")
    return root


def resolve_good_image(root: str | Path, relative_path: str, sealed_root: str | Path | None = None) -> Path:
    base = validate_root(root, sealed_root)
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 4:
        raise DataAccessError("image path must be category/train|validation/good/filename")
    if rel.parts[0] not in CATEGORIES or "/".join(rel.parts[1:3]) not in ALLOWED_SPLITS:
        raise DataAccessError(f"forbidden category or split: {relative_path}")
    if rel.suffix.lower() not in EXTENSIONS:
        raise DataAccessError("unsupported image extension")
    candidate = (base / rel).resolve(strict=True)
    if not candidate.is_file() or not is_relative_to(candidate, base):
        raise DataAccessError("symlink escape or not a file")
    forbidden = Path(sealed_root or os.getenv("LOCO_SEALED_ROOT", "/srv/loco-sealed")).resolve()
    if is_relative_to(candidate, forbidden):
        raise DataAccessError("sealed path")
    return candidate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: str | Path) -> dict:
    base = validate_root(root)
    images = []
    for cat in CATEGORIES:
        for split in ALLOWED_SPLITS:
            folder = base / cat / split
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = path.relative_to(base).as_posix()
                validated = resolve_good_image(base, relative)
                images.append({"path": relative, "sha256": sha256_file(validated), "category": cat, "split": split})
    return {"schema_version": "1.0", "type": "good_only", "root": str(base), "images": images}


def validate_manifest(manifest: dict, verify_hashes: bool = True) -> list[Path]:
    if manifest.get("type") != "good_only" or manifest.get("schema_version") != "1.0":
        raise DataAccessError("unapproved manifest type/version")
    root = manifest["root"]
    if not isinstance(manifest.get("images"), list):
        raise DataAccessError("invalid image entries")
    paths = []
    seen = set()
    for entry in manifest["images"]:
        if set(entry) != {"path", "sha256", "category", "split"}:
            raise DataAccessError("unexpected manifest fields")
        path = entry["path"]
        if path in seen:
            raise DataAccessError("duplicate manifest entry")
        seen.add(path)
        actual = resolve_good_image(root, path)
        if entry["category"] != Path(path).parts[0] or entry["split"] != "/".join(Path(path).parts[1:3]):
            raise DataAccessError("forged category or split")
        if verify_hashes and sha256_file(actual) != entry["sha256"]:
            raise DataAccessError("dataset image hash changed")
        paths.append(actual)
    return paths


def load_manifest(path: str | Path, verify_hashes: bool = True) -> tuple[dict, list[Path]]:
    manifest = json.loads(Path(path).read_text())
    return manifest, validate_manifest(manifest, verify_hashes=verify_hashes)


def manifest_fingerprint(manifest: dict) -> str:
    canonical = {"schema_version": manifest["schema_version"], "type": manifest["type"], "images": sorted(manifest["images"], key=lambda e: e["path"])}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
