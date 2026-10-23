"""Lossless storage for trusted development maps; never mounted into candidates."""
from __future__ import annotations
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np

MAX_MAP_BYTES = 256 * 2**20


def safe_path(path, root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('feedback artifact root uses an unsafe symlink')
    path, root = Path(path), root.resolve(strict=True)
    if not path.is_absolute():
        path = root/path
    if '..' in path.parts or not path.resolve().is_relative_to(root):
        raise ValueError('feedback artifact escapes its trusted root')
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('feedback artifact uses an unsafe symlink')
        current = current.parent
    return path


def map_bytes(path, root=None, *, _seen=None):
    """Read original NPY bytes, including verified reference/compressed forms."""
    path = Path(path)
    if root is None:
        path = path.absolute()
        root = path.parent
    else:
        root = Path(root)
    path = safe_path(path, root)
    seen = set() if _seen is None else _seen
    if str(path) in seen or len(seen) >= 16:
        raise ValueError('cyclic or excessive feedback artifact references')
    seen.add(str(path))
    # Prefer the original file until a verified replacement has been published.
    try:
        with path.open('rb') as stream:
            data = stream.read(MAX_MAP_BYTES+1)
    except FileNotFoundError:
        ref = safe_path(Path(str(path)+'.ref.json'), root)
        if ref.exists():
            if ref.stat().st_size > 4096:
                raise ValueError('oversized feedback artifact reference')
            record = json.loads(ref.read_text())
            target = record['target']
            if Path(target).is_absolute() or '..' in Path(target).parts or record.get('version') != 1:
                raise ValueError('unsafe feedback artifact reference')
            data = map_bytes(safe_path(Path(root)/target, root), root, _seen=seen)
            if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('referenced feedback map integrity mismatch')
        else:
            shuffled = safe_path(Path(str(path)+'.bs4.gz'), root)
            packed = shuffled if shuffled.exists() else safe_path(Path(str(path)+'.gz'), root)
            with gzip.open(packed, 'rb') as stream:
                data = stream.read(MAX_MAP_BYTES+1)
            if packed == shuffled:
                if len(data)%4:
                    raise ValueError('invalid byte-shuffled map length')
                data = np.frombuffer(data,np.uint8).reshape(4,-1).T.tobytes()
    if len(data) > MAX_MAP_BYTES:
        raise ValueError('oversized feedback map')
    return data


def map_sha256(path, root=None):
    return hashlib.sha256(map_bytes(path, root)).hexdigest()


def load_map(path, root=None):
    return np.load(io.BytesIO(map_bytes(path, root)), allow_pickle=False)
