"""Pinned custom development partition. Never accepts a final-holdout manifest."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from collections import Counter
import numpy as np
from PIL import Image
from benchmark.data_guard import sha256_file

APPROVAL = Path(__file__).resolve().parents[1] / 'configs/development_feedback.json'
LABEL = 'LOCO custom-split development feedback — not official full-test results.'
CATEGORIES = {'good', 'logical_anomalies', 'structural_anomalies'}


def approval():
    return json.loads(APPROVAL.read_text())


def schema(manifest, approved=None):
    approved = approved or approval()
    if (manifest.get('schema_version') != 1 or manifest.get('partition') != 'development' or
        manifest.get('split_id') != approved['split_id'] or manifest.get('hash_algorithm') != 'sha256' or
        manifest.get('dataset') != 'MVTec LOCO AD juice_bottle' or
        manifest.get('package', {}).get('sha256') != approved['archive_sha256']):
        raise ValueError('unapproved development schema, partition, split identifier or archive hash')
    if set(manifest) != {'schema_version','dataset','defect_encoding','hash_algorithm','images','mask_association',
                         'metadata','package','partition','path_base','split_id','summary'}:
        raise ValueError('unexpected manifest fields; private holdout metadata is not permitted')
    canonical = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if canonical != approved["manifest_canonical_sha256"]:
        raise ValueError("development manifest integrity check failed")
    records = {}
    def add(row):
        p = PurePosixPath(row['path'])
        if p.is_absolute() or '..' in p.parts or '\\' in row['path'] or str(p) != row['path']:
            raise ValueError('unsafe archive path')
        if row['path'] in records or not isinstance(row['size_bytes'], int) or not 0 < row['size_bytes'] < 20_000_000:
            raise ValueError('duplicate path or invalid member size')
        if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('invalid member digest')
        records[row['path']] = row
    metadata = {f'juice_bottle/{x}' for x in ('defects_config.json','license.txt','readme.txt')}
    if {r['path'] for r in manifest['metadata']} != metadata:
        raise ValueError('only approved public category metadata allowed')
    for row in manifest['metadata']:
        add(row)
    for row in manifest['images']:
        cat = row['category']; path = PurePosixPath(row['path'])
        if cat not in CATEGORIES or path.parent.as_posix() != f'juice_bottle/test/{cat}' or path.suffix != '.png':
            raise ValueError('unapproved development image path')
        if row['mode'] != 'RGB' or row['width'] != 800 or row['height'] != 1600:
            raise ValueError('unexpected image geometry or mode')
        if (cat == 'good') != (not row['masks']):
            raise ValueError('good/anomaly mask association invalid')
        add(row)
        for mask in row['masks']:
            mp = PurePosixPath(mask['path'])
            if mp.parent.as_posix() != f'juice_bottle/ground_truth/{cat}/{path.stem}' or mp.suffix != '.png':
                raise ValueError('mask belongs to another image')
            if (mask['width'],mask['height'],mask['mode']) != (row['width'],row['height'],'L'):
                raise ValueError('unaligned mask geometry')
            add(mask)
        if set(row['defect_types']) != {d for mask in row['masks'] for d in mask['defect_types']}:
            raise ValueError('image defect associations differ from masks')
    summary = manifest['summary']
    masks = [r for image in manifest['images'] for r in image['masks']]
    if (summary['images'] != len(manifest['images']) or summary['masks'] != len(masks) or
        summary['categories'] != dict(Counter(r['category'] for r in manifest['images'])) or
        summary['multiple_mask_images'] != sum(len(r['masks']) > 1 for r in manifest['images']) or
        summary['defect_mask_counts'] != dict(Counter(d for r in masks for d in r['defect_types'])) or
        summary['defect_image_counts'] != dict(Counter(d for r in manifest['images'] for d in set(r['defect_types'])))):
        raise ValueError('manifest summary counts differ')
    return records


def validate_archive(incoming, approved=None):
    incoming = Path(incoming)
    manifest_path, archive = incoming/'dev-manifest.json', incoming/'dev-feedback.tar.gz'
    if not manifest_path.is_file() or not archive.is_file():
        raise FileNotFoundError(f'Required transfer missing: {manifest_path} and {archive}')
    manifest = json.loads(manifest_path.read_text())
    records = schema(manifest, approved)
    if archive.stat().st_size != manifest['package']['size_bytes'] or sha256_file(archive) != manifest['package']['sha256']:
        raise ValueError('development archive integrity check failed')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(records) or {r.name for r in members} != set(records):
            raise ValueError('archive member set differs from manifest')
        defects = json.load(tar.extractfile('juice_bottle/defects_config.json'))
        lookup = {r['pixel_value']: r['defect_name'] for r in defects}
        for member in members:
            row = records[member.name]
            if not member.isfile() or member.size != row['size_bytes']:
                raise ValueError('links, special files or incorrect size forbidden')
            raw = tar.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('archive member integrity failure')
            if 'width' in row:
                image = Image.open(io.BytesIO(raw)); image.load()
                if image.size != (row['width'],row['height']) or image.mode != row['mode']:
                    raise ValueError('image metadata mismatch')
                if 'pixel_values' in row:
                    values = sorted(int(v) for v in np.unique(image) if v)
                    if values != row['pixel_values'] or sorted(lookup.get(v) for v in values) != sorted(row['defect_types']):
                        raise ValueError('mask encoding or defect semantics mismatch')
    return manifest


def imported_root(runtime):
    return Path(runtime)/'feedback/juice_bottle/v1'


def import_development(runtime):
    runtime = Path(runtime).resolve(strict=True)
    incoming = runtime/'incoming/juice_bottle_dev_v1'
    manifest = validate_archive(incoming)
    destination = imported_root(runtime)
    if any(p.is_symlink() for p in (destination, destination.parent, destination.parent.parent)):
        raise ValueError("feedback destination must not use symlinks")
    if destination.exists():
        verify_import(destination)
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.feedback-import-', dir=destination.parent) as tmp:
        stage = Path(tmp)/'v1'; stage.mkdir()
        records = schema(manifest)
        with tarfile.open(incoming/'dev-feedback.tar.gz','r:gz') as tar:
            for member in tar:
                if member.name not in records or not member.isfile() or member.size != records[member.name]['size_bytes']:
                    raise ValueError('archive changed after validation')
                path = stage/member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as out:
                    shutil.copyfileobj(tar.extractfile(member), out)
                path.chmod(0o444)
        (stage/'dev-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
        (stage/'dev-manifest.json').chmod(0o444)
        stage.rename(destination)
    verify_import(destination)
    return manifest


def verify_import(root):
    root = Path(root)
    if root.is_symlink() or (root / "dev-manifest.json").is_symlink():
        raise ValueError("feedback root/manifest must not be symlinks")
    root = root.resolve(strict=True)
    manifest = json.loads((root/'dev-manifest.json').read_text())
    records = schema(manifest)
    for name, row in records.items():
        path = root/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or sha256_file(path) != row['sha256']:
            raise ValueError('imported development member changed or escaped')
    return manifest
