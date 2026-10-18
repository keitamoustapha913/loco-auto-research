import hashlib
from pathlib import Path
import pytest
from PIL import Image
from benchmark.data_guard import build_manifest, validate_manifest, resolve_good_image, DataAccessError, manifest_fingerprint

@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / 'good'
    for split in ('train/good', 'validation/good'):
        folder = root / 'breakfast_box' / split
        folder.mkdir(parents=True)
        Image.new('RGB', (8, 8)).save(folder / '0.png')
    return root

def test_build_and_validate_manifest(tiny_dataset):
    manifest = build_manifest(tiny_dataset)
    assert len(validate_manifest(manifest)) == 2
    assert len(manifest_fingerprint(manifest)) == 64

def test_test_split_denied(tiny_dataset):
    forbidden = tiny_dataset / 'breakfast_box' / 'test' / 'logical_anomaly'
    forbidden.mkdir(parents=True)
    Image.new('RGB',(8,8)).save(forbidden/'bad.png')
    with pytest.raises(DataAccessError):
        resolve_good_image(tiny_dataset, 'breakfast_box/test/logical_anomaly/bad.png')

def test_parent_traversal_denied(tiny_dataset):
    with pytest.raises(DataAccessError):
        resolve_good_image(tiny_dataset,'../secret/train/good/0.png')

def test_symlink_escape_denied(tiny_dataset, tmp_path):
    outside = tmp_path/'secret.png'
    Image.new('RGB',(8,8)).save(outside)
    (tiny_dataset/'breakfast_box'/'train'/'good'/'link.png').symlink_to(outside)
    with pytest.raises(DataAccessError):
        resolve_good_image(tiny_dataset,'breakfast_box/train/good/link.png')

def test_modified_image_hash_denied(tiny_dataset):
    manifest = build_manifest(tiny_dataset)
    file_path = tiny_dataset / manifest['images'][0]['path']
    file_path.write_bytes(b'changed image')
    with pytest.raises(DataAccessError):
        validate_manifest(manifest)

def test_fake_split_manifest_denied(tiny_dataset):
    manifest = build_manifest(tiny_dataset)
    manifest['images'][0]['split'] = 'test/good'
    with pytest.raises(DataAccessError):
        validate_manifest(manifest)

def test_root_overlaps_sealed_denied(tiny_dataset):
    with pytest.raises(DataAccessError):
        resolve_good_image(tiny_dataset, 'breakfast_box/train/good/0.png', sealed_root=tiny_dataset)
