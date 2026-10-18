"""Prepare blinded human-review forms and validate independent decisions.

Reviewers inspect the authorized source, edited image and draft regions. The
forms contain no model scores. Two separate reviewer identities are necessary
but identity authenticity remains an operator responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from benchmark.semantic_validation import case_digest, independent_review_status


def blind_id(case, reviewer_id):
    return hashlib.sha256((case_digest(case) + ':' + reviewer_id).encode()).hexdigest()[:20]


def prepare(case_dir: Path, reviewer_id: str, packet_dir: Path | None = None):
    if not reviewer_id.strip():
        raise ValueError('reviewer identity required')
    bundle = json.loads((case_dir / 'case_manifest.json').read_text())
    normal_by_source = {case['source_path']: case for case in bundle['cases'] if case['kind'] == 'normal'}
    if packet_dir is not None:
        packet_dir = Path(packet_dir)
        if packet_dir.is_symlink():
            raise PermissionError('review packet directory must not be a symlink')
        if packet_dir.exists() and any(packet_dir.iterdir()):
            raise FileExistsError('review packet directory must be empty')
        packet_dir.mkdir(parents=True, exist_ok=True)
    forms = []
    for case in bundle['cases']:
        if case['kind'] == 'normal':
            continue
        code = blind_id(case, reviewer_id)
        if packet_dir is not None:
            source = normal_by_source[case['source_path']]
            for filename, relative in (('source.png', source['image_path']),
                                       ('edit.png', case['image_path']),
                                       ('liquid_draft.png', case['semantic_region_paths']['liquid_draft']),
                                       ('pictogram_draft.png', case['semantic_region_paths']['pictogram_draft'])):
                path = (Path(case_dir) / relative).resolve(strict=True)
                if not path.is_relative_to(Path(case_dir).resolve(strict=True)):
                    raise PermissionError('review packet source path escapes case bundle')
                shutil.copyfile(path, packet_dir / f'{code}_{filename}')
        forms.append({'blind_id': code, 'reviewer_id': reviewer_id,
                      'source_image': f'{code}_source.png', 'edited_image': f'{code}_edit.png',
                      'liquid_draft_region': f'{code}_liquid_draft.png',
                      'pictogram_draft_region': f'{code}_pictogram_draft.png',
                      'verdict': 'indeterminate', 'source_inspected': False,
                      'blind_to_scores': True, 'target_region_accepted': False,
                      'reason': 'Explain the proposed rule and visual evidence here.'})
    return sorted(forms, key=lambda row: row['blind_id'])


def merge(case_dir: Path, review_paths):
    bundle = json.loads((case_dir / 'case_manifest.json').read_text())
    reviews = []
    known_cases = {case['sample_id']: case for case in bundle['cases'] if case['kind'] != 'normal'}
    for path in review_paths:
        part = json.loads(Path(path).read_text())
        if not isinstance(part, list):
            raise ValueError('each reviewer file must contain a list')
        for form in part:
            reviewer = form.get('reviewer_id', '')
            matches = [case for case in known_cases.values()
                       if blind_id(case, reviewer) == form.get('blind_id')]
            if len(matches) != 1:
                raise ValueError('review refers to an unknown or ambiguous blinded case')
            case = matches[0]
            reviews.append({'sample_id': case['sample_id'], 'case_sha256': case_digest(case),
                            'reviewer_id': reviewer, 'verdict': form.get('verdict'),
                            'source_inspected': form.get('source_inspected'),
                            'blind_to_scores': form.get('blind_to_scores'),
                            'target_region_accepted': form.get('target_region_accepted'),
                            'reason': form.get('reason')})
    statuses = {case['sample_id']: independent_review_status(case, reviews)
                for case in bundle['cases'] if case['kind'] != 'normal'}
    return reviews, statuses


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', required=True, type=Path)
    p.add_argument('--reviewer')
    p.add_argument('--packet-dir', type=Path)
    p.add_argument('--merge', nargs='*', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if bool(args.reviewer) == (args.merge is not None):
        raise ValueError('choose exactly one of --reviewer or --merge')
    if args.reviewer:
        if args.packet_dir is None:
            raise ValueError('--packet-dir is required for blinded review')
        result = prepare(args.cases, args.reviewer, args.packet_dir)
    else:
        result, statuses = merge(args.cases, args.merge)
        print(json.dumps(statuses, sort_keys=True))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
