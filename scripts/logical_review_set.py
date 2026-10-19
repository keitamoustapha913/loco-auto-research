"""Good-only, provisional fruit-label relationship review package.

The renderer proposes cases; it cannot certify product logic or localization.
All edited sources are previously used development images. No model is run.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from benchmark.data_guard import load_manifest, manifest_fingerprint, sha256_file

PROTOCOL = 'logical-review-v1'
DECISIONS = {'violation', 'valid', 'uncertain'}


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checked_source(path, entries, paths):
    entry = entries.get(path)
    if entry is None or entry['category'] != 'juice_bottle' or entry['split'] != 'validation/good':
        raise PermissionError('only listed good validation images may be used')
    actual = paths[path]
    if sha256_file(actual) != entry['sha256']:
        raise ValueError('good source hash changed')
    image = Image.open(actual).convert('RGB')
    if image.size != (800, 1600):
        raise ValueError('unexpected source geometry')
    return image, entry['sha256']


def sticker_edit(target, donor, target_box, donor_box):
    """Same deterministic paste and alpha geometry for both donor conditions."""
    tx0, ty0, tx1, ty1 = target_box
    dx0, dy0, dx1, dy1 = donor_box
    if not (0 <= tx0 < tx1 <= target.width and 0 <= ty0 < ty1 <= target.height and
            0 <= dx0 < dx1 <= donor.width and 0 <= dy0 < dy1 <= donor.height):
        raise ValueError('sticker region outside source image')
    full_width, full_height = tx1 - tx0, ty1 - ty0
    donor_width, donor_height = dx1 - dx0, dy1 - dy0
    # Replace only the printed interior. Keeping the target sticker border
    # avoids a bright donor-border stripe and makes the intended component
    # relationship easier to inspect. The interior seam remains a confound.
    x_fraction, y_fraction = (.12, .88), (.14, .86)
    target_inner = (tx0 + round(x_fraction[0] * full_width), ty0 + round(y_fraction[0] * full_height),
                    tx0 + round(x_fraction[1] * full_width), ty0 + round(y_fraction[1] * full_height))
    donor_inner = (dx0 + round(x_fraction[0] * donor_width), dy0 + round(y_fraction[0] * donor_height),
                   dx0 + round(x_fraction[1] * donor_width), dy0 + round(y_fraction[1] * donor_height))
    ix0, iy0, ix1, iy1 = target_inner
    width, height = ix1 - ix0, iy1 - iy0
    patch = donor.crop(donor_inner).resize((width, height), Image.Resampling.BICUBIC)
    edge = Image.new('L', (width, height), 0)
    ImageDraw.Draw(edge).rectangle((6, 6, width - 7, height - 7), fill=255)
    alpha = edge.filter(ImageFilter.GaussianBlur(5))
    edited = target.copy()
    edited.paste(patch, (ix0, iy0), alpha)
    # Reviewer-only proposal. Deliberately not accepted as a semantic target.
    proposed = Image.new('L', target.size, 0)
    ImageDraw.Draw(proposed).rectangle((tx0 + round(.28 * width), ty0 + round(.22 * height),
                                        tx0 + round(.72 * width), ty0 + round(.78 * height)), fill=255)
    footprint = Image.new('L', target.size, 0)
    footprint.paste(alpha, (ix0, iy0))
    return edited, proposed, footprint


def _save_preview(edited, region, path):
    preview = edited.copy()
    mask = np.asarray(region) > 0
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('empty proposed region')
    ImageDraw.Draw(preview).rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                                      outline=(65, 190, 220), width=5)
    preview.save(path)


def _review_html(cards, rule, output):
    body = []
    for card in cards:
        ident = html.escape(card['id'])
        body.append(f'''<section><h2>Case {ident}</h2>
<p><strong>Question:</strong> {html.escape(rule)}</p>
<p>Does the edited bottle violate this rule? Judge the fruit and liquid relationship separately from visible paste artifacts.</p>
<div class="images"><figure><img src="{ident}_source.png"><figcaption>Authorized good original</figcaption></figure>
<figure><img src="{ident}_edit.png"><figcaption>Edited candidate</figcaption></figure>
<figure><img src="{ident}_preview.png"><figcaption>Proposed region for discussion only; not ground truth</figcaption></figure></div>
<p>Record valid / violation / uncertain, whether the rule is supported, whether the edit artifact dominates, and whether the proposed region is defensible. Use the separate review form.</p></section>''')
    output.write_text('<!doctype html><meta charset="utf-8"><title>Logical review v1</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:24px;color:#24292f}'
        'section{border-top:1px solid #ddd;padding:20px 0}.images{display:flex;gap:16px;flex-wrap:wrap}'
        'figure{margin:0;flex:1 1 220px;max-width:340px}img{width:100%;height:auto}figcaption{font-size:14px}</style>'
        '<h1>Juice bottle: provisional logical review</h1>'
        '<p>All images derive from authorized good validation images already used in development. '
        'The edit generator cannot certify validity. No case is an independent confirmation sample. '
        'The blue box is a draft attribution prompt, never a metric mask. '
        'Copy <a href="review_form_template.json">the review form</a>, enter your own reviewer identity, '
        'and decide each opaque case without opening the investigator manifest.</p>' + ''.join(body))


def build(manifest_path, config_path, output):
    config = json.loads(Path(config_path).read_text())
    if (config.get('protocol') != PROTOCOL or config.get('official_test_access') is not False or
            config.get('reviewers_required') != 2 or config.get('original_size') != [800, 1600] or
            config.get('eligible_source_partition') != 'stage6_assessment_validation_images_18_to_53'):
        raise ValueError('unapproved review protocol')
    manifest, resolved = load_manifest(manifest_path)
    if manifest_fingerprint(manifest) != config['dataset_fingerprint']:
        raise ValueError('dataset fingerprint differs from good-only protocol')
    entries = {row['path']: row for row in manifest['images']}
    paths = {row['path']: path for row, path in zip(manifest['images'], resolved)}
    val_paths = [row['path'] for row in manifest['images']
                 if row['category'] == 'juice_bottle' and row['split'] == 'validation/good']
    assessment_paths = set(val_paths[max(1, len(val_paths) // 3):])
    if len(val_paths) < 3:
        raise ValueError('insufficient good validation images to separate calibration')
    output = Path(output)
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise FileExistsError('review output must be a new empty real directory')
    output.mkdir(parents=True, exist_ok=True)
    cards, cases = [], []
    for group in config['cases']:
        if {group['target'], group['valid_donor'], group['cross_donor']} - assessment_paths:
            raise PermissionError('review target and donor must be outside prior calibration partition')
        target, target_sha = checked_source(group['target'], entries, paths)
        for condition, donor_key in (('same_variant', 'valid_donor'), ('cross_variant', 'cross_donor')):
            donor, donor_sha = checked_source(group[donor_key], entries, paths)
            if group['target'] == group[donor_key]:
                raise ValueError('donor must be a separate good source')
            donor_variant = group['valid_donor_variant' if condition == 'same_variant' else 'cross_donor_variant']
            if (condition == 'same_variant') != (group['target_variant'] == donor_variant):
                raise ValueError('donor variant configuration inconsistent')
            edited, proposed, footprint = sticker_edit(target, donor, group['target_sticker_xyxy'],
                group['valid_donor_sticker_xyxy' if condition == 'same_variant' else 'cross_donor_sticker_xyxy'])
            case_key = digest({'target': group['target'], 'donor': group[donor_key], 'condition': condition})[:16]
            # Opaque review names conceal the investigator's intended status.
            edited.save(output / f'{case_key}_edit.png')
            target.save(output / f'{case_key}_source.png')
            proposed.save(output / f'{case_key}_draft_region.png')
            footprint.save(output / f'{case_key}_edit_footprint.png')
            _save_preview(edited, proposed, output / f'{case_key}_preview.png')
            case = {'id': case_key, 'source_path': group['target'], 'source_sha256': target_sha,
                    'donor_path': group[donor_key], 'donor_sha256': donor_sha,
                    'source_group': group['target'].rsplit('/', 1)[-1].split('.')[0],
                    'prior_use': 'Stage 5-8 development and/or assessment; not independent confirmation',
                    'operation': 'sticker_interior_swap_feather5_v1', 'condition': condition,
                    'intended_status': 'putative_valid' if condition == 'same_variant' else 'putative_violation',
                    'rule_id': config['rule_id'], 'rule_question': config['rule_question'],
                    'proposed_violation': 'Fruit label may disagree with liquid variant' if condition == 'cross_variant' else None,
                    'proposed_region_semantics': 'fruit illustration on swapped sticker; paired liquid relation remains attribution-ambiguous',
                    'draft_region_path': f'{case_key}_draft_region.png',
                    'edit_footprint_path': f'{case_key}_edit_footprint.png',
                    'image_path': f'{case_key}_edit.png', 'source_image_path': f'{case_key}_source.png',
                    'review_status': 'provisional'}
            for key in ('draft_region_path', 'edit_footprint_path', 'image_path', 'source_image_path'):
                case[key.replace('_path', '_sha256')] = sha256_file(output / case[key])
            cases.append(case)
            cards.append({'id': case_key})
    bundle = {'protocol': PROTOCOL, 'dataset_fingerprint': config['dataset_fingerprint'],
              'config_sha256': sha256_file(Path(config_path)), 'generator_sha256': sha256_file(Path(__file__)),
              'independent_confirmation': False, 'normal_source_groups': sorted({x['source_group'] for x in cases}),
              'cases': cases}
    (output / 'investigator_manifest.json').write_text(json.dumps(bundle, indent=2, sort_keys=True))
    (output / 'review_form_template.json').write_text(json.dumps(template(bundle, 'REPLACE_WITH_REVIEWER_ID'), indent=2))
    _review_html(sorted(cards, key=lambda x: x['id']), config['rule_question'], output / 'review.html')
    return bundle


def template(bundle, reviewer_id):
    if not reviewer_id.strip():
        raise ValueError('reviewer identity required')
    return {'protocol': PROTOCOL, 'reviewer_id': reviewer_id,
            'instructions': 'Decide from product logic, not paste seams. Set each field; uncertain is acceptable.',
            'cases': [{'id': c['id'], 'case_sha256': digest(c), 'rule_supported': 'uncertain',
                       'logical_decision': 'uncertain', 'artifact_dominates': 'uncertain',
                       'region_decision': 'uncertain', 'reason': ''} for c in bundle['cases']]}


def verify_artifacts(bundle, directory):
    root = Path(directory).resolve(strict=True)
    if bundle.get('protocol') != PROTOCOL or bundle.get('independent_confirmation') is not False:
        raise ValueError('invalid review bundle')
    for case in bundle['cases']:
        for key in ('draft_region_path', 'edit_footprint_path', 'image_path', 'source_image_path'):
            relative = Path(case[key])
            if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 1:
                raise PermissionError('review artifact path escapes bundle')
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or sha256_file(path) != case[key.replace('_path', '_sha256')]:
                raise ValueError('review artifact hash or path changed')
            with Image.open(path) as image:
                if image.size != (800, 1600):
                    raise ValueError('review artifact geometry changed')
    return True


def adjudicate(bundle, reviewer_files):
    if len(reviewer_files) != 2:
        raise ValueError('exactly two independent reviewer files required')
    forms = [json.loads(Path(p).read_text()) for p in reviewer_files]
    if any(f.get('protocol') != PROTOCOL or not f.get('reviewer_id') for f in forms):
        raise ValueError('invalid review form')
    if forms[0]['reviewer_id'] == forms[1]['reviewer_id']:
        raise ValueError('reviewer identities must differ')
    by_id = {case['id']: case for case in bundle['cases']}
    outcomes = {}
    for case_id, case in by_id.items():
        opinions = []
        for form in forms:
            entries = [x for x in form['cases'] if x.get('id') == case_id]
            if len(entries) != 1 or entries[0].get('case_sha256') != digest(case):
                raise ValueError('review missing, duplicated or case hash changed')
            row = entries[0]
            if (row.get('logical_decision') not in DECISIONS or
                row.get('rule_supported') not in {'yes', 'no', 'uncertain'} or
                row.get('artifact_dominates') not in {'yes', 'no', 'uncertain'} or
                row.get('region_decision') not in {'accept', 'reject', 'uncertain'}):
                raise ValueError('invalid review decision')
            opinions.append(row)
        expected = 'valid' if case['condition'] == 'same_variant' else 'violation'
        if any(x['logical_decision'] == 'uncertain' or x['rule_supported'] == 'uncertain' or
               x['artifact_dominates'] == 'uncertain' or x['region_decision'] == 'uncertain' for x in opinions):
            status = 'uncertain'
        elif any(x['logical_decision'] != expected or x['rule_supported'] != 'yes' or
                 x['artifact_dominates'] != 'no' or
                 (expected == 'violation' and x['region_decision'] != 'accept') or
                 len(x.get('reason', '').strip()) < 15 for x in opinions):
            status = 'rejected'
        else:
            status = 'reviewed_valid' if expected == 'valid' else 'reviewed_violation_logic_only'
        outcomes[case_id] = {'status': status, 'rule_id': case['rule_id'],
                             'source_path': case['source_path'], 'source_sha256': case['source_sha256'],
                             'region_semantics': case['proposed_region_semantics'],
                             'reviewers': [{k: row[k] for k in ('logical_decision', 'rule_supported',
                                           'artifact_dominates', 'region_decision', 'reason')} |
                                           {'reviewer_id': form['reviewer_id']}
                                           for form, row in zip(forms, opinions)]}
    return {'protocol': PROTOCOL, 'case_manifest_sha256': digest(bundle), 'outcomes': outcomes,
            'metric_ready_violations': 0,
            'note': 'Independent semantic target masks and separate target acceptance are still required.'}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('template')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('adjudicate')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--reviews', nargs=2, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.manifest, args.config, args.output)
        print(json.dumps({'cases': len(result['cases']), 'independent_confirmation': False}))
    elif args.command == 'template':
        bundle = json.loads(args.bundle.read_text())
        verify_artifacts(bundle, args.bundle.parent)
        args.output.write_text(json.dumps(template(bundle, args.reviewer), indent=2))
    else:
        bundle = json.loads(args.bundle.read_text())
        verify_artifacts(bundle, args.bundle.parent)
        args.output.write_text(json.dumps(adjudicate(bundle, args.reviews), indent=2))


if __name__ == '__main__':
    main()
