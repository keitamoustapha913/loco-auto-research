"""Trusted real-development evaluator; compatible sPRO from DOI 10.1007/s11263-022-01578-9.
Original reference-code equivalence is unverified. Synthetic/historical metrics remain separate.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

VERSION='real-development-v2.0-compatible'
LOCALIZATION_STATUS='compatible development implementation; original-scorer equivalence unverified'


def detection_metrics(rows, threshold):
    if not np.isfinite(threshold): raise ValueError('finite normal-only threshold required')
    if any(not np.isfinite(r['image_score']) for r in rows): raise ValueError('nonfinite image score')
    if any(r['category'] not in {'good','logical_anomalies','structural_anomalies'} for r in rows):
        raise ValueError('unknown development category')
    good=[r['image_score'] for r in rows if r['category']=='good']
    if not good: raise ValueError('development normals required for detection/FPR')
    metrics={'dev_real/normal_images':len(good),'dev_real/normal_false_positives':sum(s>threshold for s in good),
             'dev_real/normal_image_fpr':float(np.mean(np.asarray(good)>threshold)),
             'dev_real/normal_only_threshold':float(threshold)}
    for kind in ('logical','structural'):
        anomaly=[r['image_score'] for r in rows if r['category']==kind+'_anomalies']
        if not anomaly: raise ValueError('both anomaly categories required')
        metrics[f'dev_real/{kind}_image_auroc']=float(roc_auc_score([0]*len(good)+[1]*len(anomaly),good+anomaly))
        metrics[f'dev_real/{kind}_images']=len(anomaly)
    return metrics


def saturated_region_curve(background_scores, regions, limit=0.05):
    """Exact tied-score sweep of Eq. (1), Bergmann et al. (2022), §§3.3–3.4.

    regions contains (scores within one annotated mask/defect, saturation pixels).
    Background is outside the union, including good images. Each region has equal
    weight; disconnected parts of one annotated defect remain one region.
    """
    bg = np.sort(np.asarray(background_scores, dtype=np.float32).ravel())
    if not bg.size or not np.isfinite(bg).all() or not regions or not 0 < limit <= 1:
        raise ValueError('finite background, regions and valid integration limit required')
    # Include the entire tie that crosses the integration boundary.
    cutoff = bg[max(0, bg.size - int(np.ceil(limit * bg.size)))]
    knots, counts = np.unique(bg[bg >= cutoff], return_counts=True)
    xs = [knots]; fps = [counts.astype(np.float64) / bg.size]; gains = [np.zeros(len(knots))]
    for values, saturation in regions:
        values = np.sort(np.asarray(values, dtype=np.float32).ravel())
        if not values.size or not np.isfinite(values).all() or not 0 < saturation <= values.size:
            raise ValueError('invalid region saturation or scores')
        take = int(np.ceil(saturation))
        credited = values[-take:]
        unique, n = np.unique(credited, return_counts=True)
        weight = n.astype(np.float64)
        weight[0] -= take - saturation
        keep = unique >= cutoff
        xs.append(unique[keep]); fps.append(np.zeros(keep.sum()))
        gains.append(weight[keep] / saturation / len(regions))
    scores = np.concatenate(xs); fp = np.concatenate(fps); gain = np.concatenate(gains)
    order = np.argsort(-scores, kind='stable')
    scores, fp, gain = scores[order], fp[order], gain[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
    x = np.r_[0., np.cumsum(fp)[ends]]; y = np.r_[0., np.cumsum(gain)[ends]]
    crossing = np.searchsorted(x, limit, side='right')
    if crossing < len(x):
        boundary = np.interp(limit, x[crossing-1:crossing+1], y[crossing-1:crossing+1])
        x, y = np.r_[x[:crossing], limit], np.r_[y[:crossing], boundary]
    elif x[-1] < limit:
        # Floating point accumulation may end a few ulps below the limit.
        x, y = np.r_[x, limit], np.r_[y, y[-1]]
    return float(np.clip(np.trapezoid(y, x) / limit, 0, 1))


def localization_metrics(manifest, data_root, output, check=lambda: None):
    """Trusted mask reader. Native resolution, exact pooled diagnostics, no fitting."""
    import json
    from pathlib import Path
    from PIL import Image
    from sklearn.metrics import average_precision_score
    lookup = {d['pixel_value']: d for d in json.loads((Path(data_root)/'juice_bottle/defects_config.json').read_text())}
    metrics = {}
    for kind in ('logical', 'structural'):
        backgrounds, regions, pixels, labels = [], [], [], []
        for index, entry in enumerate(manifest['images']):
            check()
            if entry['category'] not in ('good', kind+'_anomalies'): continue
            scores = np.load(Path(output)/f'case_{index:04d}'/'map.npy', allow_pickle=False)
            if scores.shape != (entry['height'],entry['width']) or not np.isfinite(scores).all():
                raise ValueError('unaligned/nonfinite full-resolution map')
            union = np.zeros(scores.shape, dtype=bool)
            for record in entry['masks']:
                mask = np.asarray(Image.open(Path(data_root)/record['path']))
                if mask.shape != scores.shape: raise ValueError('unaligned annotation')
                for value in record['pixel_values']:
                    region = mask == value; size = int(region.sum()); spec = lookup[value]
                    saturation = spec['saturation_threshold'] * size if spec['relative_saturation'] else spec['saturation_threshold']
                    # Published definition requires s <= |A|; annotation areas can
                    # be smaller than the category's fixed typical-object size.
                    regions.append((scores[region], min(float(saturation), size)))
                    union |= region
            backgrounds.append(scores[~union])
            if kind == 'logical': pixels.append(scores.ravel()); labels.append(union.ravel())
        check()
        metrics[f'dev_real/{kind}_au_spro_0_05_compatible'] = saturated_region_curve(np.concatenate(backgrounds), regions)
        metrics[f'dev_real/{kind}_annotation_regions'] = len(regions)
        metrics[f'dev_real/{kind}_background_pixels'] = sum(x.size for x in backgrounds)
        if kind == 'logical':
            check()
            y, s = np.concatenate(labels), np.concatenate(pixels)
            metrics['dev_real/logical_pixel_ap_pooled'] = float(average_precision_score(y,s))
            check()
            metrics['dev_real/logical_pixel_auroc_pooled'] = float(roc_auc_score(y,s))
        check()
    return metrics


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from benchmark.feedback_data import verify_import
    parser=argparse.ArgumentParser(description='Internal trusted scorer; managed by loco-lab')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--maps',required=True)
    parser.add_argument('--result',required=True)
    args=parser.parse_args()
    manifest=verify_import(args.data_root)
    result=localization_metrics(manifest,args.data_root,args.maps)
    Path(args.result).write_text(json.dumps(result,sort_keys=True))
