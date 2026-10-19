"""Real-development detection metrics. No substitution for unavailable AU-sPRO."""
import numpy as np
from sklearn.metrics import roc_auc_score

VERSION='real-development-detection-v1.0'
LOCALIZATION_BLOCKER='Official MVTec LOCO reference archive returned HTTP 404; AU-sPRO unavailable pending verified scorer integration.'


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
