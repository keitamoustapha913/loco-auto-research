"""Sample-aware reporting and measured paired controls for future confirmations."""
import hashlib
import json
import math
from pathlib import Path


def normal_interval(false_positives, normal_images):
    """95% Wilson interval, separately from seed or paired-difference uncertainty."""
    if not 0 <= false_positives <= normal_images or normal_images < 1:
        raise ValueError('invalid normal false-positive counts')
    z=1.959963984540054;n=normal_images;p=false_positives/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return {'false_positives':false_positives,'normal_images':n,'rate':p,
            'lower':max(0,center-radius),'upper':min(1,center+radius),'method':'Wilson 95% marginal binomial',
            'limitation':'Not a paired difference interval; repeated development selection and seed variability are separate.'}


def verify_pairs(plan, rows):
    """Compare actual recorded input/state bytes, never equal seed labels alone."""
    from orchestrator.experiment_plan import digest
    requirements=(plan.get('confirmation_requirements') or {}).get('paired_controls',[])
    if not requirements:return {'status':'NOT_MEASURED','pairs':[]}
    cells={}
    try:
        for row in rows:
            cfg=json.loads(row['config_json'])
            if cfg.get('plan_sha256')!=digest(plan):continue
            root=Path(row['artifact_dir']);timing=root/'training_timing.json'
            if root.is_symlink() or timing.is_symlink():raise ValueError('unsafe paired-control evidence')
            audit=json.loads(timing.read_text());receipt=audit.get('paired_controls',{})
            provenance=root/'run_manifest.json'
            if provenance.is_symlink():raise ValueError('unsafe paired-control provenance')
            hashes=json.loads(provenance.read_text()).get('artifact_hashes',{})
            if hashes.get('training_timing.json')!=hashlib.sha256(timing.read_bytes()).hexdigest():
                raise ValueError('paired-control timing provenance mismatch')
            if not audit.get('verified') or audit.get('status')!='COMPLETED' or receipt.get('version')!='trusted-prepared-controls-v1':
                raise ValueError('missing completed trusted paired-control receipt')
            observations=receipt['observed'];checked={}
            for name in requirements:
                value=observations[name]
                if name.startswith('input:'):
                    path=root/value['file']
                    if path.is_symlink() or path.parent!=root or not path.name.startswith('paired_input_'):
                        raise ValueError('unsafe paired-input path')
                    raw=path.read_bytes()
                    if (hashes.get(path.name)!=value['sha256'] or hashlib.sha256(raw).hexdigest()!=value['sha256'] or len(raw)!=value['updates']*32 or
                        value['updates']!=audit['completed_steps'] or not raw):
                        raise ValueError('paired-input integrity/count mismatch')
                    checked[name]=raw
                else:checked[name]=value
            cells[(cfg['plan_arm'],cfg.get('plan_pair_seed',cfg['seed']))]=checked
        pairs=[]
        for arm in plan['arms']:
            if arm['arm_id']==plan['baseline_arm_id']:continue
            for seed in plan['seeds']:
                a=cells[(plan['baseline_arm_id'],seed)];b=cells[(arm['arm_id'],seed)];coverage={}
                for name in requirements:
                    if isinstance(a[name],bytes):
                        n=min(len(a[name]),len(b[name]));same=a[name][:n]==b[name][:n]
                        coverage[name]={'shared_updates_verified':n//32,'baseline_updates':len(a[name])//32,'challenger_updates':len(b[name])//32}
                    else:same=a[name]==b[name]
                    if not same:raise ValueError(f'paired control differs: {name}, seed {seed}, arm {arm["arm_id"]}')
                pairs.append({'arm':arm['arm_id'],'seed':seed,'coverage':coverage})
        return {'status':'VERIFIED','pairs':pairs,
            'scope':'Declared controls only; exact common update prefix. Extra updates/exposures remain visible confounds.'}
    except (KeyError,ValueError,OSError,TypeError) as exc:
        return {'status':'UNVERIFIED','pairs':[],'reason':str(exc)}
