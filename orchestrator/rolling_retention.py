"""Nine model-producing runs; immutable numerical history outlives artifact bytes.

Only explicit bulky file types in ledger-owned job/archive/MLflow roots retire.
The prepared manifest is durable before unlink and is the availability authority
also after a crash. No dataset paths are enumerated by this module.
"""
from __future__ import annotations
import hashlib
import gzip
import os
import uuid
import shutil
import json
from pathlib import Path
from benchmark.data_guard import sha256_file
import time

TERMINAL = {'COMPLETED','REVIEWED','PROMOTED','REJECTED','FAILED','INVALID','INTERRUPTED','TIMED_OUT','OOM','CANCELLED'}
PREFIX = 'artifact_retention:'
BULKY = {'.pt','.pth','.npy','.png','.jpg','.jpeg','.webp','.npz'}


def availability(store, run_id):
    with store.connect() as con:
        row=con.execute('SELECT value FROM settings WHERE name=?',(PREFIX+run_id,)).fetchone()
    return json.loads(row[0]) if row else {'status':'FULL','checkpoint_available':True}


def _safe(path, root):
    # Check every component, including dangling symlinks; resolve alone is unsafe.
    path=Path(path);root=Path(root)
    relative=path.relative_to(root)
    if root.is_symlink() or root.resolve()!=root:raise ValueError('linked retention root or ancestor')
    current=root
    for part in relative.parts:
        if part in {'.','..'}:raise ValueError('unsafe retention path')
        current=current/part
        if current.is_symlink():raise ValueError('linked retention artifact')
    return path


def selection(store):
    from orchestrator import model_registry as registry
    from orchestrator.storage_cleanup import _refs
    models=registry.retained(store)
    with store.connect() as con:
        runs={r['run_id']:dict(r) for r in con.execute('SELECT * FROM runs')}
    producing={rid:r for rid,r in runs.items() if r['status'] in {'COMPLETED','REVIEWED','PROMOTED','REJECTED'} and
               availability(store,rid)['status']=='FULL' and
               r['category']!='fixture' and r['candidate_id'] not in {'toy_normal_ae','toy_normal_ae_cuda'} and
               not json.loads(r['config_json']).get('frozen_baseline') and
               json.loads(r['config_json']).get('execution_kind','train')=='train' and
               (Path(r['artifact_dir'])/'checkpoint/model.pt').is_file()}
    ranked=[];excluded=[]
    # Distinct trained runs, not repeated evaluations of a frozen checkpoint.
    for model in models.values():
        if model['run_id'] not in producing:continue
        try:
            registry.retained_category(store,model)
            if runs[model['run_id']]['status'] not in {'COMPLETED','REVIEWED','PROMOTED','REJECTED'}:continue
            ranked.append((registry.research_score(model)['score'],model))
        except (OSError,ValueError,KeyError) as exc:
            excluded.append({'run_id':model['run_id'],'reason':str(exc)})
    ranked.sort(key=lambda x:(-x[0],x[1]['created_at'],x[1]['run_id']))
    chosen=[]
    heads=registry.research_states(store)
    for head in heads.values():
        model=models.get(head['retention_id'])
        if not model:raise ValueError('incumbent has no provenance archive')
        registry.verify_archive(model)
        rid=model.get('learning_run_id',model['run_id'])
        if rid not in producing:raise ValueError('incumbent trained source unavailable')
        if rid not in chosen:chosen.append(rid)
    for _,model in ranked:
        if len(chosen)>=6:break
        if model['run_id'] not in chosen:chosen.append(model['run_id'])
    recent=sorted(producing,key=lambda rid:(-(runs[rid]['finished_at'] or runs[rid]['created_at']),rid))
    for rid in recent:
        if len(chosen)>=9:break
        if rid not in chosen:chosen.append(rid)
    # Completed studies in a long-running campaign are eligible for retirement;
    # campaign membership alone is not an active scientific dependency.
    protected={rid for rid,r in runs.items() if r['status'] in {'RUNNING','QUEUED'}}
    with store.connect() as con:
        active={r[0] for r in con.execute("SELECT id FROM campaigns WHERE status IN ('RUNNING','PAUSED','STOPPING')")}
        events=list(con.execute("SELECT campaign_id,kind,payload_json FROM campaign_events WHERE kind IN ('EXPERIMENT_PLAN_REGISTERED','EXPERIMENT_PLAN_RESULT') ORDER BY id"))
    completed=set()
    for row in events:
        value=json.loads(row['payload_json']);value=value.get('result',value)
        if row['kind']=='EXPERIMENT_PLAN_RESULT' and value.get('status','').startswith('COMPLETE'):
            completed.add((row['campaign_id'],value.get('plan_id')))
    for row in events:
        if row['campaign_id'] not in active or row['kind']!='EXPERIMENT_PLAN_REGISTERED':continue
        value=json.loads(row['payload_json']);plan=value.get('plan',value)
        if (row['campaign_id'],plan.get('plan_id')) in completed:continue
        protected.update(_refs(value))
        for rid,run in runs.items():
            cfg=json.loads(run['config_json'])
            if cfg.get('campaign_id')==row['campaign_id'] and cfg.get('plan_id')==plan.get('plan_id'):protected.add(rid)
    changed=True
    while changed:
        before=set(protected)
        for rid in list(protected):
            if rid in runs:protected.update(_refs(json.loads(runs[rid]['config_json'])))
        changed=before!=protected
    conflicts=[]
    if len(chosen)>9:conflicts.append({'reason':'more than nine independently pinned incumbents; no mandatory reference deleted','run_ids':chosen})
    for head in registry.states(store).values():
        if head.get('confirmation')=='REPEATED_SEED_SUPPORTED':
            model=models[head['retention_id']];protected.add(model.get('learning_run_id',model['run_id']));protected.add(model['run_id'])
    # Confirmation survives later promotions and rollback; an independently
    # confirmed historical checkpoint is a reported exception to nine slots.
    for record in registry.records(store,'CHAMPION_DECISION'):
        decision=record['payload']
        if decision.get('status')=='PROMOTED' and decision.get('challenger') in models:
            model=models[decision['challenger']]
            protected.update((model['run_id'],model.get('learning_run_id',model['run_id'])))
    # A frozen readout head may have its own verified source/checkpoint archive.
    for head in heads.values():
        model=models[head['retention_id']]
        if model['run_id'] not in chosen:protected.add(model['run_id'])
    extra=protected-set(chosen)
    if extra:conflicts.append({'reason':'active/recovery or confirmed scientific protection','run_ids':sorted(extra)})
    return {'version':'rolling-nine-v1','selected':chosen,'protected':sorted(protected),
            'conflicts':conflicts,'excluded':excluded,'scores':{m['run_id']:s for s,m in ranked},
            'runs':runs,'models':models}


def _bulky(path):
    return path.suffix.lower() in BULKY or path.name.endswith(('.npy.gz','.npy.bs4.gz','.npy.ref.json'))


def preview(store,runtime):
    runtime=Path(runtime).resolve();plan=selection(store)
    keep=set(plan['selected'])|set(plan['protected']);jobs=runtime/'jobs'
    # Retained views can refer to another run's maps or a compressed-map chain.
    # Protect these canonical images without retaining that run's model bytes.
    visual_roots=set()
    with store.connect() as con:
        feedback=list(con.execute("SELECT run_id,payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK'"))
    for row in feedback:
        if row['run_id'] in keep:
            summary=json.loads(row['payload_json'])
            visual_roots.add(_safe(Path(summary.get('prediction_artifact_dir',summary['artifact_dir'])),jobs))
    canonical=set()
    for root in visual_roots:
        for ref in root.glob('case_*/map.npy.ref.json'):
            _safe(ref,jobs);value=json.loads(ref.read_text());target=_safe(jobs/value['target'],jobs)
            seen=set()
            while Path(str(target)+'.ref.json').exists():
                if target in seen:raise ValueError('cyclic canonical map reference')
                seen.add(target);canonical.add(target)
                linked=_safe(Path(str(target)+'.ref.json'),jobs)
                target=_safe(jobs/json.loads(linked.read_text())['target'],jobs)
            canonical.add(target)
    candidates=[];blocked=[]
    for rid,row in plan['runs'].items():
        if rid in keep or row['status'] not in TERMINAL:continue
        root=jobs/rid
        if Path(row['artifact_dir'])!=root:continue
        roots=[root]
        roots.extend(Path(m['archive']) for m in plan['models'].values() if m['run_id']==rid)
        # MLflow local artifacts only, located through the exact ledger run id.
        mid=row.get('mlflow_run_id')
        if mid and all(c in '0123456789abcdef' for c in mid):
            roots.extend((runtime/'mlflow/artifacts').glob('*/'+mid+'/artifacts'))
        files=[]
        try:
            for base in roots:
                allowed=runtime/'controller/retained_models' if base.parent==runtime/'controller/retained_models' else (jobs if base==root else runtime/'mlflow/artifacts')
                _safe(base,allowed)
                for path in base.rglob('*'):
                    _safe(path,allowed)
                    if not path.is_file():continue
                    compressed_text=path.suffix in {'.json','.log','.jsonl'} and path.name not in {'scientific_record.json','artifact_availability.json'} and path.stat().st_size>2**21
                    if not (_bulky(path) or compressed_text):continue
                    if any(path.is_relative_to(v) for v in visual_roots):continue
                    raw=Path(str(path).removesuffix('.ref.json').removesuffix('.bs4.gz').removesuffix('.gz'))
                    if path in canonical or raw in canonical:continue
                    files.append({'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'compress':compressed_text})
            # Ledger holds config/metrics/hypothesis; job JSON/source/logs remain.
            # Require an actual provenance manifest before retiring model bytes.
            if files and not (root/'run_manifest.json').is_file():
                blocked.append({'run_id':rid,'reason':'missing run manifest; ambiguous evidence preserved'});continue
            if files:candidates.append({'run_id':rid,'files':files})
        except (ValueError,OSError) as exc:blocked.append({'run_id':rid,'reason':str(exc)})
    return {k:v for k,v in plan.items() if k not in {'runs','models'}}|{'candidates':candidates,'blocked':blocked,
            'estimated_bytes':sum(f['bytes'] for r in candidates for f in r['files'])}


def retire(store,runtime,*,max_seconds=30):
    """Caller holds production storage/GPU/feedback locks. Resume manifests first."""
    from orchestrator.storage_cleanup import _atomic_json,_event,_sync_dir
    runtime=Path(runtime).resolve();plan=preview(store,runtime)
    report={'actions':0,'reclaimed_bytes':0,'errors':[],'selected':plan['selected'],'conflicts':plan['conflicts'],'blocked':plan['blocked']}
    start=time.monotonic()
    for item in plan['candidates']:
        if time.monotonic()-start>=max_seconds:break
        rid=item['run_id'];root=runtime/'jobs'/rid
        try:
            state=availability(store,rid)
            # Durable write-ahead retirement: viewers stop claiming availability
            # before any unlink. Original hashes/provenance are never rewritten.
            if state['status']=='FULL':
                state={'status':'ARCHIVED_METRICS_ONLY','checkpoint_available':False,
                       'reason':'outside rolling nine; historical hashes retained, frozen replication unavailable',
                       'version':plan['version'],'prepared_at':time.time()}
                with store.connect() as con:
                    con.execute('INSERT OR REPLACE INTO settings(name,value) VALUES(?,?)',(PREFIX+rid,json.dumps(state)))
                    con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                                (rid,time.time(),'ARTIFACTS_RETIREMENT_PREPARED',json.dumps(item)))
            _atomic_json(root/'artifact_availability.json',state)
            # Persist and verify complete lightweight ledger snapshot as well.
            row=store.get_run(rid)
            snapshot=dict(row)
            snapshot['retired_artifact_provenance']=[{'path':e['path'],'bytes':e['bytes']} for e in item['files']]
            with store.connect() as con:
                snapshot['events']=[dict(r) for r in con.execute("SELECT * FROM events WHERE run_id=? AND kind NOT LIKE 'STORAGE_%' AND kind NOT LIKE 'ARTIFACTS_%'",(rid,))]
            _atomic_json(root/'scientific_record.json',snapshot)
            for entry in item['files']:
                path=_safe(Path(entry['path']),runtime)
                before=path.stat()
                if before.st_size!=entry['bytes'] or before.st_mtime_ns!=entry['mtime_ns']:raise ValueError('artifact changed; preserve and retry')
                sha=sha256_file(path)
                replacement_bytes=0
                if entry.get('compress'):
                    packed=Path(str(path)+'.gz');_safe(packed,runtime)
                    temporary=path.with_name('.retention-pack-'+uuid.uuid4().hex)
                    try:
                        with temporary.open('xb') as raw:
                            with path.open('rb') as source, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=1) as stream:
                                shutil.copyfileobj(source,stream,length=2**20)
                            raw.flush();os.fsync(raw.fileno())
                        if _restored_hash(temporary,before.st_size)!=sha:raise ValueError('text restoration mismatch')
                        if packed.exists():
                            if _restored_hash(packed,before.st_size)!=sha:raise ValueError('conflicting compressed evidence')
                        else:os.replace(temporary,packed);_sync_dir(path.parent)
                        replacement_bytes=packed.stat().st_size
                    finally:temporary.unlink(missing_ok=True)
                _event(store,'ARTIFACT_RETIRE_PREPARED',run_id=rid,path=str(path),sha256=sha,bytes=before.st_size,replacement=str(path)+'.gz' if entry.get('compress') else None)
                after=path.stat()
                if (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns):raise ValueError('artifact changed during hashing')
                path.unlink();_sync_dir(path.parent)
                _event(store,'ARTIFACT_RETIRED',run_id=rid,path=str(path),sha256=sha,bytes=before.st_size,replacement=str(path)+'.gz' if entry.get('compress') else None)
                report['actions']+=1;report['reclaimed_bytes']+=before.st_size-replacement_bytes
            from orchestrator.mlflow_tracker import enqueue_run_tracking
            enqueue_run_tracking(store,rid,event_key_suffix='rolling-nine-v1')
        except (OSError,ValueError,KeyError) as exc:report['errors'].append({'run_id':rid,'error':str(exc)})
    _event(store,'ROLLING_RETENTION_PASS',**report)
    return report


def _restored_hash(path, expected_size):
    digest=hashlib.sha256();total=0
    with gzip.open(path,'rb') as stream:
        while True:
            block=stream.read(min(2**20,expected_size-total+1))
            if not block:break
            total+=len(block)
            if total>expected_size:raise ValueError('oversized compressed scientific evidence')
            digest.update(block)
    if total!=expected_size:raise ValueError('truncated compressed scientific evidence')
    return digest.hexdigest()


def read_text(path, *, max_bytes=2**21):
    """Read bounded scientific text, including losslessly archived evidence."""
    path=Path(path)
    if path.is_symlink():raise ValueError('linked scientific evidence')
    if path.exists():
        if path.stat().st_size>max_bytes:return None
        return path.read_text()
    packed=Path(str(path)+'.gz')
    if not packed.exists():return None
    if packed.is_symlink():raise ValueError('linked compressed evidence')
    with gzip.open(packed,'rb') as stream:data=stream.read(max_bytes+1)
    if len(data)>max_bytes:return None
    return data.decode()
