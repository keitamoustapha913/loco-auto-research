"""Conservative, reference-aware housekeeping in the existing trusted services.

Numerical evidence and datasets are permanent. Rolling-nine retention retires
explicit bulky artifacts; remaining maps can be packed losslessly.
"""
from __future__ import annotations
import fcntl
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import uuid
import numpy as np
from contextlib import ExitStack

from benchmark.feedback_artifacts import safe_path, map_bytes, map_sha256

INTERVAL = 1800
PASS_BYTES = 1024 * 2**20
PASS_SECONDS = 30
STAGING_AGE = 86400
_next_check = {}


def inventory(runtime):
    """Stat-only quota inventory; no image, annotation or private-data reads."""
    from orchestrator.budgets import enforce_disk, DiskBudgetExceeded
    try:
        quota = enforce_disk(runtime, cleanup=False)
    except DiskBudgetExceeded as exc:
        quota = exc.detail
    return quota


def _event(store, kind, **payload):
    with store.connect() as con:
        con.execute('INSERT INTO events(time,kind,payload_json) VALUES(?,?,?)',
                    (time.time(), kind, json.dumps(payload, sort_keys=True)))


def _refs(value):
    if isinstance(value, dict):
        for key, part in value.items():
            if key == 'run_id' and isinstance(part, str):
                yield part
            yield from _refs(part)
    elif isinstance(value, list):
        for part in value:
            yield from _refs(part)
    elif isinstance(value, str) and Path(value).is_absolute():
        parts=Path(value).parts
        if 'jobs' in parts and len(parts)>parts.index('jobs')+1:
            yield parts[parts.index('jobs')+1]


def protected_runs(store):
    """Protect active campaigns, queued jobs and their frozen historical inputs."""
    with store.connect() as con:
        active = {r[0] for r in con.execute("SELECT id FROM campaigns WHERE status IN ('RUNNING','PAUSED','STOPPING')")}
        runs = list(con.execute('SELECT run_id,status,config_json FROM runs'))
        linked = list(con.execute('SELECT run_id,campaign_id FROM campaign_runs'))
        protected = {r['run_id'] for r in runs if r['status'] in {'RUNNING','QUEUED'}}
        protected.update(r['run_id'] for r in linked if r['campaign_id'] in active)
        for row in con.execute("SELECT campaign_id,payload_json FROM campaign_events WHERE kind='EXPERIMENT_PLAN_REGISTERED'"):
            if row['campaign_id'] in active:
                protected.update(_refs(json.loads(row['payload_json'])))
        feedback=list(con.execute("SELECT run_id,payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK'"))
        previous=None
        while previous!=protected:
            previous=protected.copy()
            for row in runs:
                if row['run_id'] in protected:
                    protected.update(_refs(json.loads(row['config_json'])))
            for row in feedback:
                if row['run_id'] in protected:
                    protected.update(_refs(json.loads(row['payload_json'])))
    return protected


def _maps(store, runtime, protected):
    jobs = runtime/'jobs'
    if jobs.is_symlink():raise ValueError('unsafe linked jobs root')
    if not jobs.exists():
        return []
    entries, seen = [], set()
    with store.connect() as con:
        rows = con.execute("SELECT run_id,payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' ORDER BY id").fetchall()
    for row in rows:
        if row['run_id'] in protected:
            continue
        summary = json.loads(row['payload_json'])
        if summary.get('checkpoint_unchanged') is not True:
            continue
        try:
            folder = safe_path(summary['artifact_dir'], jobs)
            # Read only ledger-declared score metadata, not arbitrary job files.
            scores = safe_path(folder/'case_scores.json', jobs)
            raw = scores.read_bytes()
            expected = summary.get('cache_artifact_hashes', {}).get('case_scores.json')
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                continue
            # Older evaluations predate cache-identity metadata, but retain
            # per-map SHA256s. Preserve their score JSON unchanged and verify
            # every map against those hashes before any lossless substitution.
            from orchestrator.development_feedback import feedback_map_root
            origin = feedback_map_root(summary, jobs)
            if origin.relative_to(jobs).parts[0] in protected:
                continue
            cases = json.loads(raw)
            if len(cases) != summary['counts']['images']:
                continue
            for case in cases:
                index = case['case_index']
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise ValueError('invalid map index')
                path = safe_path(origin/f'case_{index:04d}'/'map.npy', jobs)
                sha = case['map_sha256']
                if not isinstance(sha, str) or len(sha) != 64:
                    raise ValueError('missing map hash')
                if path not in seen:
                    entries.append((path, sha));seen.add(path)
        except (KeyError, ValueError, OSError):
            # Ambiguous or damaged evidence is preserved, never guessed away.
            continue
    return entries


def preview(store, runtime):
    """Read-only candidates; execution still verifies every byte and reference."""
    runtime = Path(runtime).resolve()
    protected = protected_runs(store)
    entries = _maps(store, runtime, protected)
    canonical, duplicates, unique = {}, [], []
    for path, sha in entries:
        if sha in canonical:
            if path.is_file():
                duplicates.append({'path':str(path),'canonical':str(canonical[sha]),'sha256':sha,'bytes':path.stat().st_size})
        else:
            canonical[sha] = path
            if path.is_file():
                unique.append({'path':str(path),'sha256':sha,'bytes':path.stat().st_size})
    return {'inventory':inventory(runtime), 'protected_runs':sorted(protected),
            'duplicates':duplicates, 'lossless_pack_candidates':unique}


def _atomic_json(path, value):
    temp = path.with_name('.storage-json-'+uuid.uuid4().hex)
    try:
        with temp.open('x') as stream:
            json.dump(value, stream, sort_keys=True);stream.flush();os.fsync(stream.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _stat(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _remove(store, path, sha, replacement, reason, *, size, replacement_bytes=0):
    token = uuid.uuid4().hex
    _event(store,'STORAGE_CLEANUP_PREPARED',token=token,path=str(path),sha256=sha,
           replacement=str(replacement),reason=reason,bytes=size,reclaimed_bytes=size-replacement_bytes)
    path.unlink()
    _sync_dir(path.parent)
    _event(store,'STORAGE_CLEANUP_DELETED',token=token,path=str(path),reason=reason,
           bytes=size,reclaimed_bytes=size-replacement_bytes)


def _temps(store, jobs, protected):
    """Only known unpublished staging; any scientific reference preserves it."""
    for candidate in list(jobs.glob('*/feedback/.feedback-reuse-*')) + list(jobs.glob('*/feedback/*/case_*/.storage-*')) + list(jobs.glob('*/**/.retention-pack-*')):
        try:
            safe_path(candidate,jobs)
            if candidate.relative_to(jobs).parts[0] in protected:continue
            files=list(candidate.rglob('*')) if candidate.is_dir() else [candidate]
            if any(p.is_symlink() for p in files):continue
            leaves=[p for p in files if p.is_file()]
            if not leaves or any(time.time()-p.stat().st_mtime<STAGING_AGE for p in leaves):continue
            # Do not remove an unusual unpublished payload merely for its name.
            if candidate.is_dir() and any(p.name not in {'summary.json','case_scores.json','localization.json',
                    'heatmap_0.png','heatmap_47.png','heatmap_118.png'} for p in leaves):continue
            with store.connect() as con:
                mentioned=any(con.execute(f'SELECT 1 FROM {table} WHERE instr({column},?)>0 '+suffix+' LIMIT 1',
                            (str(candidate),)).fetchone() for table,column,suffix in
                    [('events','payload_json',"AND kind NOT LIKE 'STORAGE_%'"),('campaign_events','payload_json',''),
                     ('runs','artifact_dir',''),('runs','config_json',''),('settings','value','')])
            if not mentioned:yield candidate,leaves
        except (ValueError,OSError):continue


def _reconcile(store, jobs):
    """An interrupted unlink never leaves an unaccounted deletion or broken map."""
    with store.connect() as con:
        rows=list(con.execute("SELECT kind,payload_json FROM events WHERE kind IN ('STORAGE_CLEANUP_PREPARED','STORAGE_CLEANUP_DELETED','STORAGE_CLEANUP_ABORTED','STORAGE_TEMP_PREPARED','STORAGE_TEMP_DELETED','STORAGE_TEMP_ABORTED')"))
    done={json.loads(r['payload_json'])['token'] for r in rows if r['kind'].endswith(('DELETED','ABORTED'))}
    for row in rows:
        if row['kind']=='STORAGE_TEMP_PREPARED':
            value=json.loads(row['payload_json'])
            if value['token'] in done:continue
            path=safe_path(value['path'],jobs)
            if not path.exists():
                _event(store,'STORAGE_TEMP_DELETED',**value,recovered_after_restart=True)
            else:
                _event(store,'STORAGE_TEMP_ABORTED',token=value['token'],path=str(path),reason='original staging file remains; retry is a new operation')
            continue
        if row['kind']!='STORAGE_CLEANUP_PREPARED':continue
        value=json.loads(row['payload_json'])
        if value['token'] in done:continue
        path=safe_path(value['path'],jobs)
        if not path.exists() and map_sha256(path,jobs)==value['sha256']:
            _event(store,'STORAGE_CLEANUP_DELETED',**value,recovered_after_restart=True)
        elif path.exists():
            _event(store,'STORAGE_CLEANUP_ABORTED',token=value['token'],path=str(path),reason='original map remains; retry is a new operation')


def cleanup(store, runtime, *, dry_run=False, max_bytes=PASS_BYTES, max_seconds=PASS_SECONDS):
    """One bounded pass. Locks serialize GPU/feedback writers and housekeeping."""
    runtime = Path(runtime).resolve(strict=True)
    if dry_run:return preview(store,runtime)
    controller=runtime/'controller'
    if controller.is_symlink() or (runtime/'jobs').is_symlink():raise ValueError('unsafe linked storage roots')
    controller.mkdir(exist_ok=True)
    from orchestrator.agent_diagnostics import prune
    diagnostic_reclaimed=prune(controller/'agent_invocations')
    report={'reclaimed_bytes':diagnostic_reclaimed,'actions':0,'errors':[],'busy':False}
    with ExitStack() as stack:
        try:
            for name in ('storage_cleanup.lock','gpu.lock','feedback.lock','tracking.lock'):
                if (controller/name).is_symlink():raise ValueError('unsafe linked storage lock')
                handle=stack.enter_context((controller/name).open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            report['busy']=True;return report
        jobs=runtime/'jobs'
        if not jobs.exists():return report
        from orchestrator.rolling_retention import retire
        rolling=retire(store,runtime,max_seconds=max_seconds)
        report['rolling']=rolling
        report['reclaimed_bytes']+=rolling['reclaimed_bytes']
        report['actions']+=rolling['actions']
        report['errors'].extend(rolling['errors'])
        _reconcile(store,jobs)
        protected=protected_runs(store)
        entries=_maps(store,runtime,protected)
        with store.connect() as con:
            saved=con.execute("SELECT value FROM settings WHERE name='storage_cleanup_pack_skips'").fetchone()
        saved_skips=json.loads(saved[0]) if saved else {}
        pack_skips=saved_skips.get('files',{}) if saved_skips.get('encoding')=='byte-shuffled-gzip-v1' else {}
        # Duplicate reclamation precedes lossless packing. Candidate hashes are
        # only an index: actual canonical and duplicate bytes are checked below.
        canonical={};duplicates=[];unique=[]
        for path,sha in entries:
            if sha in canonical:duplicates.append((path,sha,canonical[sha]))
            else:canonical[sha]=path;unique.append((path,sha,None))
        start=time.monotonic();processed=0
        for path,sha,target in duplicates+unique:
            if processed>=max_bytes or time.monotonic()-start>=max_seconds:break
            if not path.is_file():continue
            try:
                before=_stat(path)
                if target is None and pack_skips.get(str(path))==list(before):continue
                data=map_bytes(path,jobs)
                if hashlib.sha256(data).hexdigest()!=sha:
                    raise ValueError('recorded map hash mismatch; preserve original')
                if target:
                    if map_sha256(target,jobs)!=sha:raise ValueError('canonical map hash mismatch')
                    reference=Path(str(path)+'.ref.json')
                    _atomic_json(reference,dict(version=1,target=str(target.relative_to(jobs)),sha256=sha,bytes=len(data)))
                    replacement=reference;extra=reference.stat().st_size;reason='exact duplicate map'
                else:
                    shuffled = len(data)%4 == 0
                    payload = np.frombuffer(data,np.uint8).reshape(-1,4).T.tobytes() if shuffled else data
                    packed=Path(str(path)+('.bs4.gz' if shuffled else '.gz'));temp=path.with_name('.storage-pack-'+uuid.uuid4().hex)
                    try:
                        with temp.open('xb') as raw:
                            with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as stream:stream.write(payload)
                            raw.flush();os.fsync(raw.fileno())
                        with gzip.open(temp,'rb') as stream:
                            restored=stream.read(len(data)+1)
                        if shuffled:restored=np.frombuffer(restored,np.uint8).reshape(4,-1).T.tobytes()
                        if restored!=data:raise ValueError('lossless restoration failed')
                        if temp.stat().st_size>=len(data)*.95:
                            pack_skips[str(path)]=list(before)
                            continue  # insignificant saving; preserve raw map
                        if packed.exists():
                            safe_path(packed,jobs)
                            with gzip.open(packed,'rb') as stream:
                                if stream.read(len(data)+1)!=payload:
                                    raise ValueError('existing packed evidence differs; preserve both files')
                        else:
                            os.replace(temp,packed);_sync_dir(path.parent)
                    finally:temp.unlink(missing_ok=True)
                    replacement=packed;extra=packed.stat().st_size;reason='verified lossless map packing'
                # No candidate or feedback writer runs under these locks. Also
                # refuse an external write between verification and deletion.
                if _stat(path)!=before:raise ValueError('map changed during cleanup; preserve original')
                _remove(store,path,sha,replacement,reason,size=len(data),replacement_bytes=extra)
                report['reclaimed_bytes']+=len(data)-extra;report['actions']+=1
                processed+=len(data)
            except (ValueError,OSError) as exc:
                report['errors'].append({'path':str(path),'error':str(exc)})
        # Only housekeeping-owned staging under terminal, unprotected jobs.
        # Unknown failed outputs, logs, caches and checkpoints remain evidence.
        for candidate,leaves in _temps(store,jobs,protected):
            for path in leaves:
                try:
                    before=_stat(path);data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
                    token=uuid.uuid4().hex
                    _event(store,'STORAGE_TEMP_PREPARED',token=token,path=str(path),sha256=sha,bytes=len(data))
                    if _stat(path)!=before:raise ValueError('staging file changed; preserve it')
                    path.unlink();report['reclaimed_bytes']+=len(data);report['actions']+=1
                    _event(store,'STORAGE_TEMP_DELETED',token=token,path=str(path),sha256=sha,bytes=len(data))
                except (ValueError,OSError) as exc:report['errors'].append({'path':str(path),'error':str(exc)})
            if candidate.is_dir() and not any(p.is_file() for p in candidate.rglob('*')):
                shutil.rmtree(candidate)
        with store.connect() as con:
            con.execute('INSERT OR REPLACE INTO settings(name,value) VALUES(?,?)',('storage_cleanup_last',str(time.time())))
            con.execute('INSERT OR REPLACE INTO settings(name,value) VALUES(?,?)',('storage_cleanup_pack_skips',json.dumps({'encoding':'byte-shuffled-gzip-v1','files':pack_skips},sort_keys=True)))
        _event(store,'STORAGE_CLEANUP_PASS',**report)
    return report


def maybe_cleanup(runtime, store=None, *, force=False, reason='periodic'):
    """Cheap due check; cleanup failure never damages campaign recovery."""
    runtime=Path(runtime).resolve()
    now=time.time();key=str(runtime)
    if not force and now<_next_check.get(key,0):return None
    _next_check[key]=now+60
    ledger=runtime/'controller/experiments.sqlite'
    if store is None:
        if not ledger.is_file():return None
        from orchestrator.state_store import StateStore
        store=StateStore(ledger,read_only=False)
    try:
        with store.connect() as con:
            row=con.execute("SELECT value FROM settings WHERE name='storage_cleanup_last'").fetchone()
        if not force and row and now-float(row[0])<INTERVAL:
            _next_check[key]=float(row[0])+INTERVAL;return None
        result=cleanup(store,runtime)
        result['trigger']=reason
        if not result['busy']:_next_check[key]=now+INTERVAL
        return result
    except Exception as exc:
        try:
            _event(store,'STORAGE_CLEANUP_FAILED',error=f'{type(exc).__name__}: {exc}',trigger=reason)
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Storage cleanup audit unavailable; original evidence preserved where deletion was not committed')
        return {'error':str(exc)}
