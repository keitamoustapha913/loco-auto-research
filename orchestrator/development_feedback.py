"""Trusted bounded inference/scoring stage and durable research feedback."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
from PIL import Image
from benchmark.data_guard import sha256_file, load_manifest, manifest_fingerprint
from benchmark.feedback_data import LABEL, imported_root, verify_import
from benchmark.feedback_metrics import VERSION, LOCALIZATION_BLOCKER, detection_metrics
from orchestrator.experiment_runner import candidate_file
from orchestrator.sandbox import command as sandbox_command

REPO=Path(__file__).resolve().parents[1]


def latest_summary(store,run_id):
    with store.connect() as con:
        row=con.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='REAL_DEVELOPMENT_FEEDBACK' ORDER BY id DESC LIMIT 1",(run_id,)).fetchone()
    return json.loads(row['payload_json']) if row else None


def record_summary(store,run_id,summary):
    """Atomic feedback event and tracking outbox; training measurements untouched."""
    key=f"{run_id}:feedback:{summary['evaluation_id']}"
    with store.connect() as con:
        con.execute('BEGIN IMMEDIATE')
        row=dict(con.execute('SELECT * FROM runs WHERE run_id=?',(run_id,)).fetchone())
        if con.execute('SELECT 1 FROM tracking_outbox WHERE event_key=?',(key,)).fetchone(): return
        con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                    (run_id,time.time(),'REAL_DEVELOPMENT_FEEDBACK',json.dumps(summary,sort_keys=True)))
        payload={k:row[k] for k in ('run_id','status','category','stage','candidate_id','hypothesis_id','git_sha',
                    'manifest_sha256','artifact_dir','created_at','started_at','finished_at','gpu_seconds','error')}
        payload.update(metrics=json.loads(row['metrics_json']),params=json.loads(row['config_json']),feedback=summary)
        con.execute('INSERT INTO tracking_outbox(event_key,run_id,payload_json,created_at) VALUES(?,?,?,?)',
                    (key,run_id,json.dumps(payload,sort_keys=True),time.time()))



def validate_worker_outputs(folder, width, height):
    """Candidate-written artifacts must never redirect trusted reads elsewhere."""
    folder = Path(folder)
    for name, cap in (('map.npy', width * height * 4 + 4096), ('inference.json', 10000)):
        path = folder / name
        if (path.is_symlink() or not path.is_file() or
            not path.resolve(strict=True).is_relative_to(folder.resolve(strict=True)) or
            path.stat().st_size > cap):
            raise ValueError('unsafe or oversized inference artifact')


def evaluate_run(runtime,store,run_id,*,timeout=600,cancel=None,personal_mode_authorized=False):
    if not personal_mode_authorized:
        raise PermissionError('bounded evaluation requires explicit personal Bubblewrap authorization')
    if not 1<=timeout<=900: raise ValueError('CPU evaluation timeout must be within 1..900 seconds')
    runtime=Path(runtime).resolve(strict=True)
    lock_path=runtime/'controller/feedback.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _evaluate(runtime,store,run_id,timeout,cancel)


def _evaluate(runtime,store,run_id,timeout,cancel):
    started=time.monotonic()
    run=store.get_run(run_id)
    if not run or run['status'] not in {'COMPLETED','REVIEWED','PROMOTED','REJECTED'}:
        raise ValueError('feedback requires an existing completed frozen run')
    data_root=imported_root(runtime); manifest=verify_import(data_root)
    job=runtime/'jobs'/run_id
    if job.is_symlink() or not job.resolve(strict=True).is_relative_to(runtime/'jobs'):
        raise ValueError('job path escapes runtime')
    module=candidate_file(REPO,run['candidate_id'])
    provenance=json.loads((job/'run_manifest.json').read_text())
    checkpoint=job/'checkpoint/model.pt'; checkpoint_meta=job/'checkpoint/checkpoint_manifest.json'
    good,_=load_manifest(runtime/'good_manifest.json')
    if sum(e['split']=='train/good' and e['category']=='juice_bottle' for e in good['images']) != 335:
        raise ValueError('expected 335 authorized normal training images')
    if (provenance['model_module_sha256']!=sha256_file(module) or
        provenance['checkpoint_sha256']!=sha256_file(checkpoint) or
        provenance['good_manifest_sha256']!=manifest_fingerprint(good) or
        run['manifest_sha256']!=manifest_fingerprint(good)):
        raise ValueError('source, checkpoint or normal-learning provenance changed')
    threshold=json.loads(run['metrics_json']).get('normal/threshold')
    if threshold is None: raise ValueError('separate normal-only calibrated threshold is required')
    before={str(p):sha256_file(p) for p in (module,checkpoint,checkpoint_meta,job/'run_manifest.json')}
    import hashlib
    eval_id=hashlib.sha256((run_id+manifest['split_id']+before[str(checkpoint)]+VERSION).encode()).hexdigest()[:24]
    old=latest_summary(store,run_id)
    if old and old['evaluation_id']==eval_id: return old
    output=job/'feedback'/eval_id
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError('feedback output must not use symlinks')
    if output.exists():
        # A process crash cannot silently repeat a possibly costly inference stage.
        if (output/'summary.json').is_file():
            summary=json.loads((output/'summary.json').read_text()); record_summary(store,run_id,summary); return summary
        raise RuntimeError(f'interrupted feedback attempt retained at {output}; explicit operator review required')
    output.mkdir(parents=True)
    empty=output/'empty_data'; empty.mkdir()
    rows=[]
    try:
        for index,entry in enumerate(manifest['images']):
            if cancel and cancel(): raise RuntimeError('feedback cancelled by campaign stop')
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0: raise TimeoutError('bounded CPU feedback wall limit reached')
            folder=output/f'case_{index:04d}'; folder.mkdir()
            command=sandbox_command(REPO,module,empty,folder,
                ['-m','scripts.feedback_infer','--size',str(provenance['size'])],use_gpu=False,
                input_files={'model.pt':checkpoint,'checkpoint_manifest.json':checkpoint_meta,
                             'image.png':data_root/entry['path']})
            with (folder/'worker.log').open('wb') as log:
                process=subprocess.Popen(command,cwd=REPO,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
                                         stdout=log,stderr=log,start_new_session=True)
                deadline=time.monotonic()+min(30,remaining)
                try:
                    while process.poll() is None:
                        if time.monotonic()>=deadline: raise TimeoutError('single-image inference deadline reached')
                        if cancel and cancel(): raise RuntimeError('feedback cancelled')
                        time.sleep(.05)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try: process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL); process.wait()
            if process.returncode:
                raise RuntimeError(f'inference worker failed on opaque case {index}, exit {process.returncode}; inspect worker log')
            validate_worker_outputs(folder, entry['width'], entry['height'])
            result=json.loads((folder/'inference.json').read_text())
            heatmap=np.load(folder/'map.npy',allow_pickle=False)
            if heatmap.shape!=(entry['height'],entry['width']) or not np.isfinite(heatmap).all():
                raise ValueError('invalid original-resolution heatmap')
            rows.append(dict(result,category=entry['category'],source_sha256=entry['sha256'],
                             mask_count=len(entry['masks']),case_index=index,map_sha256=sha256_file(folder/'map.npy')))
            if index in (0,47,118):
                lo,hi=float(heatmap.min()),float(heatmap.max())
                scaled=np.clip((heatmap-lo)/max(hi-lo,1e-12),0,1)
                rgb=np.stack([scaled*255,scaled*60,(1-scaled)*100],axis=-1).astype(np.uint8)
                Image.fromarray(rgb).save(output/f'heatmap_{index}.png')
        if any(sha256_file(Path(path))!=value for path,value in before.items()):
            raise ValueError('immutable candidate/checkpoint/provenance changed')
        metrics=detection_metrics(rows,threshold)
        parameters={r['parameters'] for r in rows}
        if len(parameters)!=1: raise ValueError('model capacity changed between images')
        metrics.update({'dev_real/inference_e2e_ms_p95':float(np.percentile([r['e2e_ms'] for r in rows],95)),
                        'dev_real/parameters':parameters.pop(),
                        'dev_real/peak_cpu_rss_mib':max(r['process_peak_rss_mib'] for r in rows),
                        'dev_real/evaluation_wall_seconds':time.monotonic()-started,
                        'dev_real/gpu_seconds':0})
        summary={'evaluation_id':eval_id,'label':LABEL,'split_id':manifest['split_id'],'evaluator_version':VERSION,
                 'checkpoint_sha256':before[str(checkpoint)],'source_sha256':before[str(module)],
                 'training_manifest_sha256':run['manifest_sha256'],'preprocessing_size':provenance['size'],
                 'threshold_source':'originating run normal-only validation calibration; maximum model-space map score',
                 'learning_policy':'335 train/good only; no development anomaly fitting, normalization or calibration',
                 'inference_isolation':'one opaque image per fresh no-network CPU Bubblewrap process; no masks, no peer images; shared host UID residual risk',
                 'checkpoint_unchanged':True,'metrics':metrics,'counts':manifest['summary'],
                 'localization_status':'unavailable','localization_blocker':LOCALIZATION_BLOCKER,
                 'artifact_dir':str(output),'status':'DETECTION_COMPLETE_LOCALIZATION_BLOCKED'}
        (output/'case_scores.json').write_text(json.dumps(rows,sort_keys=True,indent=2))
        (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
        record_summary(store,run_id,summary)
        return summary
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'elapsed_seconds':time.monotonic()-started}))
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (run_id,time.time(),'REAL_FEEDBACK_FAILED',json.dumps({'error':str(exc),'artifact_dir':str(output)})))
        raise
