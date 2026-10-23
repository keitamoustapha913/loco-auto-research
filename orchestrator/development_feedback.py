"""Trusted bounded inference/scoring stage and durable research feedback."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import shutil
import threading
import numpy as np
from PIL import Image
from benchmark.data_guard import sha256_file, load_manifest, manifest_fingerprint
from benchmark.feedback_data import LABEL, imported_root, verify_import
from benchmark.feedback_metrics import VERSION, LOCALIZATION_STATUS, detection_metrics
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



def run_bounded(command, log_path, seconds, cancel=None):
    """Both inference and trusted scoring obey cancellation and a hard deadline."""
    if seconds <= 0: raise TimeoutError('evaluation wall limit reached')
    with Path(log_path).open('wb') as log:
        process = subprocess.Popen(command,cwd=REPO,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
                                   stdout=log,stderr=log,start_new_session=True)
        deadline=time.monotonic()+seconds
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline: raise TimeoutError('bounded evaluation stage deadline reached')
                if cancel and cancel(): raise RuntimeError('feedback cancelled')
                time.sleep(.05)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        if process.returncode:
            raise RuntimeError(f'evaluation stage failed with exit {process.returncode}; inspect {log_path.name}')



def cpu_worker_count():
    """Internal bounded concurrency; each worker still sees exactly one image.

    Reserve 2 GiB for trusted scoring/controller and allow 2 GiB per fresh worker.
    This is scheduling, not a substitute for OS memory isolation.
    """
    import psutil
    cpus = len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    memory = max(1, int((psutil.virtual_memory().available-2*2**30)//(2*2**30)))
    return max(1, min(4, cpus//2, memory))


def feedback_pipeline_identity():
    """Cost observations are comparable only for the same verified CPU path."""
    return {'version':'isolated-cpu-feedback-v1','cpu_workers':cpu_worker_count(),
            'worker_sha256':sha256_file(REPO/'scripts/feedback_infer.py'),
            'scorer_sha256':sha256_file(REPO/'benchmark/feedback_metrics.py')}


def run_isolated_cases(items, worker, workers=None):
    """Bound scheduling only; worker owns per-image sandbox and cancellation."""
    with ThreadPoolExecutor(max_workers=cpu_worker_count() if workers is None else workers) as pool:
        return list(pool.map(worker, items))


def feedback_identity(manifest, run, config, before, size, threshold, workers=None):
    # Numerical evaluator version is unchanged. Implementation hashes invalidate
    # cache after any trusted inference/scoring change. No final-holdout identity.
    identity = {'version':'frozen-feedback-cache-v1','manifest':manifest,
        'checkpoint_sha256':before['checkpoint'],'checkpoint_manifest_sha256':before['metadata'],
        'source_sha256':before['source'],'training_manifest_sha256':run['manifest_sha256'],
        'inference_configuration':config.get('inference',{}),'preprocessing_size':size,
        'normal_only_threshold':threshold,'device':'cpu','cpu_workers':cpu_worker_count() if workers is None else workers,'evaluator_version':VERSION,
        'worker_sha256':sha256_file(REPO/'scripts/feedback_infer.py'),
        'scorer_sha256':sha256_file(REPO/'benchmark/feedback_metrics.py')}
    return hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_feedback_cache(summary, key, job_root):
    """Require exact immutable identities and safe, content-verified predictions."""
    if summary.get('cache_identity') != key or summary.get('checkpoint_unchanged') is not True:
        raise ValueError('feedback cache identity mismatch')
    root = Path(summary['artifact_dir'])
    if root.is_symlink() or not root.resolve().is_relative_to(Path(job_root).resolve()):
        raise ValueError('feedback cache escapes jobs')
    for name, digest in summary['cache_artifact_hashes'].items():
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256_file(path)!=digest:
            raise ValueError('feedback cache artifact integrity mismatch')
    rows=json.loads((root/'case_scores.json').read_text())
    if len(rows)!=summary['counts']['images']:raise ValueError('cached feedback coverage mismatch')
    for row in rows:
        path=root/f"case_{row['case_index']:04d}"/'map.npy'
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256_file(path)!=row['map_sha256']:
            raise ValueError('feedback cache heatmap integrity mismatch')
    return root

def reuse_feedback_artifacts(summary, cache_key, job_root, output, config, run, eval_id):
    """Copy only verified frozen predictions; never call candidate/training code."""
    origin=verify_feedback_cache(summary,cache_key,job_root)
    # Candidate outputs must not introduce links into the trusted copy.
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in origin.rglob('*')):raise ValueError('unsafe linked feedback artifact')
    shutil.copytree(origin,output)
    summary=dict(summary,evaluation_id=eval_id,artifact_dir=str(output),reused_from_evaluation_id=summary['evaluation_id'])
    for key in ('plan_id','plan_sha256','plan_arm','seed','plan_pair_seed','inference_configuration','trusted_source_sha256'):
        summary.pop(key,None)
    if config.get('plan_sha256'):
        summary.update({key:config[key] for key in ('plan_id','plan_sha256','plan_arm','seed','trusted_source_sha256')})
        summary.update(plan_pair_seed=config.get('plan_pair_seed',config['seed']),inference_configuration=config.get('inference',{}))
    summary['metrics']=dict(summary['metrics'],**{'dev_real/training_gpu_slot_seconds':run['gpu_seconds']})
    (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
    return summary


def evaluate_run(runtime,store,run_id,*,timeout=600,cancel=None,personal_mode_authorized=False,deadline=None):
    if not personal_mode_authorized:
        raise PermissionError('bounded evaluation requires explicit personal Bubblewrap authorization')
    run=store.get_run(run_id)
    configuration=json.loads(run['config_json']) if run else {}
    timed=False
    if configuration.get('campaign_id'):
        from orchestrator.campaign_state import CampaignState
        from orchestrator.campaign_config import operation_seconds,operation_cancelled,time_based
        campaigns=CampaignState(store);row=campaigns.get(configuration['campaign_id'])
        if row and time_based(json.loads(row['config_json'])):
            timed=True
            timeout=operation_seconds(row,timeout)
            original_cancel=cancel
            cancel=lambda:operation_cancelled(campaigns,row['id']) or bool(original_cancel and original_cancel())
    if deadline is not None:timeout=min(timeout,deadline-time.time())
    if timeout<=0:raise TimeoutError('campaign research deadline reached before feedback')
    if not timed and not 1<=timeout<=900:
        raise ValueError('CPU evaluation timeout must be within 1..900 seconds')
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
    from orchestrator.experiment_plan import execution_source
    module=execution_source(run, store, runtime, REPO)
    config=json.loads(run['config_json'])
    from orchestrator.campaign_state import CampaignState
    from orchestrator.campaign_config import time_based
    campaign=CampaignState(store).get(config['campaign_id']) if config.get('campaign_id') else None
    timed=bool(campaign and time_based(json.loads(campaign['config_json'])))
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
    def check_frozen_inputs():
        if any(sha256_file(Path(path))!=value for path,value in before.items()):
            raise ValueError('immutable candidate/checkpoint/provenance changed')
    import hashlib
    workers=cpu_worker_count()
    cache_key=feedback_identity(manifest,run,config,{'source':before[str(module)],'checkpoint':before[str(checkpoint)],'metadata':before[str(checkpoint_meta)]},provenance['size'],threshold,workers=workers)
    eval_id=hashlib.sha256((run_id+cache_key).encode()).hexdigest()[:24]
    old=latest_summary(store,run_id)
    if old and old['evaluation_id']==eval_id:
        verify_feedback_cache(old,cache_key,runtime/'jobs')
        check_frozen_inputs()
        return old
    output=job/'feedback'/eval_id
    if output.is_symlink() or output.parent.is_symlink():
        raise ValueError('feedback output must not use symlinks')
    if output.exists():
        # A process crash cannot silently repeat a possibly costly inference stage.
        if (output/'summary.json').is_file():
            summary=json.loads((output/'summary.json').read_text())
            verify_feedback_cache(summary,cache_key,runtime/'jobs')
            check_frozen_inputs()
            record_summary(store,run_id,summary); return summary
        raise RuntimeError(f'interrupted feedback attempt retained at {output}; explicit operator review required')
    # Reuse exact, integrity-verified frozen readouts across runs. Copy artifacts
    # into the new run's own directory; no shared candidate-writable output.
    with store.connect() as con:
        cached=con.execute("SELECT payload_json FROM events WHERE kind='REAL_DEVELOPMENT_FEEDBACK' AND json_extract(payload_json,'$.cache_identity')=? ORDER BY id DESC LIMIT 1",(cache_key,)).fetchone()
    if cached:
        summary=reuse_feedback_artifacts(json.loads(cached[0]),cache_key,runtime/'jobs',output,config,run,eval_id)
        check_frozen_inputs()
        record_summary(store,run_id,summary)
        return summary
    output.mkdir(parents=True)
    empty=output/'empty_data'; empty.mkdir()
    rows=[]
    aborted=threading.Event()
    try:
        def infer_case_inner(pair):
            if aborted.is_set():raise RuntimeError('peer feedback worker failed')
            index,entry=pair
            if cancel and cancel(): raise RuntimeError('feedback cancelled by campaign stop')
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0: raise TimeoutError('bounded CPU feedback wall limit reached')
            folder=output/f'case_{index:04d}'; folder.mkdir()
            case_started=time.monotonic()
            command=sandbox_command(REPO,module,empty,folder,
                ['-m','scripts.feedback_infer','--size',str(provenance['size']),
                 '--inference-config-json',json.dumps(config.get('inference', {}))],use_gpu=False,
                input_files={'model.pt':checkpoint,'checkpoint_manifest.json':checkpoint_meta,
                             'image.png':data_root/entry['path']}, pinned_source=bool(config.get('frozen_baseline')))
            run_bounded(command, folder/'worker.log', remaining if timed else min(30,remaining), lambda:aborted.is_set() or bool(cancel and cancel()))
            worker_seconds=time.monotonic()-case_started
            validate_worker_outputs(folder, entry['width'], entry['height'])
            result=json.loads((folder/'inference.json').read_text())
            heatmap=np.load(folder/'map.npy',allow_pickle=False)
            if heatmap.shape!=(entry['height'],entry['width']) or not np.isfinite(heatmap).all():
                raise ValueError('invalid original-resolution heatmap')
            row=dict(result,category=entry['category'],source_sha256=entry['sha256'],
                     mask_count=len(entry['masks']),case_index=index,map_sha256=sha256_file(folder/'map.npy'))
            if index in (0,47,118):
                lo,hi=float(heatmap.min()),float(heatmap.max())
                scaled=np.clip((heatmap-lo)/max(hi-lo,1e-12),0,1)
                rgb=np.stack([scaled*255,scaled*60,(1-scaled)*100],axis=-1).astype(np.uint8)
                Image.fromarray(rgb).save(output/f'heatmap_{index}.png')
            row['worker_wall_seconds']=worker_seconds
            return row
        def infer_case(pair):
            try:return infer_case_inner(pair)
            except BaseException:
                aborted.set()
                raise
        inference_started=time.monotonic()
        rows=run_isolated_cases(enumerate(manifest['images']),infer_case,workers=workers)
        inference_wall=time.monotonic()-inference_started
        check_frozen_inputs()
        def check():
            if time.monotonic()-started >= timeout: raise TimeoutError('evaluation wall limit reached')
            if cancel and cancel(): raise RuntimeError('feedback cancelled')
        metrics=detection_metrics(rows,threshold)
        scoring_started=time.monotonic()
        run_bounded([sys.executable,'-m','benchmark.feedback_metrics','--data-root',str(data_root),
                     '--maps',str(output),'--result',str(output/'localization.json')],
                    output/'scorer.log',timeout-(time.monotonic()-started),cancel)
        scoring_wall=time.monotonic()-scoring_started
        metrics.update(json.loads((output/'localization.json').read_text()))
        parameters={r['parameters'] for r in rows}
        if len(parameters)!=1: raise ValueError('model capacity changed between images')
        metrics.update({'dev_real/inference_e2e_ms_p50':float(np.percentile([r['e2e_ms'] for r in rows],50)),
                        'dev_real/trainable_parameters_before_freeze':rows[0]['trainable_parameters_before_freeze'],
                        'dev_real/trainable_parameters_during_inference':0,
                        'dev_real/serialized_model_bytes':checkpoint.stat().st_size,
                        'dev_real/training_gpu_slot_seconds':run['gpu_seconds'],
                        'dev_real/inference_e2e_ms_p95':float(np.percentile([r['e2e_ms'] for r in rows],95)),
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
                 'cache_identity':cache_key, 'execution_profile':{'version':'feedback-timing-v1','cpu_workers':workers,
                    'pipeline_identity':dict(feedback_pipeline_identity(),cpu_workers=workers),
                    'inference_stage_wall_seconds':inference_wall,'trusted_scoring_wall_seconds':scoring_wall,
                    'sum_isolated_worker_wall_seconds':sum(r['worker_wall_seconds'] for r in rows),
                    'sum_prediction_e2e_seconds':sum(r['e2e_ms'] for r in rows)/1000},
                 'checkpoint_unchanged':True,'metrics':metrics,'counts':manifest['summary'],
                 'inference_device':'cpu','measurement_protocol':rows[0]['measurement_protocol']+f'; CPU concurrency={workers}; latency measured under concurrent load',
                 'cuda_inference_memory_status':'unavailable: CPU inference; no CUDA measurement',
                 'localization_status':LOCALIZATION_STATUS,'localization_blocker':'Official download still HTTP 404; compatible scorer uses published definition, not verified original-code equivalence.',
                 'artifact_dir':str(output),'status':'COMPLETE_COMPATIBLE_DEVELOPMENT'}
        if config.get('plan_sha256'):
            summary.update(plan_id=config['plan_id'], plan_sha256=config['plan_sha256'],
                           plan_arm=config['plan_arm'], seed=config['seed'],
                           plan_pair_seed=config.get('plan_pair_seed',config['seed']),
                           inference_configuration=config.get('inference', {}),
                           trusted_source_sha256=config['trusted_source_sha256'])
        (output/'case_scores.json').write_text(json.dumps(rows,sort_keys=True,indent=2))
        summary['cache_artifact_hashes']={name:sha256_file(output/name) for name in ('case_scores.json','localization.json')}
        (output/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2))
        record_summary(store,run_id,summary)
        return summary
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps({'error':str(exc),'elapsed_seconds':time.monotonic()-started}))
        with store.connect() as con:
            con.execute('INSERT INTO events(run_id,time,kind,payload_json) VALUES(?,?,?,?)',
                        (run_id,time.time(),'REAL_FEEDBACK_FAILED',json.dumps({'error':str(exc),'artifact_dir':str(output)})))
        raise
