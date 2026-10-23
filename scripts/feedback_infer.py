"""Single-image, CPU-only inference worker. Masks and other cases are not mounted."""
from __future__ import annotations
import time
_IMPORT_STARTED = time.perf_counter()
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from PIL import Image
from benchmark.train_adapter import load_candidate, validate_prediction
_IMPORT_SECONDS = time.perf_counter()-_IMPORT_STARTED


def state_digest(predictor):
    """Include wrapper tensors (e.g. fixed normalization), modules and arrays."""
    digest = hashlib.sha256(); seen = set()
    def visit(value):
        # Large JSON checkpoint audits (indexed training streams) are predictor
        # state too. Hash all their contents using the C JSON encoder rather
        # than millions of individual Python/hash.update calls. Tensor/object
        # subtrees still take the strict inspectable-state path below.
        if isinstance(value, (dict,list,tuple)):
            try:
                encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
            except (TypeError,ValueError):
                pass
            else:
                digest.update(b'json:');digest.update(encoded.encode());return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.nn.Module):
            for key, tensor in sorted(value.state_dict().items()):
                digest.update(key.encode()); visit(tensor)
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype,value.shape)).encode()); digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value,key=str):
                digest.update(str(key).encode()); visit(value[key])
        elif isinstance(value, (list,tuple)):
            for part in value: visit(part)
        elif isinstance(value,(str,int,float,bool,type(None),torch.device)):
            digest.update(repr(value).encode())
        elif hasattr(value,'__dict__'):
            visit(vars(value))
        else:
            raise ValueError(f'uninspectable inference state: {type(value).__name__}')
    visit(predictor)
    return digest.hexdigest()


def modules_of(predictor):
    if isinstance(predictor,torch.nn.Module):
        return [predictor]
    return [v for v in vars(predictor).values() if isinstance(v,torch.nn.Module)]


def infer_one(module, checkpoint, image_path, size, output, inference_config=None):
    torch.set_num_threads(2)
    def prohibited(*args,**kwargs):
        raise RuntimeError('optimizer steps are prohibited in real-development inference')
    for cls in vars(torch.optim).values():
        if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer):
            cls.step = prohibited
    checkpoint = Path(checkpoint)
    profile={}
    phase=time.perf_counter()
    frozen = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.parent.iterdir()
              if p.name in {'model.pt','checkpoint_manifest.json'}}
    profile['checkpoint_hash_seconds']=time.perf_counter()-phase
    from benchmark.train_adapter import load_predictor
    phase=time.perf_counter()
    predictor = load_predictor(load_candidate(module), checkpoint, 'cpu', inference_config)
    profile['model_import_and_load_seconds']=time.perf_counter()-phase
    modules = modules_of(predictor)
    if not modules:
        raise ValueError('audited inference requires an exposed torch Module on the predictor')
    parameters={id(p):p for net in modules for p in net.parameters()}
    trainable_parameters=sum(p.numel() for p in parameters.values() if p.requires_grad)
    for net in modules:
        net.eval(); net.requires_grad_(False)
    phase=time.perf_counter()
    before = state_digest(predictor)
    profile['initial_state_hash_seconds']=time.perf_counter()-phase
    started = time.perf_counter()
    with torch.inference_mode():
        with Image.open(image_path) as image:
            original_size = image.size
            x = np.asarray(image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
        profile['decode_resize_seconds']=time.perf_counter()-started
        phase=time.perf_counter()
        x = x.transpose(2,0,1)[None]
        maps,_ = validate_prediction(predictor.predict(x),(1,3,size,size))
        profile['prediction_seconds']=time.perf_counter()-phase
        phase=time.perf_counter()
        score = float(maps.max())  # matches normal-only calibration's map maximum
        full = np.asarray(Image.fromarray(maps[0,0].astype(np.float32)).resize(original_size,Image.Resampling.BILINEAR))
    profile['full_resolution_postprocess_seconds']=time.perf_counter()-phase
    elapsed = time.perf_counter()-started
    phase=time.perf_counter()
    after = state_digest(predictor)
    if before != after:
        raise ValueError('model parameters, buffers or wrapper state changed during evaluation')
    for name,sha in frozen.items():
        if hashlib.sha256((checkpoint.parent/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('checkpoint changed during evaluation')
    profile['final_integrity_seconds']=time.perf_counter()-phase
    output=Path(output)
    phase=time.perf_counter()
    np.save(output/'map.npy',full.astype(np.float32),allow_pickle=False)
    profile['map_serialization_seconds']=time.perf_counter()-phase
    profile['worker_import_seconds']=_IMPORT_SECONDS
    parameters={id(p):p.numel() for net in modules for p in net.parameters()}
    (output/'inference.json').write_text(json.dumps({'image_score':score,'e2e_ms':elapsed*1000,'execution_profile':profile,
        'parameters':sum(parameters.values()),'trainable_parameters_before_freeze':trainable_parameters,
        'trainable_parameters_during_inference':0,'serialized_model_bytes':(checkpoint.parent/'model.pt').stat().st_size,
        'measurement_protocol':'batch one, fresh process, first inference; image decode/resize/predict/map restore; excludes model load and process startup',
        'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'state_before_sha256':before,'state_after_sha256':after,
        'inference_mode':True,'device':'cpu','width':original_size[0],'height':original_size[1]}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,required=True)
    p.add_argument('--inference-config-json',default='{}')
    a=p.parse_args()
    if not 16<=a.size<=1024: raise ValueError('invalid frozen preprocessing size')
    infer_one('/candidate/model.py','/inputs/checkpoint_manifest.json','/inputs/image.png',a.size,'/output',json.loads(a.inference_config_json))
