"""Optional trusted observations for future confirmation runs, never old evidence."""
import hashlib
import json
from pathlib import Path


def tensor_hash(value):
    """Hash actual ordered prepared inputs, including dtype and geometry."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((str(v.dtype),tuple(v.shape))).encode())
            h.update(v.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(v,(list,tuple)):
            h.update(str(type(v)).encode())
            for x in v:visit(x)
        elif isinstance(v,dict):
            for key in sorted(v):h.update(str(key).encode());visit(v[key])
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    visit(value)
    return h.digest()


class PairedControls:
    def __init__(self, requirements, folder):
        self.requirements=requirements;self.folder=Path(folder);self.handles={};self.record={}
        if not requirements:return
        if not isinstance(requirements,list) or len(set(requirements))!=len(requirements):
            raise ValueError('paired_control_requirements must be distinct names')
        for key in requirements:
            if key not in {'initial_state','optimizer'} and not (key.startswith('input:') and key[6:].isdigit()):
                raise ValueError('unsupported trusted paired control: '+str(key))
            if key.startswith('input:'):
                path=self.folder/('paired_input_'+key[6:]+'.sha256')
                self.handles[key]=(path,path.open('xb'))

    def step(self, step):
        if 'initial_state' in self.requirements and 'initial_state' not in self.record:
            self.record['initial_state']=tensor_hash(dict(step.objective.named_parameters()) |
                dict(step.objective.named_buffers())).hex()
        if 'optimizer' in self.requirements and 'optimizer' not in self.record:
            self.record['optimizer']=hashlib.sha256(json.dumps({'class':type(step.optimizer).__module__+'.'+type(step.optimizer).__qualname__,
                'groups':[{k:v for k,v in group.items() if k!='params'} for group in step.optimizer.param_groups]},
                sort_keys=True,default=str).encode()).hexdigest()
        for key,(_,handle) in self.handles.items():
            index=int(key[6:])
            if index>=len(step.args):raise ValueError('paired input index not present in prepared TrainingStep.args: '+key)
            handle.write(tensor_hash(step.args[index]))

    def finish(self):
        if not self.requirements:return {}
        for key,(path,handle) in self.handles.items():
            handle.close()
            self.record[key]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                              'updates':path.stat().st_size//32}
        return {'version':'trusted-prepared-controls-v1','observed':self.record,
                'scope':'actual prepared objective inputs at every update; source/mask identity only where encoded in selected inputs; no claim of OS isolation from hostile Python'}

    def close(self):
        for _,handle in self.handles.values():handle.close()
