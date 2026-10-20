"""Original operator limits for regression tests of legacy authorizations."""
from pathlib import Path
import yaml


def legacy_config():
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/autonomy.yaml').read_text())
    config['campaign'].pop('start_from',None)
    config['limits']={'max_gpu_hours':4,'max_wall_hours':12,'max_experiments':8,
        'max_codex_calls':30,'max_job_seconds':1800,'max_consecutive_failures':3,'max_parallel_gpu_jobs':1}
    return config
