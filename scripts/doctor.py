"""Read-only environment diagnostics. No implicit installation or GPU workloads."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from orchestrator.state_store import default_db_path


def diagnose():
    status = {
        "codex_cli": shutil.which("codex"),
        "git": shutil.which("git"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "mlflow_installed": bool(importlib.util.find_spec("mlflow")),
        "streamlit_installed": bool(importlib.util.find_spec("streamlit")),
        "torch_installed": bool(importlib.util.find_spec("torch")),
        "ledger_path": str(default_db_path()),
        "runtime_root": str(Path(os.getenv("LOCO_RUNTIME", ".runtime")).resolve()),
        "sealed_evaluator_available_to_development": False, # MUST separately verify as actual OS user
        "os_sandbox_operator_asserted": os.getenv("LOCO_SANDBOX_VERIFIED") == "1",
    }
    if status["torch_installed"]:
        try:
            import torch
            status["torch_cuda_available"] = torch.cuda.is_available()
            status["torch_version"] = torch.__version__
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            status["torch_error"] = str(exc)
    if status["nvidia_smi"]:
        done = subprocess.run([status["nvidia_smi"], "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        status["nvidia_smi_output"] = done.stdout.strip() if done.returncode == 0 else done.stderr.strip()
    return status

if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2, default=str))
