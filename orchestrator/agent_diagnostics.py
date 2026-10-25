"""Bounded, redacted invocation evidence; no prompts, credentials or new service."""
import json
import os
from pathlib import Path
import re
import time

TAIL_BYTES = 65536
RETAIN_TRANSCRIPTS = 128


def redact(text):
    text = str(text)
    text = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----',
                  '[REDACTED PRIVATE KEY]', text, flags=re.S)
    text = re.sub(r'(?i)(bearer\s+)[\w.\-]+', r'\1[REDACTED]', text)
    text = re.sub(r'\b(?:sk-|ghp_|github_pat_)[A-Za-z0-9_\-]{8,}', '[REDACTED]', text)
    text = re.sub(r'(?i)((?:access_token|refresh_token|id_token|api_key|authorization|password|secret|token)["\x27]?\s*[:=]\s*["\x27]?)[^\s,"\x27}]+',
                  r'\1[REDACTED]', text)
    return text


def prune(folder):
    """Older transcripts retire; ledger invocation identity/error records remain."""
    folder = Path(folder)
    if folder.is_symlink():raise PermissionError('unsafe agent diagnostics directory')
    if not folder.exists():return 0
    files = [p for p in folder.glob('*.json') if re.fullmatch(r'[a-f0-9]{32}\.json', p.name)
             and not p.is_symlink() and p.is_file()]
    removed = 0
    for p in sorted(files, key=lambda p:p.stat().st_mtime, reverse=True)[RETAIN_TRANSCRIPTS:]:
        # Active calls have no published diagnostic file, so cannot be retired.
        removed += p.stat().st_size
        p.unlink()
    for p in folder.glob('*.tmp'):
        if (re.fullmatch(r'[a-f0-9]{32}\.tmp',p.name) and not p.is_symlink() and p.is_file()
            and time.time()-p.stat().st_mtime>3600):
            # Atomic publication takes milliseconds; no bounded active call can
            # still be writing a one-hour-old final diagnostic temporary file.
            removed+=p.stat().st_size;p.unlink()
    return removed


def save(folder, invocation_id, record):
    if folder is None:return None
    folder = Path(folder)
    if not re.fullmatch('[a-f0-9]{32}', invocation_id):raise ValueError('unsafe invocation ID')
    if folder.is_symlink() or folder.parent.is_symlink():raise PermissionError('unsafe agent diagnostics directory')
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = folder/(invocation_id+'.json')
    temporary = folder/(invocation_id+'.tmp')
    # Exclusive creation prevents following a staged link or overwriting evidence.
    with temporary.open('x', encoding='utf-8') as out:
        os.chmod(temporary, 0o600)
        def scrub(value):
            if isinstance(value,str):return redact(value)
            if isinstance(value,dict):return {k:scrub(v) for k,v in value.items()}
            if isinstance(value,list):return [scrub(v) for v in value]
            return value
        out.write(json.dumps(scrub(record),ensure_ascii=True))
    temporary.replace(target)
    prune(folder)
    return str(target)
