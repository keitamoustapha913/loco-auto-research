"""Bounded, structured Codex invocation. No researcher-authored shell commands run."""
from __future__ import annotations
import hashlib
import copy
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import uuid
import signal
import time
import selectors
import traceback
from jsonschema import validate, Draft202012Validator, SchemaError, ValidationError

class AgentFailure(RuntimeError):
    pass


class AgentTimedOut(AgentFailure):
    """Recoverable invocation failure, distinct from scientific/candidate errors."""


class InvalidAgentOutput(AgentFailure):
    """An agent completed but its response needs correction, not host changes."""


class AgentUnavailable(AgentFailure):
    """Missing host executable requires operator maintenance, not research retries."""


class ProviderSchemaError(AgentFailure):
    """Local schema defect: never retry it as an agent-output repair."""


def run_cancellable(command, *, timeout, cancel=None, deadline=None, observation=None, **kwargs):
    """Poll cancellation and reap the whole owned process group on every exit."""
    from orchestrator.agent_diagnostics import TAIL_BYTES
    task=kwargs.pop('input',None)
    text_mode=kwargs.pop('text',False)
    kwargs.pop('capture_output',None)
    start=time.time(); end_mono=time.monotonic()+max(0,min(timeout,deadline-start if deadline is not None else timeout))
    observed=observation if observation is not None else {}
    output={'stdout':b'', 'stderr':b''}; totals={'stdout':0,'stderr':0}
    task=task.encode() if isinstance(task,str) else task
    with subprocess.Popen(command,stdin=subprocess.PIPE if task is not None else subprocess.DEVNULL,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,**kwargs) as process:
        observed.update(pid=process.pid,last_observed_at=start,last_output_at=None,progress_events=[])
        selector=selectors.DefaultSelector()
        for stream,name in ((process.stdout,'stdout'),(process.stderr,'stderr')):
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,name)
        pending=memoryview(task or b''); lines=b''
        if task:
            os.set_blocking(process.stdin.fileno(),False);selector.register(process.stdin,selectors.EVENT_WRITE,'stdin')
        elif process.stdin:process.stdin.close()
        last_sample=0.
        try:
            while selector.get_map() or process.poll() is None:
                now=time.time()
                if (cancel and cancel()) or time.monotonic()>=end_mono:
                    observed['termination_reason']=('campaign_deadline' if deadline is not None and now>=deadline else
                        'call_deadline' if time.monotonic()>=end_mono else 'cancelled')
                    raise subprocess.TimeoutExpired(command,timeout,output=output['stdout'],stderr=output['stderr'])
                if now-last_sample>=1:
                    observed['last_observed_at']=now;last_sample=now
                    try:
                        import psutil
                        p=psutil.Process(process.pid)
                        children=p.children(recursive=True)
                        cpu=sum(sum(proc.cpu_times()[:2]) for proc in [p]+children)
                        previous=observed.get('process_sample',{}).get('cpu_seconds')
                        if previous is not None and cpu!=previous:observed['last_process_counter_change_at']=now
                        observed['process_sample']={'status':p.status(),'cpu_seconds':cpu,'children':len(children)}
                    except (OSError,psutil.Error):pass
                for key,_ in selector.select(min(.1,max(.001,end_mono-time.monotonic()))):
                    name=key.data
                    if name=='stdin':
                        try:pending=pending[os.write(key.fd,pending[:65536]):]
                        except BrokenPipeError:pending=memoryview(b'')
                        if not pending:selector.unregister(key.fileobj);key.fileobj.close()
                        continue
                    chunk=os.read(key.fd,65536)
                    if not chunk:selector.unregister(key.fileobj);continue
                    totals[name]+=len(chunk);output[name]=(output[name]+chunk)[-TAIL_BYTES:]
                    observed['last_output_at']=now
                    if name=='stdout':
                        lines+=chunk
                        while b'\n' in lines:
                            line,lines=lines.split(b'\n',1)
                            try:
                                event=json.loads(line)
                                if isinstance(event,dict) and isinstance(event.get('type'),str):
                                    observed['progress_events']=(observed['progress_events']+[{'at':now,'type':event['type'][:128]}])[-32:]
                                    if event['type']=='turn.failed':
                                        from orchestrator.agent_diagnostics import redact
                                        observed['termination_reason']='structured_terminal_failure'
                                        error=event.get('error',{})
                                        if isinstance(error,dict) and error.get('code') in {'invalid_api_key','authentication_error'}:
                                            raise AgentUnavailable('Codex reported nonrecoverable authentication failure')
                                        raise AgentFailure('Codex turn failed: '+redact(json.dumps(event))[-1024:])
                            except (ValueError,UnicodeDecodeError):pass
                        lines=lines[-TAIL_BYTES:]
            observed['termination_reason']='completed' if process.returncode==0 else 'process_error'
            out,err=output['stdout'],output['stderr']
            return subprocess.CompletedProcess(command,process.returncode,out.decode(errors='replace') if text_mode else out,
                                               err.decode(errors='replace') if text_mode else err)
        finally:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
            # Descendants must not survive a completed wrapper or inherited pipes.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            selector.close()
            observed.update(exit_code=process.returncode,stdout=output['stdout'].decode(errors='replace'),
                            stderr=output['stderr'].decode(errors='replace'),output_bytes=totals)


def preflight_provider_schema(schema):
    """Conservative local subset checks, not proof of remote API acceptance.

    Source: https://developers.openai.com/api/docs/guides/structured-outputs
    Lookaround/backreference rejection also follows saved provider error evidence.
    Visit schema positions only; enum/const values may contain arbitrary JSON.
    Unknown keywords fail closed so future additions require explicit review.
    """
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ProviderSchemaError('Invalid JSON schema: '+exc.message) from exc
    allowed={'$schema','$defs','$ref','type','properties','required','additionalProperties',
             'items','anyOf','enum','const','title','description','pattern','format',
             'minLength','maxLength','minimum','maximum','exclusiveMinimum','exclusiveMaximum',
             'multipleOf','minItems','maxItems'}
    issues=[]
    if not isinstance(schema,dict):raise ProviderSchemaError('Provider schema root must be an object schema')
    if schema.get('type')!='object' or 'anyOf' in schema:
        issues.append('$: provider root must be an object without anyOf')
    def walk(node,path='$'):
        if not isinstance(node,dict):return
        for key in sorted(set(node)-allowed):
            issues.append(f'{path}.{key}: unsupported or unreviewed provider keyword')
        reference=node.get('$ref')
        if reference is not None:
            target=schema
            try:
                if not reference.startswith('#/'):
                    raise ValueError('only local JSON-pointer references are supported')
                for part in reference[2:].split('/'):
                    target=target[part.replace('~1','/').replace('~0','~')]
            except (KeyError,TypeError,ValueError) as exc:
                issues.append(f'{path}.$ref: unresolved or nonlocal reference {reference}: {exc}')
        pattern=node.get('pattern')
        if isinstance(pattern,str):
            if any(token in pattern for token in ('(?=','(?!','(?<=','(?<!','(?P=','(?(', '(?R)')) or re.search(r'\\[1-9]|\\[kg][<{]',pattern):
                issues.append(f'{path}.pattern: regex lookaround/backreferences are unsupported')
            try:re.compile(pattern)
            except re.error as exc:issues.append(f'{path}.pattern: invalid regex: {exc}')
        types=node.get('type',[])
        if types=='object' or 'properties' in node or (isinstance(types,list) and 'object' in types):
            if node.get('additionalProperties') is not False:
                issues.append(f'{path}.additionalProperties: strict objects require false')
            if set(node.get('required',[]))!=set(node.get('properties',{})):
                issues.append(f'{path}.required: every declared property must be required')
        for name in ('properties','$defs','definitions','patternProperties','dependentSchemas'):
            for key,child in node.get(name,{}).items():walk(child,f'{path}.{name}.{key}')
        for name in ('items','additionalProperties','contains','not','if','then','else','propertyNames'):
            walk(node.get(name),f'{path}.{name}')
        for name in ('anyOf','allOf','oneOf','prefixItems'):
            for index,child in enumerate(node.get(name,[])):walk(child,f'{path}.{name}[{index}]')
    walk(schema)
    if issues:raise ProviderSchemaError('Provider schema preflight failed: '+'; '.join(issues))
    return schema


def provider_output_schema(schema):
    """Compile the API subset; the original schema still validates every response.

    Strict structured output rejects uniqueItems. Removing it only from the
    provider's generation schema does not relax trusted acceptance constraints.
    Traverse schema positions, never literal const/default/example dictionaries.
    """
    result=copy.deepcopy(schema)
    def walk(node):
        if not isinstance(node,dict):return
        node.pop('uniqueItems',None)
        for name in ('properties','$defs','definitions','patternProperties','dependentSchemas'):
            for child in node.get(name,{}).values():walk(child)
        for name in ('items','additionalProperties','contains','not','if','then','else','propertyNames'):
            walk(node.get(name))
        for name in ('anyOf','allOf','oneOf','prefixItems'):
            for child in node.get(name,[]):walk(child)
    walk(result)
    return preflight_provider_schema(result)


def invoke(role: str, repo_root: Path, schema_path: Path, prompt: str, timeout=1800, worktree=None,
           personal_mode=False, reference_files=None, cancel=None, deadline=None, invocation_id=None, diagnostics_dir=None):
    if role not in {"scout", "researcher", "implementer", "critic"}:
        raise ValueError("unknown agent role")
    root = Path(repo_root).resolve(strict=True)
    cwd = Path(worktree).resolve(strict=True) if worktree else root
    if role == "implementer" and not personal_mode:
        operator_uid = root.stat().st_uid
        if os.getenv('LOCO_CODEX_ISOLATION_VERIFIED') != '1' or os.geteuid() == operator_uid:
            raise AgentFailure('Implementer Codex execution requires a separate verified OS identity and workspace mounts')
        if worktree is None or not cwd.is_relative_to((root / "candidates").resolve()):
            raise AgentFailure("implementer needs assigned candidates/ worktree")
    elif role != "implementer" and worktree is not None:
        raise AgentFailure("only implementer receives a writeable candidate worktree")
    if role == "implementer" and (worktree is None or not cwd.is_relative_to((root / "candidates" / "worktrees").resolve())):
        raise AgentFailure("implementer needs assigned candidates/ worktree")
    role_text = (root / "agents" / f"{role}.md").read_text()
    charter = (root / "RESEARCH_CHARTER.md").read_text()
    task = f"{role_text}\n\nHUMAN RESEARCH CHARTER:\n{charter}\n\nTASK:\n{prompt}"
    if role=='implementer' and personal_mode:
        task += (f"\nEXECUTION INTERFACE: cwd=/workspace; writable candidate directory=/workspace/candidates/{cwd.name}. "
            "Use the Codex file-editing/shell tools provided in this sandbox to read /workspace/AGENTS.md, "
            "/workspace/RESEARCH_CHARTER.md, /workspace/research_program.md and the mounted evidence. "
            f"Create /workspace/candidates/{cwd.name}/model.py and test_model.py before returning. "
            "No separate trusted-runner command tool exists in the agent session: host validation runs after your response. "
            "The assigned edits and CPU/static inspection are authorized within these mounts. GPU execution and "
            "candidate evaluation remain controller-owned; no installs, credentials, sudo, network fetching or "
            "protected-file modifications are authorized. implemented_files must identify the actual files; "
            "a blocked/no-work response is not success.")
    schema = json.loads(Path(schema_path).read_text())
    with tempfile.TemporaryDirectory(prefix="codex-response-") as temp:
        response_file = Path(temp) / "response.json"
        provider_schema_file=Path(temp)/'provider-schema.json'
        provider_schema_file.write_text(json.dumps(provider_output_schema(schema)))
        if personal_mode:
            from orchestrator.codex_sandbox import command as sandbox_command
            candidate_dir = cwd / "candidates" / cwd.name if role == "implementer" else None
            if candidate_dir is not None:
                candidate_dir.mkdir(parents=True, exist_ok=True)
            planning_reference = None
            if role == 'researcher':
                generated = json.loads(provider_schema_file.read_text())
                plan = generated.get('$defs',{}).get('proposal',{}).get('properties',{}).get('experiment_plan')
                if plan is None and 'arms' in generated.get('properties',{}):plan = generated
                if plan is not None:
                    planning_reference = Path(temp)/'experiment-plan.json'
                    planning_reference.write_text(json.dumps(plan))
            command = sandbox_command(root, role, provider_schema_file, response_file, task, candidate_dir,
                                      reference_files=reference_files, planning_reference=planning_reference)
            env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
        else:
            command = ["codex", "exec", "--json", "--cd", str(cwd), "--sandbox", "workspace-write" if role == "implementer" else "read-only", "--output-schema", str(provider_schema_file), "--output-last-message", str(response_file), '-']
            env = {k: v for k, v in os.environ.items() if k not in ("LOCO_SEALED_ROOT", "MLFLOW_TRACKING_TOKEN", "GITHUB_TOKEN")}
        invocation_id=invocation_id or uuid.uuid4().hex
        observation={'invocation_id':invocation_id,'role':role,'started_at':time.time(),
                     'campaign_deadline':deadline,'timeout_seconds':min(timeout,1800),
                     'prompt_sha256':hashlib.sha256(task.encode()).hexdigest()}
        try:
            try:
                # Whole-history evidence can exceed Linux's per-argument size limit.
                # The CLI's stdin prompt mode preserves the exact task and sandbox.
                done=run_cancellable(command,input=task,cwd=cwd,env=env,text=True,
                                     timeout=min(timeout,1800),cancel=cancel,deadline=deadline,observation=observation)
            except FileNotFoundError as exc:
                raise AgentUnavailable("Codex CLI not installed on server") from exc
            except subprocess.TimeoutExpired as exc:
                from orchestrator.agent_diagnostics import redact
                detail = (exc.stderr or b'')
                if isinstance(detail, bytes):
                    detail = detail.decode(errors='replace')
                observation.setdefault('termination_reason','call_deadline')
                observation['traceback']=traceback.format_exc()[-8192:]
                raise AgentTimedOut(f"Codex invocation timed out ({observation['termination_reason']}; invocation {invocation_id}): {redact(detail[-600:])}") from exc
            except Exception:
                observation['traceback']=traceback.format_exc()[-8192:]
                raise
            if done.returncode:
                from orchestrator.agent_diagnostics import redact
                raise AgentFailure(f"Codex exited {done.returncode}: {redact(done.stderr[-800:])}")
            if not response_file.is_file():
                raise InvalidAgentOutput('Codex completed without the required structured response file')
            try:
                output = json.loads(response_file.read_text())
                validate(instance=output, schema=schema)
            except (json.JSONDecodeError,ValidationError) as exc:
                from orchestrator.agent_diagnostics import redact
                raise InvalidAgentOutput(redact(f"Codex produced invalid schema-constrained JSON: {exc}\n"
                                         f"Rejected response: {response_file.read_text()[-8000:]}")) from exc
        except Exception as exc:
            observation["error_type"]=type(exc).__name__
            observation["traceback"]=traceback.format_exc()[-8192:]
            raise
        finally:
            observation.update(finished_at=time.time(),response_file_created=response_file.is_file())
            observation.setdefault('exit_code',None)
            observation.setdefault('termination_reason','invocation_error')
            if response_file.is_file():observation['response_tail']=response_file.read_bytes()[-65536:].decode(errors='replace')
            if diagnostics_dir is not None:
                from orchestrator.agent_diagnostics import save
                save(diagnostics_dir,invocation_id,observation)
    return {"response": output, "prompt_sha256": hashlib.sha256(task.encode()).hexdigest(), "schema_sha256": hashlib.sha256(Path(schema_path).read_bytes()).hexdigest()}


def create_candidate_worktree(repo_root, hypothesis_id, base_ref="HEAD"):
    """Operator must first approve ID and use dedicated OS user for implementer."""
    root = Path(repo_root).resolve(strict=True)
    if not hypothesis_id or not all(c.isascii() and (c.isalnum() or c in "_-") for c in hypothesis_id):
        raise ValueError("invalid hypothesis ID")
    destination = root / "candidates" / "worktrees" / hypothesis_id
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", str(destination), base_ref], check=True)
    return destination


def commit_candidate_snapshot(repo_root, hypothesis_id):
    """Trusted controller commits only reviewed candidate files on a local branch."""
    root = Path(repo_root).resolve(strict=True)
    if not hypothesis_id or not all(c.isascii() and (c.isalnum() or c in '_-') for c in hypothesis_id):
        raise ValueError('invalid hypothesis ID')
    tree = (root / 'candidates' / 'worktrees' / hypothesis_id).resolve(strict=True)
    candidate = tree / 'candidates' / hypothesis_id
    if candidate.is_symlink() or not candidate.resolve(strict=True).is_relative_to(tree):
        raise PermissionError('candidate directory escapes assigned worktree')
    allowed = {'model.py', 'README.md', 'test_model.py', 'candidate_manifest.json'}
    entries = list(candidate.iterdir())
    files = sorted(entries)
    if not (candidate / 'model.py').is_file() or any(not p.is_file() or p.is_symlink() or p.name not in allowed for p in files):
        raise ValueError('candidate snapshot contains missing or unapproved files')
    status = subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(tree),
                             'status', '--porcelain', '--untracked-files=all'],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    for line in status:
        if line[3:].replace('"', '') not in {f'candidates/{hypothesis_id}/{p.name}' for p in files}:
            raise PermissionError('candidate worktree contains out-of-scope changes')
    branch = f'loco-candidate/{hypothesis_id}'
    current_branch = subprocess.run(['git', '-C', str(tree), 'branch', '--show-current'],
                                    check=True, capture_output=True, text=True).stdout.strip()
    if current_branch != branch:
        subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(tree), 'switch', '-c', branch],
                       check=True, capture_output=True, text=True)
    subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-C', str(tree), 'add', '--',
                    *[str(p.relative_to(tree)) for p in files]], check=True)
    changed = subprocess.run(['git', '-C', str(tree), 'diff', '--cached', '--quiet'], check=False)
    if changed.returncode not in (0, 1):
        raise RuntimeError('cannot inspect repaired candidate snapshot')
    if changed.returncode:
        subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=LOCO Research Controller',
                    '-c', 'user.email=loco-controller@localhost', '-C', str(tree), 'commit',
                    '-m', f'Candidate {hypothesis_id}: isolated research snapshot'],
                       check=True, capture_output=True, text=True)
    sha = subprocess.run(['git', '-C', str(tree), 'rev-parse', 'HEAD'], check=True,
                         capture_output=True, text=True).stdout.strip()
    return sha
