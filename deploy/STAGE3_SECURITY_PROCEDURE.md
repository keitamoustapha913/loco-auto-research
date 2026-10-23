# Stage 3 security implementation procedure — approval required

**Status: audit and proposed procedure only. No host permissions, users, ACLs, units, or autonomy flags have been changed.** Run the privileged steps only after operator approval and a maintenance window. Commands below assume the existing Ubuntu 22.04 host and administrator `theyeq-admin-pc`; substitute paths only after review. Do not inspect or mount official test images or masks during development validation.

## 1. Read-only host assessment (2026-10-18)

| Control | Observed state | Consequence |
| --- | --- | --- |
| Identities | Administrator UID 1000; `loco-codex`, `loco-runner`, `loco-worker`, `loco-operator`, and evaluator names absent from passwd lookup | Codex and candidates currently share the administrator's host UID when invoked. |
| Administrator groups | Includes `sudo`, `docker`, and `lxd` | Never give either proposed identity these groups; container socket access would defeat isolation. |
| Repository | Root, `.git`, `benchmark`, `orchestrator` mode 775, administrator owned | Same-group writers can change trusted code; world can read after directory traversal. Do not share the administrator group. |
| Home and runtime | Home mode 750; runtime root mode 700 | New identities currently cannot traverse to the good dataset. |
| Runtime children | `controller`, `mlflow`, `jobs`, `logs` and existing per-run directories mode 775; ledger and MLflow DB mode 644 | Granting runtime-root traversal alone would expose controller, MLflow and other jobs. Harden siblings first. |
| Secrets | `.ssh` mode 700; `.codex` and `.aws` directories mode 775; Codex auth file mode 600; research env file mode 664 | Keep credentials in per-identity private homes; tighten directory and env-file modes before granting traversal. Contents were not read. |
| Services | MLflow, Streamlit, tracking worker run as administrator via `loco-lab`; `loco-lab.service` is not installed/enabled | Keep their existing CLI lifecycle. Introduce job units only. |
| cgroups | Unified cgroup v2; CPU, memory, IO and pids controllers present; systemd 249; current user slice has no delegated controller and no memory ceiling | Use root-owned systemd *system* template units for per-job limits. Do not depend on direct user-slice cgroup writes. |
| GPU devices | `/dev/nvidia0`, `nvidiactl`, `nvidia-uvm`, `nvidia-uvm-tools` mode 666; RTX 4090 has 24,564 MiB | Device DAC does not serialize GPU access. Retain the trusted `gpu.lock` and add explicit unit device allowlisting. VRAM is not capped by `MemoryMax`. |
| Candidate Bubblewrap | User, mount, PID and network namespaces; approved mounts; no controller or evaluator mount; toy CUDA run and dummy adversarial test passed | Good defense for candidate code, but it still starts from an administrator-owned scheduler and lacks a separate host UID/cgroup. |
| Codex | `codex_runner.invoke` still launches read-only roles as the administrator; implementer path fails closed on same UID | All Codex roles need a dedicated identity and private credential before autonomous operation. |

Current protection from Python checks: manifest allowlist, candidate ID checks, status transitions, budget accounting and `LOCO_*` gates. OS-enforced today: Bubblewrap candidate mounts/network namespaces, a per-host `flock`, Unix directory modes, and subprocess process-group timeouts. Python checks and environment flags are not a substitute for ACLs, separate identities or cgroups.

## 2. Three-zone target

- **A — trusted administrator/infrastructure:** existing administrator owns repository, ledger, MLflow, dashboard, policies and job approval. The controller validates all requests, holds the GPU lock, performs budget accounting, starts fixed systemd job units, receives exit status, and publishes MLflow events. No candidate module is imported into this zone.
- **B — `loco-codex`:** private UID/group and mode-0700 home at `/var/lib/loco-codex`. It receives a read-only research snapshot and writes only its private workspace and structured response. It may use network for literature/API access through a reviewed allowlist. Its credential store is inaccessible to Zone C. It is not in `sudo`, `docker`, `lxd`, `video`, `render`, the administrator group, or the runner group.
- **C — `loco-runner`:** different private UID/group; fixed systemd unit and Bubblewrap. It reads only staged approved candidate code, selected good images and immutable job inputs; candidate code sees only `/candidate`, `/data`, `/inputs`, `/output`, trusted development adapter files and selected NVIDIA nodes. It has no network, Codex home, SSH keys, controller, MLflow DB, or official evaluator mount. The official evaluator remains in a separate account and outside all three development process mounts.

**Approval boundary:** The administrator may review/copy Codex-produced files from a private workspace into an immutable approved-candidate staging directory. Do not give Codex or the runner write access to the main repository, `.git`, `benchmark`, `orchestrator`, policies, or evaluator. The trusted controller must hash staged source and compare it with the queued run before launching.

## 3. Snapshot and preflight (read-only or nonprivileged)

Run as `theyeq-admin-pc`; these commands do not change host security settings:

```bash
# [read-only]
getent passwd loco-codex loco-runner || true
# [read-only]
getent group loco-codex loco-runner video render docker lxd
# [read-only]
stat -c '%a %U:%G %n' /home/theyeq-admin-pc /home/theyeq-admin-pc/workspace-research /home/theyeq-admin-pc/workspace-research/loco-runtime/{controller,good_data,jobs,mlflow,logs}
# [read-only]
cat /sys/fs/cgroup/cgroup.controllers
# [read-only]
nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
# [read-only]
uv run --extra all loco-lab status
```

Save rollback metadata outside Git before any change:

```bash
# [nonprivileged modification: writes only the administrator's backup directory]
LAB_BACKUP=/home/theyeq-admin-pc/workspace-research/loco-security-backup-$(date +%Y%m%d-%H%M%S)
umask 077
mkdir -p "$LAB_BACKUP"
getfacl -p /home/theyeq-admin-pc /home/theyeq-admin-pc/.codex /home/theyeq-admin-pc/.aws /home/theyeq-admin-pc/.ssh /home/theyeq-admin-pc/workspace-research /home/theyeq-admin-pc/workspace-research/loco-auto-research >> "$LAB_BACKUP/acl.before"
getfacl -Rp /home/theyeq-admin-pc/workspace-research/loco-runtime >> "$LAB_BACKUP/acl.before"
stat -c '%a %U:%G %n' /home/theyeq-admin-pc/.codex /home/theyeq-admin-pc/.aws /home/theyeq-admin-pc/workspace-research/loco-research.env > "$LAB_BACKUP/modes.before"
systemctl --user list-unit-files 'loco*' > "$LAB_BACKUP/user-units.before"
```

Review the backup location and make a separate normal backup of the ledger before ACL edits. Do not copy datasets or checkpoints into Git.

## 4. Proposed host changes — **do not execute before approval**

All commands in this section are **privileged modifications**. They are ordered so private siblings are hardened *before* runtime traversal is granted. Keep `LOCO_SANDBOX_VERIFIED`, `LOCO_CODEX_ISOLATION_VERIFIED`, and `LOCO_ALLOW_UNATTENDED` unset throughout.

```bash
# [privileged modification: identities with private primary groups]
sudo useradd --system --user-group --create-home --home-dir /var/lib/loco-codex --shell /usr/sbin/nologin loco-codex
sudo useradd --system --user-group --create-home --home-dir /var/lib/loco-runner --shell /usr/sbin/nologin loco-runner
sudo chmod 0700 /var/lib/loco-codex /var/lib/loco-runner

# [privileged modification: harden sensitive siblings before traversal]
sudo chmod 0700 /home/theyeq-admin-pc/.codex /home/theyeq-admin-pc/.aws
sudo chmod 0600 /home/theyeq-admin-pc/workspace-research/loco-research.env
sudo chmod 0700 /home/theyeq-admin-pc/workspace-research/loco-runtime/controller
sudo chmod 0700 /home/theyeq-admin-pc/workspace-research/loco-runtime/mlflow
sudo chmod 0700 /home/theyeq-admin-pc/workspace-research/loco-runtime/logs
sudo chmod 0700 /home/theyeq-admin-pc/workspace-research/loco-runtime/jobs
sudo find /home/theyeq-admin-pc/workspace-research/loco-runtime/jobs -mindepth 1 -maxdepth 1 -type d -exec chmod 0700 {} +

# [privileged modification: private staged inputs; no new dataset copy]
sudo install -d -m 0700 -o theyeq-admin-pc -g theyeq-admin-pc /var/lib/loco-approved-candidates
sudo install -d -m 0700 -o theyeq-admin-pc -g theyeq-admin-pc /var/lib/loco-job-requests
sudo install -d -m 0700 -o root -g root /var/lib/loco-runner-output
sudo setfacl -m u:loco-runner:--x /var/lib/loco-approved-candidates /var/lib/loco-job-requests

# [privileged modification: only runner may traverse to authorized good data]
sudo setfacl -m u:loco-runner:--x /home/theyeq-admin-pc /home/theyeq-admin-pc/workspace-research /home/theyeq-admin-pc/workspace-research/loco-runtime
sudo setfacl -R -m u:loco-runner:r-X /home/theyeq-admin-pc/workspace-research/loco-runtime/good_data
sudo find /home/theyeq-admin-pc/workspace-research/loco-runtime/good_data -type d -exec setfacl -m d:u:loco-runner:r-x {} +

# [privileged modification: deny known credential/trusted source directories]
sudo setfacl -m u:loco-runner:--- /home/theyeq-admin-pc/.codex /home/theyeq-admin-pc/.aws /home/theyeq-admin-pc/.ssh /home/theyeq-admin-pc/workspace-research/loco-auto-research
```

Do not add either identity to `sudo`, `docker`, `lxd`, or the administrator group. The observed NVIDIA device nodes are mode 666, so neither identity needs `video` or `render` now. If a future driver update narrows device modes, add `loco-runner` only to the minimum required device group after testing; never add `loco-codex` to GPU groups.

For each validated run UUID, the trusted controller creates an immutable request under `/var/lib/loco-job-requests/<UUID>` and grants `loco-runner` read/traverse ACL only on that request and its approved input files. The controller must validate the request's owner, mode, schema and UUID before unit launch. The runner **never receives traversal on `runtime/jobs`**. A trusted root-owned broker mounts a 4 GiB tmpfs at `/var/lib/loco-runner-output/<UUID>` before launch, binds it into the unit as `/run/loco-candidate-out`, and copies allowlisted artifacts into the administrator-owned job directory after candidate exit. The broker then unmounts and removes the temporary directory. Revoke each request ACL after collection. Older job directories stay private. The runner cannot traverse the host output parent; its unit sees only its own bind mount.

The approved-candidate staging directory remains administrator-owned. After static review, the controller copies only candidate code and declared resources into `/var/lib/loco-approved-candidates/<id>`, records hashes, removes write permission, and grants `loco-runner` traverse on the staging root and read/traverse only for that candidate. A separate, immutable runner Python/CUDA environment must be staged outside the trusted repository (for example `/opt/loco-runner-venv` with read-only ACL); preserve the installed PyTorch/CUDA package selection. Codex's private workspace is never mounted in a candidate unit. No runner ACL is granted on the trusted repository or `.venv` there.

For each approved UUID, these are the **privileged modification** ACL commands the trusted administrator must apply *after* copying and hashing inputs; never substitute a path supplied directly by research JSON:

```bash
# [privileged modification: UUID-specific read access; replace both placeholders with validated values]
sudo setfacl -R -m u:loco-runner:r-X /var/lib/loco-approved-candidates/<approved-id>
sudo setfacl -R -m u:loco-runner:r-X /var/lib/loco-job-requests/<UUID>
# [privileged modification: revoke access immediately after collection]
sudo setfacl -R -x u:loco-runner /var/lib/loco-job-requests/<UUID>
```

The approved source remains immutable for the job lifetime; do not allow shared group write or default write ACLs. The runner runtime under `/opt` must similarly be root-owned, read-only to `loco-runner`, and built from the already verified CUDA package versions. Codex's private workspace must be transferred to the trusted controller by a root-owned broker after its unit exits; do not open a shared writable spool between the two identities.

The trusted broker's per-job output setup and teardown use the following **privileged modifications** with a UUID already validated by the controller. These commands are a specification for the broker, not a manual research workflow:

```bash
# [privileged modification: create and mount one bounded output filesystem]
sudo install -d -m 0700 -o root -g root /var/lib/loco-runner-output/<UUID>
sudo mount -t tmpfs -o nosuid,nodev,noexec,size=4G,mode=0700 tmpfs /var/lib/loco-runner-output/<UUID>
sudo chown loco-runner:loco-runner /var/lib/loco-runner-output/<UUID>
# [privileged modification: after the unit is stopped and trusted artifact collection is complete]
sudo umount /var/lib/loco-runner-output/<UUID>
sudo rmdir /var/lib/loco-runner-output/<UUID>
```

The broker must use directory file descriptors, refuse symlinks and hard links, enforce artifact count and byte limits, and verify file types and names when collecting output. A failed or crashed job must still trigger collection or quarantine, unmount, ACL revocation and GPU lock release. Never leave a writable mount accessible to the next job.

## 5. Required source and unit changes after approval

The following are **nonprivileged repository modifications** to implement and review before installing units; they must not activate autonomy:

1. `orchestrator/codex_runner.py` and `create_candidate_worktree`: send schema-constrained role jobs to a fixed `loco-codex@.service`; use a private workspace outside the trusted checkout. Remove direct `codex exec` as administrator. Stage responses in a controller-readable, runner-inaccessible spool. The Codex unit gets only its own auth material and network; it does not receive the administrator's environment.
2. `orchestrator/controller.py`: validate role responses, source references, dependency approvals, hashes, candidate provenance and budget; stage approved source; request only fixed UUID instance units. Never execute a command from research JSON.
3. `orchestrator/experiment_runner.py` and `sandbox.py`: launch the approved candidate through `loco-candidate@<UUID>.service`, using a root-prepared, per-job bounded tmpfs output. A trusted broker copies only allowlisted artifacts back to the existing runtime and unmounts the tmpfs after the unit exits. The candidate cannot see the persistent runtime tree. The trusted controller records all exit states, cgroup OOMs, wall time, and failed artifact validation in SQLite.
4. `orchestrator/gpu_queue.py`: keep the existing `flock` through unit completion. Serialize all RTX 4090 jobs. Cancellation must signal the fixed unit and charge consumed GPU time.
5. Add root-owned, fixed-command `/usr/local/libexec/loco-codex-role` and `/usr/local/libexec/loco-candidate-job` launchers and a trusted root-owned job preparation/collection broker, plus `deploy/systemd/loco-codex@.service` and `deploy/systemd/loco-candidate@.service`. Validate the instance name as a UUID and read only trusted job-request files. Launchers must never accept arbitrary shell commands or paths from agent JSON. The privileged broker must never import or execute candidate code.
6. Add a narrow polkit rule allowing only the administrator identity to start/stop the two fixed template unit families, and only after validated UUID requests exist. No Codex or runner authorization. Do not grant general `systemd-run`, `systemctl`, or root shell permission to research code. The trusted controller must retain sole possession of the request-signing or approval capability.

Proposed candidate unit core (template file staged in repo, then installed root-owned at `/etc/systemd/system/loco-candidate@.service` after approval):

```ini
[Unit]
Description=LOCO candidate %i
[Service]
Type=exec
User=loco-runner
Group=loco-runner
ExecStart=/usr/local/libexec/loco-candidate-job %i
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
PrivateNetwork=yes
ProtectSystem=strict
CPUQuota=200%
MemoryMax=16G
MemorySwapMax=0
TasksMax=64
RuntimeMaxSec=1800
TimeoutStopSec=10
KillMode=control-group
OOMPolicy=stop
DevicePolicy=closed
DeviceAllow=/dev/nvidia0 rw
DeviceAllow=/dev/nvidiactl rw
DeviceAllow=/dev/nvidia-uvm rw
DeviceAllow=/dev/nvidia-uvm-tools rw
TemporaryFileSystem=/tmp:rw,nosuid,nodev,size=1G,mode=1777
BindPaths=/var/lib/loco-runner-output/%i:/run/loco-candidate-out
```

The broker prepares the bounded tmpfs and bind source before unit start. The launcher binds `/run/loco-candidate-out` as Bubblewrap `/output`. A trusted administrator process copies allowlisted artifacts to the assigned persistent job directory **after** candidate exit; the runner must not gain write access there. The existing `--tmpfs /tmp` in `sandbox.py` must be replaced by binding the unit's bounded `/tmp`; otherwise the 1 GiB limit would not apply inside Bubblewrap. `ProtectSystem=strict`, `BindPaths=` and `DevicePolicy=closed` require a positive CUDA and output test before use. The unit must not mount the ledger, MLflow DB, Codex home or official evaluator.

Proposed Codex unit core: `User=loco-codex`, `NoNewPrivileges=yes`, `ProtectSystem=strict`, `DevicePolicy=closed`, `CPUQuota=100%`, `MemoryMax=4G`, `TasksMax=64`, `RuntimeMaxSec=600`, private `HOME=/var/lib/loco-codex`, and narrowly scoped `ReadWritePaths` for only its private workspace. Network access is required for Codex but must be restricted to an approved egress proxy, with `IPAddressDeny=any` and a narrowly reviewed `IPAddressAllow=` for that proxy or equivalent firewall rules. Direct unrestricted Codex egress remains a blocker. Its credentials must be provisioned only in that private home or via systemd credentials, never the candidate environment.

**Resource truth:** CPU, host RAM, swap, task count and wall time are enforceable by the systemd unit. A bounded per-job tmpfs enforces the output byte ceiling and survives the candidate unit until trusted collection; cgroup memory accounts for the unit's `/tmp` pages, while the separate output tmpfs must also be monitored and limited by its mount size. Existing `LOCO_GPU_BUDGET_HOURS`, job timeout and `gpu.lock` remain trusted scheduler checks. Ordinary memory cgroups do **not** hard-cap RTX 4090 VRAM. A candidate may OOM or contend with other host GPU users; use one-job serialization, CUDA monitoring, a bounded workload, and explicit `OOM`/`TIMED_OUT`/`CANCELLED` accounting. Device allowlisting controls which nodes are visible; it does not reserve GPU memory or compute time.

## 6. Privileged installation commands, only after code and unit review

These are **privileged modifications**, not part of this audit:

```bash
# [privileged modification: install fixed reviewed files]
sudo install -m 0755 -o root -g root deploy/bin/loco-codex-role /usr/local/libexec/loco-codex-role
sudo install -m 0755 -o root -g root deploy/bin/loco-candidate-job /usr/local/libexec/loco-candidate-job
sudo install -m 0644 -o root -g root deploy/systemd/loco-codex@.service /etc/systemd/system/loco-codex@.service
sudo install -m 0644 -o root -g root deploy/systemd/loco-candidate@.service /etc/systemd/system/loco-candidate@.service
sudo install -m 0644 -o root -g root deploy/polkit/49-loco-job-units.rules /etc/polkit-1/rules.d/49-loco-job-units.rules
sudo systemctl daemon-reload
# [read-only verification]
systemd-analyze verify /etc/systemd/system/loco-codex@.service /etc/systemd/system/loco-candidate@.service
```

The `deploy/bin` and `deploy/polkit` files above are **to be implemented and reviewed after approval**. Do not run these commands against nonexistent or unreviewed files. Installing templates does not enable a campaign. Do not enable lingering or replace the running `loco-lab` services. The operator continues using `loco-lab start/status/smoke/logs/stop` exactly as today.

## 7. Adversarial acceptance matrix (dummy fixtures only)

Create a fake protected-data tree and a fake Codex token **outside** the candidate and Codex mounts. The test runner should run both identities and attempt:

| Test | Expected result |
| --- | --- |
| Candidate reads dummy protected image or fake token | Permission denied or path absent |
| Candidate writes trusted controller, evaluator or policy fixture | Permission denied or path absent; hash unchanged |
| Candidate reaches MLflow `127.0.0.1:56000` | Connection refused/unreachable in private network namespace |
| Candidate uses `..`, symlink, bind or `/proc/self/fd` to escape output | No host path outside assigned output becomes readable/writable |
| Candidate reads another UUID's artifacts or staged input | Permission denied; only its own UUID is granted |
| Candidate starts shell, `sudo`, `systemctl` or nested subprocess | Privileged commands fail; permitted children remain in the same cgroup and namespace with no network gain |
| Candidate spins more than two CPU cores | Cgroup CPU usage is throttled to `CPUQuota=200%`; job remains cancellable |
| Candidate fork bomb | `TasksMax=64` stops growth; job marked failed and charged |
| Candidate exceeds 16 GiB RAM | Unit OOM kill; ledger status `OOM`; outbox retained |
| Candidate fills bounded `/tmp` or output | `ENOSPC`; job fails without filling host filesystem |
| Candidate exceeds 1800 seconds or operator cancels | Entire cgroup terminated; ledger `TIMED_OUT` or `CANCELLED` and GPU time charged |
| Codex attempts to write trusted repo or read fake evaluator fixture | Denied; only private workspace writable |
| Runner attempts to read Codex private home | Denied by DAC/ACL and absent from Bubblewrap |
| Positive: toy CUDA train/infer on authorized good images | Completes; valid heatmap, GPU timing, SQLite and MLflow records |

Existing `tests/test_sandbox.py` covers mount allowlisting, dummy file denial, trusted-write denial, fake secret inheritance and MLflow loopback isolation. Add integration tests under the *actual* new UIDs and systemd units for the remaining rows. Never point tests at real official anomaly files or masks. Ask the independent evaluator operator to attest its account and mount permissions separately; do not grant this account access to the sealed data for testing.

## 8. Verification gate and rollback

After implementation, run these **read-only** checks as administrator:

```bash
# [read-only]
getent passwd loco-codex loco-runner
# [read-only]
id loco-codex && id loco-runner
# [read-only]
getfacl -cp /home/theyeq-admin-pc/workspace-research/loco-runtime/{controller,good_data,jobs,mlflow}
# [read-only]
systemctl show 'loco-candidate@<UUID>.service' -p Result -p MemoryMax -p TasksMax -p CPUQuotaPerSecUSec
# [read-only]
uv run --extra all loco-lab status
# [read-only]
curl -fsS http://127.0.0.1:56000/health && curl -fsS http://127.0.0.1:8501/_stcore/health
# [read-only]
uv run --extra all loco-lab doctor
```

Check that the pre-existing SQLite run count and dataset fingerprint are unchanged, all existing tests pass, the dummy adversarial matrix passes, CUDA still works in the new unit, and a bounded toy run reaches MLflow. Record unit exit/OOM evidence and ACL snapshots. Only then may the operator consider setting the autonomy flags. A dashboard page or successful Bubblewrap smoke test alone is insufficient.

Rollback if any denial or positive test fails:

```bash
# [privileged modification: stop only new job units; do not stop existing lab services]
sudo systemctl stop loco-candidate@<reviewed-UUID>.service loco-codex@<reviewed-UUID>.service
# [privileged modification: remove only new unit/polkit files]
sudo rm -f /etc/systemd/system/loco-candidate@.service /etc/systemd/system/loco-codex@.service /etc/polkit-1/rules.d/49-loco-job-units.rules
sudo systemctl daemon-reload
# [privileged modification: restore saved ACLs/modes, using the exact reviewed backup]
sudo setfacl --restore=/path/to/acl.before
# [privileged modification: restore selected mode bits from modes.before after review]
# sudo chmod <previous-mode> <exact-path>
# [privileged modification: revoke only the new staging-root access]
sudo setfacl -x u:loco-runner /var/lib/loco-approved-candidates /var/lib/loco-job-requests
# [privileged modification: unmount any reviewed leftover per-job output before removing its mount point]
sudo umount /var/lib/loco-runner-output/<reviewed-UUID>
```

Do not delete identities, workspaces, job outputs, ledger, checkpoints or candidate source automatically; preserve evidence for diagnosis. Revoke new ACLs or disable unit authorization first. Never roll back by granting broad write permissions. Keep `LOCO_ALLOW_UNATTENDED` unset throughout rollback.

## 9. Explicit approval request

Approval is needed for the two user accounts, ACL/mode hardening, root-owned launchers and systemd templates, the narrow polkit rule, and bounded job test units. The host changes above are **not authorized by this audit**. Approval should identify whether the administrator may execute those specific commands and whether an independent evaluator operator will attest sealed-storage permissions. No official test data should enter the development account at any point.
