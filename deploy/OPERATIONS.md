# Operator operations

Use `uv run --extra all loco-lab start`, `status`, `doctor`, `logs`, `restart`, and `stop` from the repository. The CLI supervises MLflow, Streamlit, and the tracking consumer through PID ownership records in `$LOCO_RUNTIME/controller`, a structured lifecycle log and service logs in `$LOCO_RUNTIME/logs`, local HTTP health checks, and a lifecycle file lock. It binds only to `127.0.0.1` on ports 56000 and 8501. An occupied unmanaged port is an error; the CLI will not kill it.

To run the bounded infrastructure check explicitly, use `uv run --extra all loco-lab smoke --dataset juice_bottle --device cuda`. It validates the authorized good-only manifest and CUDA, enforces the existing compute budget and GPU lock, then uses the toy candidate through the existing guarded runner. Scientific campaign execution is separate and remains blocked until OS isolation is verified.

Optional startup after user login: copy `deploy/systemd/loco-lab.service` into `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, then `systemctl --user enable --now loco-lab.service`. This template does not enable user lingering. Do not run it alongside the legacy individual service units.

For laptop access, forward both ports in one SSH command as shown in README. The browser is read-only. SQLite is authoritative; MLflow outages leave outbox events queued. Inspect `loco-lab logs` and the System page for retry errors.
