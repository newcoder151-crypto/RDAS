#!/usr/bin/env python3
"""
run_all.py — single command to bring up the whole RDAS stack:

    MediaMTX (RTSP relay)  ->  backend (stream_pipeline.py)  ->  frontend (vite dev)

Usage:
    python run_all.py                  # run everything
    python run_all.py --skip-setup     # skip the auto asset-download check
    python run_all.py --skip-mediamtx  # you're already running MediaMTX yourself
    python run_all.py --skip-frontend  # backend + MediaMTX only
    python run_all.py --skip-backend   # MediaMTX + frontend only

Ctrl+C stops everything, in reverse start order.

What this does NOT do: fix your RDAS_INPUT_URL. That's your real camera's
RTSP URL/credentials and only you know it — edit backend/.env before running.
"""
import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
MEDIAMTX_DIR = ROOT / "mediamtx"
SCRIPTS_DIR = ROOT / "scripts"

IS_WINDOWS = platform.system().lower() == "windows"

# ANSI colors for the log prefixes (harmless if the terminal doesn't support them)
COLORS = {"mediamtx": "\033[36m", "backend": "\033[35m", "frontend": "\033[33m", "setup": "\033[32m"}
RESET = "\033[0m"

_procs: list[tuple[str, subprocess.Popen]] = []
_shutting_down = threading.Event()


def _log(tag: str, line: str) -> None:
    color = COLORS.get(tag, "")
    print(f"{color}[{tag}]{RESET} {line.rstrip()}", flush=True)


def _stream_output(tag: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for raw_line in iter(proc.stdout.readline, b""):
        if not raw_line:
            break
        _log(tag, raw_line.decode(errors="replace"))


def _popen(tag: str, cmd, cwd=None, env=None) -> subprocess.Popen:
    _log(tag, f"$ {' '.join(str(c) for c in cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    kwargs = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    _procs.append((tag, proc))
    threading.Thread(target=_stream_output, args=(tag, proc), daemon=True).start()
    return proc


def _kill(tag: str, proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _log(tag, f"stopping (pid={proc.pid})...")
    try:
        if IS_WINDOWS:
            # Kills the whole process tree — plain terminate() on the shell
            # process alone tends to leave node/vite orphaned on Windows.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _shutdown(*_):
    if _shutting_down.is_set():
        return
    _shutting_down.set()
    print()
    _log("run_all", "shutting down all components...")
    for tag, proc in reversed(_procs):
        _kill(tag, proc)
    sys.exit(0)


def ensure_env_file(dir_: Path, tag: str) -> None:
    env_file = dir_ / ".env"
    example = dir_ / ".env.example"
    if env_file.exists():
        return
    if example.exists():
        shutil.copy(example, env_file)
        _log(tag, f"no .env found — created one from .env.example. "
                   f"EDIT {env_file} with your real camera URL before trusting this run.")
    else:
        _log(tag, f"WARNING: no .env or .env.example found in {dir_}")


def run_setup(force: bool = False) -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import setup_assets  # noqa: E402

    _log("setup", "checking MediaMTX + face_landmarker.task...")
    try:
        setup_assets.ensure_mediamtx(force=force)
    except Exception as e:
        _log("setup", f"MediaMTX check failed: {e} — you'll need to fetch it "
                       f"manually into {MEDIAMTX_DIR}/ (or pass --skip-mediamtx "
                       f"if you already have it running elsewhere).")
    try:
        setup_assets.ensure_face_landmarker(force=force)
    except Exception as e:
        _log("setup", f"face_landmarker.task download failed: {e} — the backend "
                       f"will fail to start RealPersonFaceDetector until this file "
                       f"exists at backend/face_landmarker.task.")


def backend_deps_installed(python_exe: str) -> bool:
    """Cheap check: can this interpreter import the packages requirements.txt
    lists? Doesn't do a full pip-style resolution, just enough to catch the
    common case of running against a venv/interpreter that's never had
    `pip install -r requirements.txt` run against it (e.g. a fresh venv, or
    the system python vs. an intended venv)."""
    probe = "import cv2, numpy, ultralytics, mediapipe, websockets"
    result = subprocess.run([python_exe, "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def frontend_pkg_manager() -> list[str]:
    """Prefer bun (this repo ships a bun.lock); fall back to npm."""
    if (FRONTEND_DIR / "bun.lock").exists() and shutil.which("bun"):
        return ["bun"]
    return ["npm"]


def frontend_deps_installed() -> bool:
    return (FRONTEND_DIR / "node_modules").exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-setup", action="store_true", help="Skip the MediaMTX/model auto-download check")
    parser.add_argument("--force-setup", action="store_true", help="Re-download assets even if present")
    parser.add_argument("--skip-mediamtx", action="store_true")
    parser.add_argument("--skip-backend", action="store_true")
    parser.add_argument("--skip-frontend", action="store_true")
    args = parser.parse_args()

    if IS_WINDOWS:
        # So ANSI color codes actually render in cmd.exe / older PowerShell.
        os.system("")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ensure_env_file(BACKEND_DIR, "backend")
    ensure_env_file(FRONTEND_DIR, "frontend")

    if not args.skip_setup:
        run_setup(force=args.force_setup)

    if not shutil.which("ffmpeg"):
        _log("run_all", "WARNING: ffmpeg not found on PATH — the backend's RTSP "
                         "publish step (RTSPPublisher) will raise at startup without it.")

    # 1) MediaMTX
    if not args.skip_mediamtx:
        exe = MEDIAMTX_DIR / ("mediamtx.exe" if IS_WINDOWS else "mediamtx")
        if not exe.exists():
            _log("run_all", f"MediaMTX binary not found at {exe} — run "
                             f"`python scripts/setup_assets.py` first, or pass --skip-mediamtx "
                             f"if you're running it yourself.")
            sys.exit(1)
        _popen("mediamtx", [str(exe)], cwd=str(MEDIAMTX_DIR))
        time.sleep(1.5)  # give it time to bind :8554/:8888/:8889 before backend tries to publish

    # 2) Backend
    if not args.skip_backend:
        python_exe = sys.executable
        _log("run_all", f"using interpreter: {python_exe}")
        if not backend_deps_installed(python_exe):
            _log("run_all", "backend deps missing in this interpreter — running "
                             "`pip install -r requirements.txt` first...")
            install = subprocess.run(
                [python_exe, "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=str(BACKEND_DIR),
            )
            if install.returncode != 0:
                _log("run_all", "backend dependency install failed — see output above.")
                _shutdown()
                return
        _popen("backend", [python_exe, "stream_pipeline.py"], cwd=str(BACKEND_DIR))
        time.sleep(1.0)

    # 3) Frontend
    if not args.skip_frontend:
        pm = frontend_pkg_manager()
        if not frontend_deps_installed():
            _log("run_all", f"frontend/node_modules missing — running `{' '.join(pm)} install` first...")
            install = subprocess.run(pm + ["install"], cwd=str(FRONTEND_DIR))
            if install.returncode != 0:
                _log("run_all", "dependency install failed — see output above.")
                _shutdown()
                return
        dev_cmd = pm + (["run", "dev"] if pm[0] == "npm" else ["dev"])
        _popen("frontend", dev_cmd, cwd=str(FRONTEND_DIR))

    _log("run_all", "all requested components started. Ctrl+C to stop everything.")

    try:
        while True:
            time.sleep(0.5)
            for tag, proc in _procs:
                if proc.poll() is not None and not _shutting_down.is_set():
                    _log("run_all", f"'{tag}' exited on its own (code {proc.returncode}) — "
                                     f"stopping the rest of the stack.")
                    _shutdown()
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
