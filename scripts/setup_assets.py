#!/usr/bin/env python3
"""
setup_assets.py — fetches the binary/model files this repo needs but doesn't
ship in source control:

  1. MediaMTX standalone server  -> ./mediamtx/mediamtx(.exe)
     (the SETUP.md note said this came from a separate code.zip that wasn't
     part of the merged upload — this script re-fetches the right build for
     whatever OS/arch you're running on, straight from the project's GitHub
     releases.)
  2. mediapipe FaceLandmarker model bundle -> ./backend/face_landmarker.task
     (explicitly called out in stream_pipeline.py as "must be downloaded
     manually" — it's a ~5-6MB model file, not something pip/ultralytics
     fetches for you.)

yolo26m.pt is NOT handled here on purpose: RealPersonFaceDetector loads it
via ultralytics' YOLO(...), which auto-downloads the weights file itself the
first time the backend runs if it isn't already sitting next to the script.

Safe to re-run — every step is a no-op if the target file already exists,
unless you pass --force.
"""
import argparse
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEDIAMTX_DIR = ROOT / "mediamtx"
BACKEND_DIR = ROOT / "backend"

MEDIAMTX_REPO_API = "https://api.github.com/repos/bluenviron/mediamtx/releases/latest"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _log(msg: str) -> None:
    print(f"[setup_assets] {msg}", flush=True)


def _platform_tag() -> tuple[str, str, str]:
    """Returns (os_tag, arch_tag, archive_ext) matching mediamtx's release
    asset naming, e.g. mediamtx_v1.19.2_windows_amd64.zip"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        os_tag, ext = "windows", "zip"
    elif system == "darwin":
        os_tag, ext = "darwin", "tar.gz"
    elif system == "linux":
        os_tag, ext = "linux", "tar.gz"
    else:
        raise RuntimeError(f"Unsupported OS for MediaMTX auto-download: {system}")

    if machine in ("x86_64", "amd64"):
        arch_tag = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch_tag = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture for MediaMTX auto-download: {machine}")

    return os_tag, arch_tag, ext


def ensure_mediamtx(force: bool = False) -> Path:
    exe_name = "mediamtx.exe" if platform.system().lower() == "windows" else "mediamtx"
    exe_path = MEDIAMTX_DIR / exe_name

    if exe_path.exists() and not force:
        _log(f"MediaMTX already present at {exe_path}, skipping download.")
        return exe_path

    os_tag, arch_tag, ext = _platform_tag()
    _log(f"Detected platform: {os_tag}/{arch_tag}. Looking up latest MediaMTX release...")

    with urllib.request.urlopen(MEDIAMTX_REPO_API, timeout=20) as resp:
        import json
        release = json.loads(resp.read().decode())

    tag = release.get("tag_name", "unknown")
    wanted_suffix = f"_{os_tag}_{arch_tag}.{ext}"
    asset = next((a for a in release.get("assets", []) if a["name"].endswith(wanted_suffix)), None)
    if asset is None:
        raise RuntimeError(
            f"Could not find a MediaMTX {tag} asset ending in {wanted_suffix}. "
            f"Download manually from https://github.com/bluenviron/mediamtx/releases "
            f"and extract it into {MEDIAMTX_DIR}/"
        )

    _log(f"Downloading MediaMTX {tag} ({asset['name']})...")
    MEDIAMTX_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / asset["name"]
        urllib.request.urlretrieve(asset["browser_download_url"], archive_path)

        _log("Extracting...")
        if ext == "zip":
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(MEDIAMTX_DIR)
        else:
            with tarfile.open(archive_path) as tf:
                tf.extractall(MEDIAMTX_DIR)

    if not exe_path.exists():
        raise RuntimeError(
            f"Extracted MediaMTX but {exe_path} still doesn't exist — the "
            f"release layout may have changed. Check {MEDIAMTX_DIR}/ manually."
        )
    if platform.system().lower() != "windows":
        exe_path.chmod(exe_path.stat().st_mode | 0o111)  # ensure +x

    _log(f"MediaMTX ready at {exe_path}")
    return exe_path


def ensure_face_landmarker(force: bool = False) -> Path:
    dest = BACKEND_DIR / "face_landmarker.task"
    if dest.exists() and not force:
        _log(f"face_landmarker.task already present at {dest}, skipping download.")
        return dest

    _log("Downloading mediapipe FaceLandmarker model bundle (~6MB)...")
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".task.part")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp_dest)
    tmp_dest.rename(dest)
    _log(f"face_landmarker.task ready at {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    parser.add_argument("--skip-mediamtx", action="store_true")
    parser.add_argument("--skip-face-landmarker", action="store_true")
    args = parser.parse_args()

    try:
        if not args.skip_mediamtx:
            ensure_mediamtx(force=args.force)
        if not args.skip_face_landmarker:
            ensure_face_landmarker(force=args.force)
    except Exception as e:
        _log(f"ERROR: {e}")
        sys.exit(1)

    _log("Done. yolo26m.pt is not fetched here — ultralytics downloads it "
          "automatically the first time stream_pipeline.py runs, if it's "
          "not already sitting in backend/.")


if __name__ == "__main__":
    main()
