"""
RDAS — single-file prototype (drowsiness, yawning, no-driver, camera-covered).

Consolidates config.py + ear_mar.py + state_machine.py + perception.py +
report.py + pipeline.py + test_logic.py into one file for portability
(e.g. copying onto an edge device as a single artifact). The modular version
has clean import boundaries and per-module test isolation; this trades that
for a single file — see README.md for that tradeoff spelled out.

No CLI args. Edit RDASConfig below (or at the bottom, under MAIN), then:
    python3 final_pipeline.py
"""

import time
import threading
import subprocess
import os
import json
import asyncio
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np


def _load_dotenv(path: str) -> None:
    """Minimal, dependency-free .env loader. Populates os.environ from
    KEY=VALUE lines in `path`, skipping blanks/comments, and only setting a
    key if it isn't already present in the environment — same precedence
    rule python-dotenv uses, so a real `export FOO=bar` in your shell still
    wins over whatever's in the file.

    This didn't exist before: RDASConfig's fields all read from
    os.environ.get(...), but nothing in this script ever loaded backend/.env
    into os.environ — so the file was silently ignored no matter what was in
    it, and every field just fell back to its hardcoded default. Copying
    .env.example to .env did nothing without this.
    """
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class RDASConfig:
    # ---- Run settings ----
    # All of these can be overridden with environment variables instead of
    # editing this file directly (handy when backend + MediaMTX + frontend
    # are started from a script/compose file with different values per
    # environment). Falls back to the hardcoded default if the env var isn't set.
    input_video_path: str = field(default_factory=lambda: os.environ.get(
        "RDAS_INPUT_URL", "rtsp://admin:PASSWORD@CAMERA_IP:554/Streaming/Channels/101"))
    output_video_path: str = field(default_factory=lambda: os.environ.get(
        "RDAS_OUTPUT_URL", "rtsp://localhost:8554/rdas"))
    use_real_detector: bool = True  # False = scripted stub, no models needed. True = real YOLO26+mediapipe.
    yolo_weights_path: str = field(default_factory=lambda: os.environ.get(
        "RDAS_YOLO_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26m.pt")))
    face_task_path: str = field(default_factory=lambda: os.environ.get(
        "RDAS_FACE_TASK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")))
    print_report: bool = True   # end-of-run terminal stats summary
    run_self_tests: bool = False  # run offline logic tests before the pipeline
    draw_face_contours: bool = True
    # Continuous per-frame eye/mouth contour overlay on the main output video
    # (green convex hulls, like a typical dlib/mediapipe EAR-MAR demo shows) —
    # different from debug_visualize_frame_index below, which is an on-demand
    # single-frame verification dump, not a continuous video overlay.
    debug_visualize_frame_index: Optional[int] = None
    # If set, on that exact detector-call frame the pipeline saves:
    #   rdas_debug_full_frame.jpg  — YOLO box, ROI box, all landmark points, calibration state
    #   rdas_debug_roi_crop.jpg    — the ROI crop alone, upscaled 2x, same points redrawn
    #   rdas_debug_calculations.txt — plain-text EAR/MAR/pitch math, point by point
    # Verification tooling — leave None for normal runs. Only meaningful with
    # use_real_detector=True (the stub has no real box/ROI/landmarks to show).

    # ---- Camera / sampling ----
    camera_fps: int = 30
    max_alarm_latency_sec: float = 1.0
    # Max acceptable time from event start to alarm. Set to your real safety
    # requirement — sampling interval below is derived from this, not guessed.

    # ---- Eye closure (drowsiness, item 1) ----
    ear_drop_ratio: float = 0.75
    # "Closed" = live EAR < calibrated_baseline * this ratio. Relative to each
    # driver's own baseline, not a fixed global EAR value (eye shape varies).
    microsleep_continuous_closure_sec: float = 1.5  # continuous closure past this -> immediate alarm
    perclos_window_sec: float = 60.0
    perclos_threshold: float = 0.15  # fraction of rolling window spent closed -> fatigue-trend alarm
    perclos_min_window_fill_fraction: float = 0.5
    # Don't evaluate PERCLOS until the window is at least this fraction full —
    # otherwise a couple seconds of closure early in a session reads as a
    # misleadingly high percentage (caught in testing, see README).

    # ---- Yawning (item 2) ----
    mar_yawn_threshold: float = 0.6
    yawn_min_duration_sec: float = 1.2  # min sustained mouth-open duration to count as a yawn

    # ---- Distraction / head pose (item 3) ----
    distraction_pitch_ratio: float = 1.15
    # Alarm when live pitch_proxy exceeds calibrated_baseline_pitch * this ratio.
    # Grounded in real measurement (see head_pitch_proxy docstring): baseline
    # ~0.68, phone-use ~0.81 on the one video this was tuned against. 1.15
    # sits inside that separation with margin below the noise ceiling seen
    # at the boundary — but this is a single-video calibration, revisit with
    # more subjects/cameras before trusting it in production.
    distraction_min_duration_sec: float = 2.0
    # Longer than yawn_min_duration_sec on purpose — a glance at a gauge or
    # mirror is normal and brief; sustained multi-second downward gaze is
    # the target signal here, not every head movement.
    distraction_gap_tolerance_sec: float = 0.5
    # Real pitch_proxy data is noisy, not a clean step function — validated
    # on real footage, ~34% of frames within a confirmed 10s phone-use
    # window briefly dipped just under threshold (mostly 0.77-0.80 vs an
    # 0.808 threshold), which reset the duration timer before it could
    # accumulate. This tolerates brief dips without resetting accumulated
    # duration; only a gap longer than this counts as "no longer elevated."

    # ---- Calibration ----
    calibration_duration_sec: float = 8.0  # "eyes open, neutral face" window at shift start

    # ---- No-driver (item 6) ----
    no_driver_timeout_sec: float = 5.0

    # ---- Camera covered (item 7) ----
    brightness_threshold: float = 25.0   # mean pixel intensity, 0-255 scale
    variance_threshold: float = 15.0     # std-dev of pixel intensity
    # Both must be breached (dark AND flat) — brightness alone misfires on
    # tunnels/night (dark but textured) and reflective spoofs (bright but flat).

    # ---- Live stream input (RTSP/RTMP/HTTP — auto-detected by URL scheme) ----
    stream_reconnect_delay_sec: float = 2.0   # wait between reconnect attempts on a dropped feed
    stream_max_reconnect_attempts: int = -1   # -1 = retry forever (appropriate for an always-on monitor)
    stream_startup_timeout_sec: float = 10.0  # how long to wait for the first frame before giving up
    stream_stale_threshold_sec: float = 2.0
    # If no new frame has arrived in this long, the feed is treated as frozen,
    # not just slow — same treatment as camera_covered: detection is skipped
    # (a stale frame fed into the temporal state machines as if it were a new
    # observation would corrupt microsleep/PERCLOS/distraction timing, not
    # just look outdated) and it's surfaced as its own alarm, not silently
    # ignored. Irrelevant for file input — files don't go stale.

    # ---- Live stream OUTPUT (RTSP publish — auto-detected from output_video_path) ----
    rtsp_publish_preset: str = "ultrafast"   # libx264 preset — speed over compression for low latency
    rtsp_publish_bitrate: str = "2M"         # target bitrate; raise for sharper detail, lower for weak uplinks

    # ---- Structured alarm/telemetry side-channel (for the web HUD) ----
    # The RTSP output only carries a human-readable text overlay baked into
    # pixels — a browser frontend can't parse that back into discrete alarm
    # objects, driver-status widgets, or an event log. This WebSocket server
    # is the side-channel that carries the SAME alarm state as structured
    # JSON, in parallel with (not instead of) the video overlay.
    enable_ws_server: bool = True
    ws_host: str = field(default_factory=lambda: os.environ.get("RDAS_WS_HOST", "0.0.0.0"))
    ws_port: int = field(default_factory=lambda: int(os.environ.get("RDAS_WS_PORT", "8765")))
    ws_broadcast_interval_sec: float = 0.2   # ~5 Hz — plenty for a HUD, avoids flooding the UI thread
    # cv2.VideoWriter cannot publish RTSP — it only writes local files via an
    # FFmpeg muxer, it has no concept of serving a live stream to viewers.
    # When output_video_path starts with rtsp://, RTSPPublisher instead pipes
    # raw annotated frames into an ffmpeg subprocess, which encodes and
    # pushes them into an RTSP server that must ALREADY be running and
    # listening at that URL (e.g. MediaMTX) — this publishes INTO an
    # existing relay, it does not host one itself. Requires ffmpeg on PATH.

    def derive_sampling_interval(self) -> int:
        """Frames to skip between detector runs, derived from the latency
        budget: one third of the budget goes to subsampling delay, leaving
        two thirds for the confirmation/debounce logic downstream (the
        microsleep/yawn duration timers). This leans more of the total
        latency budget toward confirmation than a straight 50/50 split would —
        intentional if you'd rather catch events sooner and let the
        duration-based confirmation stages do more of the false-positive
        suppression, at the cost of ~50% more detector calls per second
        versus a /2.0 split at the same fps."""
        budget_for_subsampling = self.max_alarm_latency_sec / 3.0
        return max(1, int(budget_for_subsampling * self.camera_fps))


# ============================================================================
# EAR / MAR MATH
# ============================================================================
# CAVEAT: the indices below are the commonly-cited mapping used across public
# mediapipe EAR/MAR implementations, unverified here against a real model
# output (no model access in this sandbox this was built in). Visually check
# them against one real annotated frame before trusting the numbers — index
# mistakes fail silently, they just give you the wrong number, not an error.

# [corner, top1, top2, corner, bottom1, bottom2] per eye (Soukupová & Čech EAR)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Inner-lip top/bottom center + mouth corners
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308

# Fuller lip outline — drawing only, NOT used in mouth_aspect_ratio() above.
# A 4-point quad (the MOUTH_* indices) is enough for the MAR ratio but looks
# like a diamond, not a mouth, when drawn as a contour; this wider set gives
# a visually accurate lip outline instead. Same "commonly-cited mediapipe
# mapping, not independently re-verified" caveat as every other index here.
LIP_OUTLINE_IDX_FOR_DRAWING = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                                291, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def _dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))



def eye_aspect_ratio(landmarks_xy, eye_idx):
    """landmarks_xy: indexable by landmark id -> (x, y), consistent units."""
    p1, p2, p3, p4, p5, p6 = [landmarks_xy[i] for i in eye_idx]
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    return vertical / horizontal if horizontal else 0.0


def mouth_aspect_ratio(landmarks_xy):
    top, bottom = landmarks_xy[MOUTH_TOP_IDX], landmarks_xy[MOUTH_BOTTOM_IDX]
    left, right = landmarks_xy[MOUTH_LEFT_IDX], landmarks_xy[MOUTH_RIGHT_IDX]
    horizontal = _dist(left, right)
    return _dist(top, bottom) / horizontal if horizontal else 0.0


def both_eyes_ear(landmarks_xy):
    """Average of left/right EAR. Uses average, not min/max, so one noisy/
    occluded eye doesn't unilaterally trigger or suppress an alarm — revisit
    if your camera angle reliably occludes one eye (may want max() instead)."""
    left = eye_aspect_ratio(landmarks_xy, LEFT_EYE_IDX)
    right = eye_aspect_ratio(landmarks_xy, RIGHT_EYE_IDX)
    return (left + right) / 2.0


# ============================================================================
# HEAD PITCH PROXY — for item 3 (distraction), NOT used to gate EAR/MAR
# ============================================================================
# Deliberately 2D-only (no landmark z-coordinates), reusing the same eye
# corner points already used for EAR rather than introducing another
# unverified dependency on top of the landmark-index caveat above.
#
# This is a custom heuristic, not a peer-reviewed metric like EAR/MAR — it
# was built and threshold-calibrated in this project against ONE real video
# (see README, "Distraction detection" section) using the same legacy-
# mediapipe bridge described there. Measured on that footage: baseline
# (neutral) ~0.68, during confirmed phone-use ~0.81 — a real, measured ~20%
# separation, not a guess — but n=1 video, one subject, one camera angle.
# Needs revalidation on more footage before trusting the default threshold.

NOSE_TIP_IDX = 1  # standard mediapipe nose tip; same unverified-indices caveat as above


def head_pitch_proxy(landmarks_xy):
    """Rises as the head pitches down: nose drops toward/past the eye line
    relative to inter-eye distance (used as a scale reference so this is
    roughly independent of how close the driver sits to the camera)."""
    nose = landmarks_xy[NOSE_TIP_IDX]
    left_c = _midpoint(landmarks_xy[LEFT_EYE_IDX[0]], landmarks_xy[LEFT_EYE_IDX[3]])
    right_c = _midpoint(landmarks_xy[RIGHT_EYE_IDX[0]], landmarks_xy[RIGHT_EYE_IDX[3]])
    eye_line_y = (left_c[1] + right_c[1]) / 2.0
    inter_eye = _dist(left_c, right_c)
    return (nose[1] - eye_line_y) / inter_eye if inter_eye else 0.0


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


# ============================================================================
# STATE MACHINE — turns per-frame EAR/MAR into duration-based alarms
# ============================================================================
# Drowsiness/yawning are durations, not instants; a single closed-eye frame
# is indistinguishable from a blink. All "time" args are seconds (video
# timestamp or wall-clock, caller's choice).

@dataclass
class CalibrationTracker:
    cfg: RDASConfig
    _samples: list = field(default_factory=list)
    _start_t: Optional[float] = None
    baseline_ear: Optional[float] = None

    def update(self, t: float, ear: float) -> bool:
        """Feed one EAR sample during the calibration window.
        Returns True once calibration is complete."""
        if self.baseline_ear is not None:
            return True
        if self._start_t is None:
            self._start_t = t
        self._samples.append(ear)
        if t - self._start_t >= self.cfg.calibration_duration_sec:
            s = sorted(self._samples)  # median — robust to blinks in the window
            self.baseline_ear = s[len(s) // 2]
            return True
        return False

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_ear is not None


@dataclass
class DrowsinessStateMachine:
    cfg: RDASConfig
    calibration: CalibrationTracker

    _closure_start_t: Optional[float] = None
    _history: deque = field(default_factory=deque)  # (t, is_closed)
    microsleep_alarm: bool = False
    perclos_alarm: bool = False
    current_ear: Optional[float] = None
    is_closed: bool = False

    def _closure_threshold(self) -> float:
        return self.calibration.baseline_ear * self.cfg.ear_drop_ratio

    def update(self, t: float, ear: float):
        self.current_ear = ear

        if not self.calibration.is_calibrated:
            self.calibration.update(t, ear)
            return

        self.is_closed = ear < self._closure_threshold()

        # --- continuous-closure / microsleep check ---
        if self.is_closed:
            if self._closure_start_t is None:
                self._closure_start_t = t
            closed_duration = t - self._closure_start_t
            self.microsleep_alarm = closed_duration >= self.cfg.microsleep_continuous_closure_sec
        else:
            self._closure_start_t = None
            self.microsleep_alarm = False

        # --- PERCLOS rolling-window check ---
        self._history.append((t, self.is_closed))
        while self._history and t - self._history[0][0] > self.cfg.perclos_window_sec:
            self._history.popleft()
        history_span = self._history[-1][0] - self._history[0][0] if self._history else 0.0
        min_span_required = self.cfg.perclos_window_sec * self.cfg.perclos_min_window_fill_fraction
        if history_span >= min_span_required:
            closed_time = sum(1 for _, c in self._history if c) / len(self._history)
            self.perclos_alarm = closed_time >= self.cfg.perclos_threshold
        else:
            self.perclos_alarm = False  # window not populated enough yet, don't evaluate

    @property
    def any_alarm(self) -> bool:
        return self.microsleep_alarm or self.perclos_alarm


@dataclass
class YawnStateMachine:
    cfg: RDASConfig
    _open_start_t: Optional[float] = None
    yawn_alarm: bool = False
    yawn_count: int = 0
    _counted_this_yawn: bool = False
    current_mar: Optional[float] = None

    def update(self, t: float, mar: float):
        self.current_mar = mar
        is_open = mar > self.cfg.mar_yawn_threshold

        if is_open:
            if self._open_start_t is None:
                self._open_start_t = t
                self._counted_this_yawn = False
            duration = t - self._open_start_t
            self.yawn_alarm = duration >= self.cfg.yawn_min_duration_sec
            if self.yawn_alarm and not self._counted_this_yawn:
                self.yawn_count += 1  # count once per sustained open-stretch, not per frame
                self._counted_this_yawn = True
        else:
            self._open_start_t = None
            self.yawn_alarm = False


@dataclass
class PitchCalibrationTracker:
    """Mirrors CalibrationTracker's pattern exactly (median over a fixed
    warm-up window). Kept as a separate small class rather than generalizing
    CalibrationTracker into a shared multi-signal calibrator — CalibrationTracker
    is already covered by passing tests and cross-validated against real
    footage twice; duplicating ~15 lines here is a smaller risk than touching
    proven code to make it generic for one new caller."""
    cfg: RDASConfig
    _samples: list = field(default_factory=list)
    _start_t: Optional[float] = None
    baseline_pitch: Optional[float] = None

    def update(self, t: float, pitch: float) -> bool:
        if self.baseline_pitch is not None:
            return True
        if self._start_t is None:
            self._start_t = t
        self._samples.append(pitch)
        if t - self._start_t >= self.cfg.calibration_duration_sec:
            s = sorted(self._samples)
            self.baseline_pitch = s[len(s) // 2]
            return True
        return False

    @property
    def is_calibrated(self) -> bool:
        return self.baseline_pitch is not None


@dataclass
class DistractionStateMachine:
    """Item 3. Deliberately independent of DrowsinessStateMachine — this
    reports distraction as its own alarm rather than gating/suppressing the
    EAR-based microsleep/PERCLOS signal. A hard gate ("ignore EAR when pitch
    is high") would also blind the system to genuine fatigue head-droop,
    which is itself a real drowsiness symptom, not just a distraction
    artifact — so gating would trade one false positive for a worse false
    negative. Both alarms are meant to fire independently and let whatever
    consumes this system's output (log, reviewer, downstream logic)
    reconcile them, rather than one silently overriding the other here."""
    cfg: RDASConfig
    calibration: PitchCalibrationTracker
    _high_start_t: Optional[float] = None
    _last_high_t: Optional[float] = None
    distraction_alarm: bool = False
    distraction_count: int = 0
    _counted_this_event: bool = False
    current_pitch: Optional[float] = None

    def update(self, t: float, pitch: float):
        self.current_pitch = pitch

        if not self.calibration.is_calibrated:
            self.calibration.update(t, pitch)
            return

        threshold = self.calibration.baseline_pitch * self.cfg.distraction_pitch_ratio
        is_high = pitch > threshold

        if is_high:
            if self._high_start_t is None:
                self._high_start_t = t
                self._counted_this_event = False
            self._last_high_t = t
            duration = t - self._high_start_t
            self.distraction_alarm = duration >= self.cfg.distraction_min_duration_sec
            if self.distraction_alarm and not self._counted_this_event:
                self.distraction_count += 1
                self._counted_this_event = True
        elif self._high_start_t is not None:
            # Below threshold, but mid-episode: only reset if the gap since
            # the last high reading exceeds tolerance. Otherwise hold current
            # state (don't advance duration, don't reset it either) — a
            # single noisy sample shouldn't erase several seconds of
            # already-accumulated evidence.
            gap = t - self._last_high_t
            if gap > self.cfg.distraction_gap_tolerance_sec:
                self._high_start_t = None
                self.distraction_alarm = False
        else:
            self.distraction_alarm = False


@dataclass
class NoDriverTracker:
    cfg: RDASConfig
    _last_seen_t: Optional[float] = None
    no_driver_alarm: bool = False

    def update(self, t: float, person_detected: bool):
        if person_detected:
            self._last_seen_t = t
            self.no_driver_alarm = False
            return
        if self._last_seen_t is None:
            self._last_seen_t = t  # first frame ever — don't alarm instantly
            return
        self.no_driver_alarm = (t - self._last_seen_t) >= self.cfg.no_driver_timeout_sec


def camera_covered(mean_brightness: float, std_brightness: float, cfg: RDASConfig) -> bool:
    """Dark AND flat required — brightness alone misfires on tunnels/night
    (dark but textured) and reflective spoofs (bright but flat)."""
    return mean_brightness < cfg.brightness_threshold and std_brightness < cfg.variance_threshold


def stream_stale(seconds_since_last_frame: float, cfg: RDASConfig) -> bool:
    """Same pattern as camera_covered(): a simple threshold check, kept as
    its own function so the state machine layer can treat 'feed frozen' as
    a first-class condition rather than an ad-hoc check buried in run()."""
    return seconds_since_last_frame > cfg.stream_stale_threshold_sec


# ============================================================================
# STREAMING INPUT — live RTSP/RTMP/HTTP sources, auto-detected by URL scheme
# ============================================================================
# A file has a defined end and a fixed frame sequence; a live feed has
# neither. Wrapping cv2.VideoCapture directly in run()'s loop (as file input
# does) would mean: (a) a dropped connection looks identical to "end of
# file" and kills the pipeline instead of reconnecting, and (b) if
# processing (YOLO+mediapipe) is ever slower than the camera's frame rate,
# OpenCV's own internal buffer backs up and .read() starts silently serving
# older and older frames — the output drifts further behind real time
# forever, not just occasionally. LiveStreamReader runs a background thread
# that always keeps only the MOST RECENT frame, discarding anything older —
# trading "process every frame" for "never fall behind," which is the right
# tradeoff for a live alarm system: whether the driver's eyes are closed
# RIGHT NOW matters more than not missing a single frame.
#
# For a file path (no rtsp://, rtmp://, http://, https:// prefix), this
# class changes nothing — .read()/.get()/.release() all just forward
# directly to a plain cv2.VideoCapture, identical to file behavior
# throughout the rest of this project. The threading/reconnect/staleness
# machinery only activates for detected live sources.

class LiveStreamReader:
    def __init__(self, source: str, cfg: RDASConfig):
        self.source = source
        self.cfg = cfg
        self.is_stream = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open input source: {source}")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_ok = False
        self._last_frame_time = None  # monotonic timestamp of last successful read
        self._frame_ready = threading.Event()
        # .read() blocks on this instead of returning immediately with
        # whatever's cached — without it, a consumer loop that's faster than
        # the camera's actual delivery rate just spins, re-reading and
        # re-processing the SAME frame many times before a genuinely new one
        # arrives. Confirmed happening in practice, not a theoretical
        # concern: a 15fps source produced ~113 "frames processed" per
        # second in one real run, ~7.5x oversampling, each duplicate costing
        # a full YOLO+mediapipe pass on identical pixel data for nothing.
        self._stopped = False
        self._reconnect_count = 0
        self._thread = None

        if self.is_stream:
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
            start = time.monotonic()
            while self._latest_frame is None and time.monotonic() - start < cfg.stream_startup_timeout_sec:
                time.sleep(0.05)
            if self._latest_frame is None:
                self._stopped = True
                raise RuntimeError(
                    f"Could not read any frame from {source} within "
                    f"{cfg.stream_startup_timeout_sec}s — check the URL, network route, "
                    f"and camera credentials."
                )
            print(f"[stream] connected: {source}")

    def _reconnect(self):
        self._reconnect_count += 1
        print(f"[stream] connection lost, reconnecting (attempt {self._reconnect_count})...")
        self._cap.release()
        time.sleep(self.cfg.stream_reconnect_delay_sec)
        self._cap = cv2.VideoCapture(self.source)

    def _reader_loop(self):
        while not self._stopped:
            ok, frame = self._cap.read()
            if not ok:
                limit = self.cfg.stream_max_reconnect_attempts
                if limit >= 0 and self._reconnect_count >= limit:
                    print(f"[stream] gave up after {self._reconnect_count} reconnect attempts")
                    with self._lock:
                        self._latest_ok = False
                    break
                self._reconnect()
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_ok = True
                self._last_frame_time = time.monotonic()
            self._frame_ready.set()

    def read(self):
        if not self.is_stream:
            return self._cap.read()  # unchanged file behavior
        # Block until a genuinely NEW frame arrives, paced to the source's
        # real delivery rate — this is what stops the consumer loop from
        # outrunning the camera and reprocessing duplicates. Timeout matches
        # stream_stale_threshold_sec: if nothing new shows up in that long,
        # return anyway with whatever's cached so the caller's own staleness
        # check (via seconds_since_last_frame()) can fire on the next
        # iteration — this must never hang indefinitely just because the
        # camera stalled.
        got_new = self._frame_ready.wait(timeout=self.cfg.stream_stale_threshold_sec)
        with self._lock:
            if self._latest_frame is None:
                return False, None
            if got_new:
                self._frame_ready.clear()
            return self._latest_ok, self._latest_frame.copy()

    def seconds_since_last_frame(self) -> float:
        """0.0 for file input (files don't go stale). For streams, time since
        the background thread's last successful read — this is what feeds
        stream_stale(), separate from whether the connection is technically
        still open (a connection can stay open while delivering nothing)."""
        if not self.is_stream or self._last_frame_time is None:
            return 0.0
        return time.monotonic() - self._last_frame_time

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def release(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cap.release()


# ============================================================================
# STREAMING OUTPUT — RTSP publish, auto-detected from output_video_path
# ============================================================================
# cv2.VideoWriter writes a local file via an FFmpeg muxer; it has no concept
# of serving a live stream to viewers. Publishing real RTSP means piping raw
# annotated frames into an ffmpeg subprocess, which encodes and pushes them
# into an RTSP server that must ALREADY be running at the target URL — this
# class publishes INTO an existing relay (e.g. MediaMTX), it does not host
# one. Piping mechanism validated in this project against a local ffmpeg
# instance (raw BGR frames -> subprocess stdin -> libx264 -> valid output,
# frame order and content confirmed intact) — the network-publish step
# itself uses standard, widely-documented ffmpeg RTSP output syntax but
# could not be tested against a live RTSP server in the sandbox this was
# built in (see README for why). Verify your first real run against your
# actual relay rather than assuming this is bug-free out of the box.
#
# For a local file path (no rtsp:// prefix), this is a thin, unchanged
# pass-through to cv2.VideoWriter — identical to every prior file-output run
# in this project.

class RTSPPublisher:
    def __init__(self, output_path: str, fps: float, width: int, height: int, cfg: RDASConfig):
        self.output_path = output_path
        self.is_stream = output_path.lower().startswith("rtsp://")
        self._writer = None
        self._proc = None
        self._stderr_thread = None

        if not self.is_stream:
            self._writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            return

        # -g/-keyint_min/-sc_threshold: without an explicit keyframe interval,
        # libx264 defaults to a ~250-frame GOP — at a 15fps source that's a
        # keyframe roughly every 16s. MediaMTX can't start an HLS segment
        # without a keyframe, so the practical symptom is "no video shows up
        # in the browser" for a long time (or effectively never, if segment
        # settings expect something shorter). Forcing a keyframe every 2s
        # (fps*2 frames), with scene-cut-triggered extra keyframes disabled
        # so the interval stays regular, fixes HLS startup time.
        gop = max(1, int(round(fps * 2)))
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", cfg.rtsp_publish_preset, "-tune", "zerolatency",
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-b:v", cfg.rtsp_publish_bitrate,
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", "-rtsp_transport", "tcp",
            output_path,
        ]
        try:
            # stderr used to be DEVNULL — if ffmpeg failed to connect, encode,
            # or publish, that failure was completely invisible. Piping it and
            # echoing it via a background thread means a broken publish now
            # actually shows up in the console instead of just "no video, no
            # explanation why."
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg not found on PATH — required to publish RTSP output. "
                "cv2.VideoWriter cannot do this on its own; install ffmpeg and "
                "ensure it's callable from a terminal, then retry."
            )

        def _echo_stderr(proc):
            assert proc.stderr is not None
            for raw_line in iter(proc.stderr.readline, b""):
                if not raw_line:
                    break
                print(f"[rtsp-out][ffmpeg] {raw_line.decode(errors='replace').rstrip()}")

        self._stderr_thread = threading.Thread(target=_echo_stderr, args=(self._proc,), daemon=True)
        self._stderr_thread.start()

        print(f"[rtsp-out] publishing to {output_path} via ffmpeg (pid={self._proc.pid}) — "
              f"make sure an RTSP server is already listening at that URL, or this will "
              f"connect-fail silently into a broken pipe on the first frame written")

    def write(self, frame_bgr: np.ndarray):
        if not self.is_stream:
            self._writer.write(frame_bgr)
            return
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, OSError) as e:
            # Deliberately not raised — a dead publish target shouldn't crash
            # the whole monitoring pipeline (detection/alarms/logging should
            # keep running even if nobody's currently watching the stream).
            # Printed on every failure rather than rate-limited: a broken
            # RTSP output is exactly the kind of thing that should be loud,
            # not swallowed into an easy-to-miss occasional log line.
            print(f"[rtsp-out] WARNING: write failed ({e}) — is the RTSP server still running?")

    def release(self):
        if not self.is_stream:
            self._writer.release()
            return
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()



# RealPersonFaceDetector: actual YOLO26 (person bbox) + mediapipe
#   FaceLandmarker (468 landmarks). Needs model weight files — see README.
# StubPersonFaceDetector: scripted fake output, no models needed. Validates
#   the video I/O + state-machine + overlay plumbing, not perception accuracy.
# Both return a PerceptionResult so the pipeline doesn't care which is in use.

@dataclass
class PerceptionResult:
    person_detected: bool
    face_landmarks_xy: Optional[Dict[int, Tuple[float, float]]]
    ear: Optional[float]
    mar: Optional[float]
    pitch_proxy: Optional[float] = None


class RealPersonFaceDetector:
    """Setup (see README for full instructions):
    1. YOLO26 weights ('yolo26m.pt') — auto-downloads via ultralytics on
       first use, or pass a local path.
    2. mediapipe FaceLandmarker .task bundle — must be downloaded manually,
       pass its local path via face_task_path.
    """

    PERSON_CLASS_ID = 0  # COCO 'person'

    def __init__(self, yolo_weights_path: str = "D:/IPVSS/AI_BOX/yolo26m.pt",
                 face_task_path: str = "D:/IPVSS/AI_BOX/RDAS/face_landmarker.task",
                 yolo_conf: float = 0.4):
        from ultralytics import YOLO
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._yolo = YOLO(yolo_weights_path)
        self._yolo_conf = yolo_conf

        base_options = mp_python.BaseOptions(model_asset_path=face_task_path)
        options = mp_vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        self._calls = 0
        self._misses = 0
        self._yolo_time_total = 0.0
        self._landmark_time_total = 0.0
        # Debug-only, for draw_debug_visualization() — populated each call so
        # a specific frame's full detection context (box, crop bounds, raw
        # landmarks) can be visualized after the fact from run(). None on
        # any call where that stage didn't succeed (no person, empty ROI, etc).
        self.last_yolo_box = None
        self.last_roi_bounds = None
        self.last_landmarks_xy = None

    def detect(self, frame_bgr: np.ndarray, t: float = None) -> PerceptionResult:
        # t unused here — kept for interface parity with StubPersonFaceDetector
        h, w = frame_bgr.shape[:2]
        self.last_yolo_box = None
        self.last_roi_bounds = None
        self.last_landmarks_xy = None

        _yolo_start = time.time()
        yolo_results = self._yolo(frame_bgr, classes=[self.PERSON_CLASS_ID],
                                   conf=self._yolo_conf, verbose=False)
        self._yolo_time_total += time.time() - _yolo_start
        boxes = yolo_results[0].boxes
        person_detected = boxes is not None and len(boxes) > 0
        if not person_detected:
            return PerceptionResult(False, None, None, None)

        # Largest box = closest to camera (driver seat). Crop upper ~60% as
        # the face ROI — mediapipe finds the face within it.
        areas = [(b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]) for b in boxes]
        biggest = boxes[int(np.argmax([a.item() for a in areas]))]
        x1, y1, x2, y2 = [int(v) for v in biggest.xyxy[0]]
        roi_y2 = min(h, y1 + int((y2 - y1) * 0.6))
        roi_x1, roi_y1, roi_x2 = max(0, x1), max(0, y1), min(w, x2)
        roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        self.last_yolo_box = (x1, y1, x2, y2)  # full-frame coords, as YOLO returned it
        self.last_roi_bounds = (roi_x1, roi_y1, roi_x2, roi_y2)  # full-frame coords, the actual crop used
        if roi.size == 0:
            return PerceptionResult(True, None, None, None)
        if self._calls == 0:
            cv2.imwrite("rdas_debug_first_roi.jpg", roi)
            print(f"[RealPersonFaceDetector] saved first ROI (shape={roi.shape}) to "
                  f"rdas_debug_first_roi.jpg — open it and check a face is actually visible")

        # cv2.cvtColor returns a proper C-contiguous array; roi[:,:,::-1] does
        # NOT (it's a reversed-stride view) and mp.Image can silently fail on
        # non-contiguous input — this was very likely why every detector call
        # found zero faces despite YOLO consistently finding the person.
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        _lm_start = time.time()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_roi)
        result = self._face_landmarker.detect(mp_image)
        self._landmark_time_total += time.time() - _lm_start
        self._calls += 1
        if not result.face_landmarks:
            self._misses += 1
            if self._calls <= 20 or self._calls % 50 == 0:
                print(f"[RealPersonFaceDetector] call {self._calls}: NO FACE FOUND "
                      f"in ROI shape={roi.shape} (misses so far: {self._misses}/{self._calls})")
            return PerceptionResult(True, None, None, None)

        lm = result.face_landmarks[0]
        roi_h, roi_w = roi.shape[:2]
        landmarks_xy = {i: (pt.x * roi_w, pt.y * roi_h) for i, pt in enumerate(lm)}
        self.last_landmarks_xy = landmarks_xy  # ROI-local coords — see draw_debug_visualization()
        ear = both_eyes_ear(landmarks_xy)
        mar = mouth_aspect_ratio(landmarks_xy)
        pitch = head_pitch_proxy(landmarks_xy)
        return PerceptionResult(True, landmarks_xy, ear, mar, pitch)


class StubPersonFaceDetector:
    """Deterministic fake detector driven by elapsed time — lets the pipeline
    run end-to-end with no model weights. Schedule (seconds since start):
      0-8    person present, eyes open           (calibration window)
      11-11.2 quick blink
      14-16  sustained closure -> microsleep alarm
      16.5-18.7 sustained downward pitch -> distraction alarm
      19-21  sustained yawn -> yawn alarm
      21-27  person missing -> no-driver alarm after timeout
    Baseline/elevated pitch values (0.68 / 0.81) are not arbitrary — they're
    the real neutral vs. phone-use pitch_proxy values measured on actual
    footage during this project (see head_pitch_proxy docstring), so the
    stub's distraction scenario behaves like the real one it's standing in for.
    """

    def detect(self, frame_bgr: np.ndarray, t: float) -> PerceptionResult:
        OPEN_EAR, CLOSED_EAR = 0.30, 0.10
        CLOSED_MAR, YAWN_MAR = 0.15, 0.80
        BASELINE_PITCH, HIGH_PITCH = 0.68, 0.81

        if 21.0 <= t < 27.0:
            return PerceptionResult(False, None, None, None, None)

        ear = CLOSED_EAR if (14.0 <= t < 16.0 or 11.0 <= t < 11.2) else OPEN_EAR
        mar = YAWN_MAR if 19.0 <= t < 21.0 else CLOSED_MAR
        pitch = HIGH_PITCH if 16.5 <= t < 18.7 else BASELINE_PITCH
        fake_landmarks = {i: (0.0, 0.0) for i in range(468)}  # placeholder, unused downstream
        return PerceptionResult(True, fake_landmarks, ear, mar, pitch)


def frame_brightness_stats(frame_bgr: np.ndarray) -> Tuple[float, float]:
    """Mean and std of pixel intensity, feeding camera_covered().

    Was frame_bgr.mean(axis=2) + .mean() + .std() — silently upcasts to
    float64 and does three separate full-frame passes over a large
    intermediate array. Measured at 55.41 ms/frame on real 1920x1080 cabin
    footage — 56.7s of a 94s total pipeline run, the single largest cost in
    the whole system, more expensive than the actual YOLO detection call.
    cv2.cvtColor + cv2.meanStdDev does both stats in one optimized call and
    stays off the float64 upcast: 1.23 ms/frame on the same footage, a 45x
    speedup, benchmarked not assumed.

    Not a byte-identical swap: cv2's grayscale uses luminance weighting
    (0.299R + 0.587G + 0.114B), the old code used a flat unweighted channel
    average. Values differ slightly (measured ~0.02% on brightness, ~2.5% on
    std on real footage) — close enough to not matter in practice, but this
    is a semantic change worth knowing about, not a free optimization."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    return float(mean[0][0]), float(std[0][0])


# ============================================================================
# DEBUG VISUALIZATION — verification tooling, not used in the main run() loop
# ============================================================================
# Draws every piece of geometry the pipeline actually uses onto real frames,
# for verifying the pipeline is doing what it claims rather than trusting it
# blindly: the YOLO box, the ROI cropped from it, every individual landmark
# point EAR/MAR/pitch are computed from (labeled, not just dots), and the
# calibration state. Calculations are deliberately NOT drawn here — see
# format_calculations_text() below, kept as plain text by design so the
# image stays readable and the numbers stay exact/copyable.
#
# Colors are BGR (OpenCV convention): left eye points blue, right eye yellow,
# mouth magenta, nose/derived-pitch-points green, YOLO box orange, ROI box white.

_DEBUG_COLOR_LEFT_EYE = (255, 0, 0)
_DEBUG_COLOR_RIGHT_EYE = (0, 255, 255)
_DEBUG_COLOR_MOUTH = (255, 0, 255)
_DEBUG_COLOR_PITCH = (0, 255, 0)
_DEBUG_COLOR_YOLO_BOX = (0, 165, 255)
_DEBUG_COLOR_ROI_BOX = (255, 255, 255)


def _offset_point(p, ox, oy):
    return (int(p[0] + ox), int(p[1] + oy))


def draw_landmark_contours(frame_bgr, roi_bounds, landmarks_xy, color=(0, 255, 0)):
    """Draws green convex-hull contours over both eyes and the mouth onto
    frame_bgr IN PLACE (mutates the frame, same convention as the main
    overlay loop's cv2.putText calls) — for continuous per-frame visual
    verification in the main output video, not a one-off debug capture.
    No-op if either input is None (nothing detected this cycle, or a stub
    detector with no real landmarks). Offsets ROI-local landmark coordinates
    onto full-frame coordinates using roi_bounds' top-left corner — skipping
    this offset would draw the contours in the wrong place entirely, since
    RealPersonFaceDetector's landmarks are ROI-local, not full-frame."""
    if roi_bounds is None or landmarks_xy is None:
        return
    ox, oy = roi_bounds[0], roi_bounds[1]

    def hull_points(indices):
        pts = [_offset_point(landmarks_xy[i], ox, oy) for i in indices if i in landmarks_xy]
        return np.array(pts, dtype=np.int32) if pts else None

    for indices in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
        pts = hull_points(indices)
        if pts is not None and len(pts) >= 3:
            cv2.drawContours(frame_bgr, [cv2.convexHull(pts)], -1, color, 1)

    lip_pts = hull_points(LIP_OUTLINE_IDX_FOR_DRAWING)
    if lip_pts is not None and len(lip_pts) >= 3:
        cv2.drawContours(frame_bgr, [cv2.convexHull(lip_pts)], -1, color, 1)


def draw_debug_visualization(frame_bgr, yolo_box, roi_bounds, landmarks_xy,
                              ear_calibrated: bool, baseline_ear: Optional[float],
                              pitch_calibrated: bool, baseline_pitch: Optional[float],
                              ear_value: Optional[float], mar_value: Optional[float],
                              pitch_value: Optional[float]):
    """Returns (annotated_full_frame, annotated_roi_crop) — two separate
    images, both copies (originals untouched). annotated_roi_crop is the
    actual cropped pixels, upscaled 2x for visibility, with the same points
    redrawn at the correspondingly scaled coordinates.

    Any of yolo_box/roi_bounds/landmarks_xy may be None (e.g. nothing was
    detected this frame, or you're inspecting a StubPersonFaceDetector run,
    which has no real box/ROI/landmarks to show) — everything that can still
    be drawn is drawn; missing pieces are simply skipped, not faked.
    """
    annotated = frame_bgr.copy()
    roi_crop_annotated = None

    if yolo_box is not None:
        x1, y1, x2, y2 = yolo_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _DEBUG_COLOR_YOLO_BOX, 2)
        cv2.putText(annotated, "YOLO PERSON BOX", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DEBUG_COLOR_YOLO_BOX, 2)

    if roi_bounds is not None:
        rx1, ry1, rx2, ry2 = roi_bounds
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), _DEBUG_COLOR_ROI_BOX, 2)
        cv2.putText(annotated, "FACE ROI", (rx1, min(annotated.shape[0] - 5, ry2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _DEBUG_COLOR_ROI_BOX, 2)

        roi_crop = frame_bgr[ry1:ry2, rx1:rx2]
        if roi_crop.size:
            scale = 2
            roi_crop_annotated = cv2.resize(roi_crop, None, fx=scale, fy=scale,
                                             interpolation=cv2.INTER_LINEAR)
        ox, oy = rx1, ry1  # landmark coords below are ROI-local; offset onto full frame
    else:
        ox, oy = 0, 0

    def _draw_point_set(indices, color, label_prefix, target_img, target_scale=1):
        if landmarks_xy is None:
            return
        for n, idx in enumerate(indices, start=1):
            if idx not in landmarks_xy:
                continue
            p_local = landmarks_xy[idx]
            if target_img is annotated:
                p = _offset_point(p_local, ox, oy)
            else:  # drawing on the upscaled ROI crop — no full-frame offset, just scale
                p = (int(p_local[0] * target_scale), int(p_local[1] * target_scale))
            cv2.circle(target_img, p, 3, color, -1)
            cv2.putText(target_img, f"{label_prefix}{n}", (p[0] + 4, p[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for target_img, scale in [(annotated, 1)] + ([(roi_crop_annotated, 2)] if roi_crop_annotated is not None else []):
        _draw_point_set(LEFT_EYE_IDX, _DEBUG_COLOR_LEFT_EYE, "L", target_img, scale)
        _draw_point_set(RIGHT_EYE_IDX, _DEBUG_COLOR_RIGHT_EYE, "R", target_img, scale)
        _draw_point_set([MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX],
                         _DEBUG_COLOR_MOUTH, "M", target_img, scale)
        _draw_point_set([NOSE_TIP_IDX], _DEBUG_COLOR_PITCH, "NOSE", target_img, scale)
        # Eye-center midpoints are DERIVED (used only for pitch), not raw
        # landmarks — drawn as an 'x' cross, not a filled dot, to visually
        # distinguish "computed from points above" from "a raw point itself".
        if landmarks_xy is not None:
            for idx_pair, tag in [((LEFT_EYE_IDX[0], LEFT_EYE_IDX[3]), "L-EYE-C"),
                                   ((RIGHT_EYE_IDX[0], RIGHT_EYE_IDX[3]), "R-EYE-C")]:
                if idx_pair[0] in landmarks_xy and idx_pair[1] in landmarks_xy:
                    mid = _midpoint(landmarks_xy[idx_pair[0]], landmarks_xy[idx_pair[1]])
                    p = _offset_point(mid, ox, oy) if target_img is annotated else \
                        (int(mid[0] * scale), int(mid[1] * scale))
                    size = 5
                    cv2.line(target_img, (p[0] - size, p[1] - size), (p[0] + size, p[1] + size),
                              _DEBUG_COLOR_PITCH, 1)
                    cv2.line(target_img, (p[0] - size, p[1] + size), (p[0] + size, p[1] - size),
                              _DEBUG_COLOR_PITCH, 1)
                    cv2.putText(target_img, tag, (p[0] + 6, p[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _DEBUG_COLOR_PITCH, 1)

    # --- calibration annotation ---
    cal_y = 25
    for label, calibrated, baseline, live in [
        ("EAR", ear_calibrated, baseline_ear, ear_value),
        ("PITCH", pitch_calibrated, baseline_pitch, pitch_value),
    ]:
        if calibrated and baseline is not None:
            text = f"{label} BASELINE={baseline:.3f} [CALIBRATED]"
            color = (0, 255, 0)
        else:
            text = f"{label}: CALIBRATING..."
            color = (0, 140, 255)
        cv2.putText(annotated, text, (10, cal_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cal_y += 25
    live_text = (f"live: EAR={ear_value:.3f}  MAR={mar_value:.3f}  PITCH={pitch_value:.3f}"
                 if all(v is not None for v in (ear_value, mar_value, pitch_value))
                 else "live: -- (no face resolved this frame)")
    cv2.putText(annotated, live_text, (10, cal_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated, roi_crop_annotated


def format_calculations_text(landmarks_xy, ear_value, mar_value, pitch_value) -> str:
    """Plain-text, point-by-point breakdown of exactly how EAR/MAR/pitch were
    computed — deliberately not drawn onto any image. Returns a multi-line
    string; print it or write it to a file."""
    if landmarks_xy is None:
        return "No landmarks available for this frame (no face resolved)."

    lines = []
    lines.append("=" * 60)
    lines.append("CALCULATIONS")
    lines.append("=" * 60)

    for eye_name, idx in [("LEFT EYE", LEFT_EYE_IDX), ("RIGHT EYE", RIGHT_EYE_IDX)]:
        p = [landmarks_xy[i] for i in idx]
        vert = _dist(p[1], p[5]) + _dist(p[2], p[4])
        horiz = 2.0 * _dist(p[0], p[3])
        this_ear = vert / horiz if horiz else 0.0
        lines.append(f"\n{eye_name} EAR:")
        for n, pt in enumerate(p, start=1):
            lines.append(f"  p{n} = ({pt[0]:.1f}, {pt[1]:.1f})")
        lines.append(f"  vertical   = |p2-p6| + |p3-p5| = {_dist(p[1],p[5]):.2f} + {_dist(p[2],p[4]):.2f} = {vert:.2f}")
        lines.append(f"  horizontal = 2 * |p1-p4| = 2 * {_dist(p[0],p[3]):.2f} = {horiz:.2f}")
        lines.append(f"  EAR = vertical / horizontal = {vert:.2f} / {horiz:.2f} = {this_ear:.4f}")

    lines.append(f"\nEAR (avg of both eyes) = {ear_value:.4f}" if ear_value is not None else "\nEAR = N/A")

    top, bottom = landmarks_xy[MOUTH_TOP_IDX], landmarks_xy[MOUTH_BOTTOM_IDX]
    left, right = landmarks_xy[MOUTH_LEFT_IDX], landmarks_xy[MOUTH_RIGHT_IDX]
    m_vert = _dist(top, bottom)
    m_horiz = _dist(left, right)
    lines.append(f"\nMAR:")
    lines.append(f"  top    = ({top[0]:.1f}, {top[1]:.1f})   bottom = ({bottom[0]:.1f}, {bottom[1]:.1f})")
    lines.append(f"  left   = ({left[0]:.1f}, {left[1]:.1f})   right  = ({right[0]:.1f}, {right[1]:.1f})")
    lines.append(f"  vertical   = |top-bottom| = {m_vert:.2f}")
    lines.append(f"  horizontal = |left-right| = {m_horiz:.2f}")
    lines.append(f"  MAR = vertical / horizontal = {m_vert:.2f} / {m_horiz:.2f} = "
                 f"{(m_vert/m_horiz if m_horiz else 0.0):.4f}")

    nose = landmarks_xy[NOSE_TIP_IDX]
    left_c = _midpoint(landmarks_xy[LEFT_EYE_IDX[0]], landmarks_xy[LEFT_EYE_IDX[3]])
    right_c = _midpoint(landmarks_xy[RIGHT_EYE_IDX[0]], landmarks_xy[RIGHT_EYE_IDX[3]])
    eye_line_y = (left_c[1] + right_c[1]) / 2.0
    inter_eye = _dist(left_c, right_c)
    p_proxy = (nose[1] - eye_line_y) / inter_eye if inter_eye else 0.0
    lines.append(f"\nHEAD PITCH PROXY:")
    lines.append(f"  nose        = ({nose[0]:.1f}, {nose[1]:.1f})")
    lines.append(f"  L-EYE-C     = midpoint(L-eye corners) = ({left_c[0]:.1f}, {left_c[1]:.1f})")
    lines.append(f"  R-EYE-C     = midpoint(R-eye corners) = ({right_c[0]:.1f}, {right_c[1]:.1f})")
    lines.append(f"  eye_line_y  = (L-EYE-C.y + R-EYE-C.y) / 2 = {eye_line_y:.2f}")
    lines.append(f"  inter_eye   = |L-EYE-C - R-EYE-C| = {inter_eye:.2f}")
    lines.append(f"  pitch_proxy = (nose.y - eye_line_y) / inter_eye "
                 f"= ({nose[1]:.2f} - {eye_line_y:.2f}) / {inter_eye:.2f} = {p_proxy:.4f}")
    if pitch_value is not None:
        lines.append(f"  (pipeline's own computed value: {pitch_value:.4f} — "
                      f"{'matches' if abs(p_proxy - pitch_value) < 1e-6 else 'MISMATCH — investigate'})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _save_debug_artifacts(frame_bgr, detector, calib, pitch_calib, last_result):
    """Wired to RDASConfig.debug_visualize_frame_index — saves all three
    verification artifacts for one specific frame. getattr(..., None) guards
    every field since StubPersonFaceDetector has none of these attributes."""
    yolo_box = getattr(detector, "last_yolo_box", None)
    roi_bounds = getattr(detector, "last_roi_bounds", None)
    landmarks_xy = getattr(detector, "last_landmarks_xy", None)

    annotated, roi_crop = draw_debug_visualization(
        frame_bgr, yolo_box, roi_bounds, landmarks_xy,
        calib.is_calibrated, calib.baseline_ear,
        pitch_calib.is_calibrated, pitch_calib.baseline_pitch,
        last_result.ear, last_result.mar, last_result.pitch_proxy,
    )
    cv2.imwrite("rdas_debug_full_frame.jpg", annotated)
    print("[debug] saved rdas_debug_full_frame.jpg (YOLO box, ROI box, landmark points, calibration state)")
    if roi_crop is not None:
        cv2.imwrite("rdas_debug_roi_crop.jpg", roi_crop)
        print("[debug] saved rdas_debug_roi_crop.jpg (ROI crop alone, 2x upscaled, same points)")
    else:
        print("[debug] no ROI crop available for this frame (nothing detected / stub detector)")

    calc_text = format_calculations_text(landmarks_xy, last_result.ear, last_result.mar,
                                          last_result.pitch_proxy)
    with open("rdas_debug_calculations.txt", "w") as f:
        f.write(calc_text)
    print("[debug] saved rdas_debug_calculations.txt (EAR/MAR/pitch math, point by point)")
    print(calc_text)


# ============================================================================
# REPORT — run-level stats collection and end-of-run terminal report
# ============================================================================

@dataclass
class PerfTimers:
    """Accumulates wall-clock time per pipeline STAGE across the whole run —
    separate from RunStats, which accumulates per-EVENT counts (alarms, etc).
    Every field here is a running total in seconds, added to once per frame
    (or once per detector-call, where noted) via simple `t1 - t0` wraps around
    the existing loop code — no behavior changes, just timing added around
    what was already there."""
    read_total: float = 0.0            # cap.read()
    covered_check_total: float = 0.0   # frame_brightness_stats() + camera_covered(), every frame
    yolo_total: float = 0.0            # YOLO26 person-detection call only, detector-call frames only
    landmark_total: float = 0.0        # mediapipe FaceLandmarker call only, detector-call frames only
    state_update_total: float = 0.0    # drowsy/yawn/distraction/no_driver .update() calls
    overlay_total: float = 0.0         # building overlay_lines + cv2.putText loop, every frame
    write_total: float = 0.0           # writer.write(frame), every frame


@dataclass
class RunStats:
    fps: float = 30.0
    total_frames: int = 0        # every frame written to output
    detector_calls: int = 0      # frames where YOLO+mediapipe actually ran
    covered_frames: int = 0
    stale_frames: int = 0        # live-stream feed frozen — see stream_stale()
    no_driver_alarm_frames: int = 0
    microsleep_alarm_frames: int = 0
    perclos_alarm_frames: int = 0
    distraction_alarm_frames: int = 0
    ear_values: List[float] = field(default_factory=list)  # post-calibration only
    mar_values: List[float] = field(default_factory=list)
    pitch_values: List[float] = field(default_factory=list)
    total_inference_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    baseline_ear: Optional[float] = None
    baseline_pitch: Optional[float] = None

    def record_frame(self, covered: bool, no_driver_alarm: bool,
                      microsleep_alarm: bool, perclos_alarm: bool, distraction_alarm: bool,
                      stale: bool = False):
        """Called once per output frame — alarm flags persist across skipped
        frames, so counting here gives % of VIDEO TIME in each state, not
        just % of detector calls."""
        self.total_frames += 1
        self.covered_frames += covered
        self.stale_frames += stale
        self.no_driver_alarm_frames += no_driver_alarm
        self.microsleep_alarm_frames += microsleep_alarm
        self.perclos_alarm_frames += perclos_alarm
        self.distraction_alarm_frames += distraction_alarm

    def record_detector_call(self, inference_time_sec: float):
        self.detector_calls += 1
        self.total_inference_time += inference_time_sec

    def record_ear_mar(self, ear: Optional[float], mar: Optional[float]):
        # Only called post-calibration — calibration-window samples aren't
        # "monitored" readings and would skew the average toward baseline.
        if ear is not None:
            self.ear_values.append(ear)
        if mar is not None:
            self.mar_values.append(mar)

    def record_pitch(self, pitch: Optional[float]):
        if pitch is not None:
            self.pitch_values.append(pitch)


def _pct(count: int, total: int) -> float:
    return (count / total * 100) if total else 0.0


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_report(stats: RunStats, yawn_count: int, distraction_count: int,
                  output_path: str, sampling_interval: int):
    processing_time = time.time() - stats.start_time
    video_duration = stats.total_frames / stats.fps if stats.fps else 0.0

    avg_inference = stats.total_inference_time / stats.detector_calls if stats.detector_calls else 0.0
    inference_fps = 1 / avg_inference if avg_inference else 0.0
    pipeline_fps = stats.total_frames / processing_time if processing_time else 0.0

    print("\n" + "=" * 55)
    print("RDAS RUN REPORT")
    print("=" * 55)
    print(f"Output Video         : {output_path}")
    print(f"Video Duration (sec) : {video_duration:.2f}")
    print(f"Total Frames         : {stats.total_frames}")
    print(f"Detector Calls       : {stats.detector_calls}  (sampling interval: {sampling_interval})")

    print(f"\nCalibrated Baseline EAR   : "
          f"{stats.baseline_ear:.3f}" if stats.baseline_ear is not None else "\nCalibrated Baseline EAR   : not reached")
    print(f"Calibrated Baseline Pitch : "
          f"{stats.baseline_pitch:.3f}" if stats.baseline_pitch is not None else "Calibrated Baseline Pitch : not reached")
    print(f"Average EAR (post-cal)    : {_avg(stats.ear_values):.3f}")
    print(f"Average MAR (post-cal)    : {_avg(stats.mar_values):.3f}")
    print(f"Average Pitch (post-cal)  : {_avg(stats.pitch_values):.3f}")

    print(f"\nCamera Covered   : {stats.covered_frames} frames "
          f"({_pct(stats.covered_frames, stats.total_frames):.1f}% of video time)")
    print(f"Stream Stale     : {stats.stale_frames} frames "
          f"({_pct(stats.stale_frames, stats.total_frames):.1f}% of video time) "
          f"[always 0 for file input]")
    print(f"No-Driver Alarm  : {stats.no_driver_alarm_frames} frames "
          f"({_pct(stats.no_driver_alarm_frames, stats.total_frames):.1f}% of video time)")
    print(f"Microsleep Alarm : {stats.microsleep_alarm_frames} frames "
          f"({_pct(stats.microsleep_alarm_frames, stats.total_frames):.1f}% of video time)")
    print(f"PERCLOS Alarm    : {stats.perclos_alarm_frames} frames "
          f"({_pct(stats.perclos_alarm_frames, stats.total_frames):.1f}% of video time)")
    print(f"Distraction Alarm: {stats.distraction_alarm_frames} frames "
          f"({_pct(stats.distraction_alarm_frames, stats.total_frames):.1f}% of video time)")
    print(f"Yawn Count       : {yawn_count}")
    print(f"Distraction Count: {distraction_count}")

    print(f"\nAverage Inference  : {avg_inference * 1000:.2f} ms")
    print(f"Inference FPS      : {inference_fps:.2f}")
    print(f"Pipeline FPS       : {pipeline_fps:.2f}")
    print(f"Total Processing   : {processing_time:.2f} sec")
    print("=" * 55)


def print_performance_report(perf: PerfTimers, video_duration: float, total_time: float,
                              total_frames: int, detector_calls: int):
    """Per-STAGE wall-clock breakdown — separate from print_report's per-EVENT
    summary above. Answers "which part is actually taking the time", not
    "what did the video contain". All percentages are of total_time, so they
    sum to ~100% minus whatever small untimed overhead sits between stages
    (loop bookkeeping, Python's own overhead) — if the stages don't add up
    close to total_time, that gap is itself informative (means something
    outside these wraps is expensive, e.g. VideoWriter codec setup once).
    """
    stage_total = (perf.read_total + perf.covered_check_total + perf.yolo_total +
                   perf.landmark_total + perf.state_update_total +
                   perf.overlay_total + perf.write_total)

    def pct(x):
        return (x / total_time * 100) if total_time else 0.0

    pipeline_fps = total_frames / total_time if total_time else 0.0
    # AI FPS: throughput of the detector-call stage alone (YOLO+landmark),
    # i.e. how many detector calls could run per second if that were the
    # only thing happening — NOT the same as pipeline_fps, which is diluted
    # by every-frame overhead (read/write/overlay) across ALL frames.
    ai_time = perf.yolo_total + perf.landmark_total
    ai_fps = detector_calls / ai_time if ai_time else 0.0
    # Real Time Factor: how many seconds of wall-clock this pipeline needs
    # per second of video. RTF < 1.0 = faster than real-time (good for a
    # live camera feed); RTF > 1.0 = the pipeline cannot keep up live and
    # would need to drop frames or fall behind on a real-time stream.
    rtf = total_time / video_duration if video_duration else 0.0

    print("\n" + "=" * 50)
    print("PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Read Time          : {perf.read_total:.2f} sec  ({pct(perf.read_total):.1f}%)")
    print(f"Covered-Check Time : {perf.covered_check_total:.2f} sec  ({pct(perf.covered_check_total):.1f}%)")
    print(f"YOLO Detection Time: {perf.yolo_total:.2f} sec  ({pct(perf.yolo_total):.1f}%)")
    print(f"Landmark Time      : {perf.landmark_total:.2f} sec  ({pct(perf.landmark_total):.1f}%)")
    print(f"State Update Time  : {perf.state_update_total:.2f} sec  ({pct(perf.state_update_total):.1f}%)")
    print(f"Drawing Time       : {perf.overlay_total:.2f} sec  ({pct(perf.overlay_total):.1f}%)")
    print(f"Video Write Time   : {perf.write_total:.2f} sec  ({pct(perf.write_total):.1f}%)")
    print(f"[unaccounted overhead: {total_time - stage_total:.2f} sec "
          f"({pct(total_time - stage_total):.1f}%) — model load, codec setup, loop bookkeeping]")
    print()
    print(f"Video Duration     : {video_duration:.2f} sec")
    print(f"Processing Time    : {total_time:.2f} sec")
    print(f"Pipeline FPS       : {pipeline_fps:.2f}")
    print(f"AI FPS             : {ai_fps:.2f}")
    print(f"Real Time Factor   : {rtf:.3f}" +
          ("  (SLOWER than real-time)" if rtf > 1.0 else "  (faster than real-time)"))
    print("=" * 50)


# ============================================================================
# PIPELINE — video in -> annotated video out
# ============================================================================
# Camera-covered check runs on EVERY frame (cheap: just mean/std of pixels).
# Person/face detection only runs every `sampling_interval` frames (expensive:
# YOLO + mediapipe). Between processed frames, last known alarm state is
# carried forward for overlay/logging continuity.
# Items 4/5/8 (phone, smoking, headphones) aren't wired in yet.

# ============================================================================
# STRUCTURED ALARM/TELEMETRY BROADCAST (WebSocket side-channel for the HUD)
# ============================================================================

ALARM_DEFS: Dict[str, Tuple[str, str, str]] = {
    # code -> (title, detail, level)
    "stale":       ("Stream Stale", "No new frame received from the source feed", "critical"),
    "covered":     ("Camera Covered", "Camera view is dark and low-variance", "high"),
    "no_driver":   ("No Driver Detected", "No person detected in frame beyond timeout", "critical"),
    "microsleep":  ("Microsleep Alarm", "Sustained eye closure detected", "critical"),
    "perclos":     ("Drowsiness Trend (PERCLOS)", "Elevated eye-closure ratio over rolling window", "warning"),
    "yawn":        ("Yawn Alarm", "Sustained mouth-open (yawn) detected", "warning"),
    "distraction": ("Distraction Alarm", "Sustained downward head pitch (gaze off-road / phone-use proxy)", "high"),
}


class AlarmBroadcastServer:
    """Background WebSocket server broadcasting JSON state snapshots to any
    connected frontend (e.g. the React HUD). Runs its own asyncio event loop
    on a dedicated thread so it never blocks the synchronous OpenCV capture
    loop in run().

    Also receives commands FROM the frontend (alarm acknowledge/silence
    button presses) — previously this was one-way only ("frontend is
    receive-only; nothing to process from it"), which is why those buttons
    did nothing no matter how many times you clicked them. Incoming frames
    are parsed as JSON and pushed onto a thread-safe queue; the main
    pipeline thread drains it via get_pending_commands() and applies them to
    AlarmStateTracker — asyncio objects aren't safe to touch directly from
    another thread, so the queue is the hand-off point.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._clients = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._failed = False
        self.command_queue: "queue.Queue[dict]" = queue.Queue()

    def start(self):
        threading.Thread(target=self._run_loop, daemon=True, name="rdas-ws-server").start()
        self._ready.wait(timeout=5.0)
        if self._failed:
            print("[ws] WARNING: alarm broadcast server failed to start — "
                  "the HUD will show no live data. `pip install websockets` and check ws_port is free.")

    def _run_loop(self):
        try:
            import websockets
        except ImportError:
            print("[ws] the 'websockets' package is not installed (pip install websockets) — "
                  "structured alarm broadcast is disabled; RTSP video overlay will still work.")
            self._failed = True
            self._ready.set()
            return

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(ws):
            self._clients.add(ws)
            print(f"[ws] HUD client connected ({len(self._clients)} total)")
            try:
                async for raw in ws:
                    try:
                        cmd = json.loads(raw)
                    except (ValueError, TypeError):
                        print(f"[ws] WARNING: dropped malformed command frame: {raw!r}")
                        continue
                    if not isinstance(cmd, dict) or "action" not in cmd:
                        print(f"[ws] WARNING: dropped command missing 'action': {cmd!r}")
                        continue
                    self.command_queue.put(cmd)
            except Exception:
                pass
            finally:
                self._clients.discard(ws)
                print(f"[ws] HUD client disconnected ({len(self._clients)} total)")

        async def _serve():
            async with websockets.serve(handler, self.host, self.port):
                print(f"[ws] alarm broadcast server listening on ws://{self.host}:{self.port}")
                self._ready.set()
                await asyncio.Future()  # run forever

        try:
            self._loop.run_until_complete(_serve())
        except Exception as e:
            print(f"[ws] server failed to start: {e}")
            self._failed = True
            self._ready.set()

    def get_pending_commands(self) -> List[dict]:
        """Drains and returns every command received since the last call.
        Non-blocking — safe to call every frame from the main pipeline
        thread even if nothing's arrived."""
        cmds = []
        while True:
            try:
                cmds.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def publish(self, state: dict):
        """Thread-safe: called from the main (synchronous) pipeline thread."""
        if self._loop is None or not self._clients:
            return
        payload = json.dumps(state)

        async def _broadcast():
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for d in dead:
                self._clients.discard(d)

        try:
            asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)
        except Exception:
            pass


class AlarmStateTracker:
    """Tracks alarm on/off edges to build the structured payload each cycle:
    which alarms are currently active (with a 'since' timer), plus an
    append-only event log entry each time an alarm's state flips. This is
    what feeds AlarmPanel / DriverStatusPanel / EventLogWidget in the HUD —
    none of that structure exists in the video-overlay text alone.

    Also tracks dispatcher actions from the frontend's ACKNOWLEDGE/SILENCE
    buttons (see AlarmBroadcastServer.get_pending_commands): 'ack' marks a
    currently-active alarm as acknowledged (still shown, but flagged so the
    HUD can stop blinking/insisting on it); 'silence' hides an alarm from
    the outgoing payload for a fixed duration even if the underlying
    condition is still active, e.g. a dispatcher saying "I see it, stop
    showing it for a minute." Both reset the moment the alarm clears and
    re-triggers as a fresh occurrence — an old ack/silence shouldn't quietly
    apply to a brand new instance of the same alarm type.
    """

    def __init__(self):
        self.active_since: Dict[str, float] = {}
        self.events: deque = deque(maxlen=300)
        self._event_seq = 0
        self.acked: set = set()
        self.silenced_until: Dict[str, float] = {}

    def apply_command(self, cmd: dict, now_t: float) -> None:
        action = cmd.get("action")
        code = cmd.get("code")
        if code not in ALARM_DEFS:
            print(f"[alarm-cmd] WARNING: ignoring command with unknown code: {cmd!r}")
            return
        if action == "ack":
            # Only log the first ack of an active alarm (ACK ALL / double clicks
            # shouldn't spam the history), and ignore acks for inactive alarms.
            if code in self.active_since and code not in self.acked:
                title = ALARM_DEFS[code][0]
                self._event_seq += 1
                self.events.appendleft({
                    "id": str(self._event_seq),
                    "ts": _fmt_clock(now_t),
                    "event": f"{code}_acked",
                    "level": "normal",
                    "message": f"{title} acknowledged",
                })
            self.acked.add(code)
            print(f"[alarm-cmd] '{code}' acknowledged")
        elif action == "silence":
            duration = cmd.get("duration_sec", 60)
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = 60.0
            self.silenced_until[code] = now_t + duration
            print(f"[alarm-cmd] '{code}' silenced for {duration:.0f}s")
        else:
            print(f"[alarm-cmd] WARNING: ignoring unknown action: {cmd!r}")

    def update(self, t: float, flags: Dict[str, bool]) -> Tuple[List[dict], List[dict]]:
        for code, is_active in flags.items():
            was_active = code in self.active_since
            if is_active and not was_active:
                self.active_since[code] = t
                # Fresh occurrence — any ack from a previous bout of this
                # alarm no longer applies. (Silence is left alone: a timed
                # silence should keep holding through a brief flicker rather
                # than un-silence itself the instant the condition clears
                # and re-triggers; it still expires on its own via the time
                # check below.)
                self.acked.discard(code)
                title, detail, level = ALARM_DEFS[code]
                self._event_seq += 1
                self.events.appendleft({
                    "id": str(self._event_seq),
                    "ts": _fmt_clock(t),
                    "event": code,
                    "level": level,
                    "message": title,
                })
            elif not is_active and was_active:
                del self.active_since[code]

        alarms = []
        for code, since_t in self.active_since.items():
            if self.silenced_until.get(code, 0.0) > t:
                continue  # dispatcher silenced this one — suppress from output until it expires
            title, detail, level = ALARM_DEFS[code]
            alarms.append({
                "code": code,
                "title": title,
                "detail": detail,
                "level": level,
                "since": _fmt_clock(max(0.0, t - since_t)),
                "acked": code in self.acked,
            })
        return alarms, list(self.events)[:50]


def _fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(cfg: RDASConfig):
    cap = LiveStreamReader(cfg.input_video_path, cfg)  # handles both files and rtsp/rtmp/http streams

    src_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.camera_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0  # live streams don't report a frame count — treat as "unknown", not an error

    cfg.camera_fps = int(round(src_fps))
    sampling_interval = cfg.derive_sampling_interval()
    print(f"[pipeline] source fps={src_fps:.1f}, sampling_interval={sampling_interval} frames "
          f"(~{sampling_interval/src_fps:.3f}s between detector runs)")
    if cap.is_stream:
        print(f"[pipeline] LIVE STREAM input — no defined end, will run until interrupted or "
              f"reconnect attempts are exhausted (stream_max_reconnect_attempts="
              f"{cfg.stream_max_reconnect_attempts})")

    writer = RTSPPublisher(cfg.output_video_path, src_fps, width, height, cfg)
    if cap.is_stream and not writer.is_stream:
        print("[pipeline] NOTE: input is a live stream but output_video_path is a local file — "
              "set output_video_path to an rtsp:// URL to publish live instead")

    if cfg.use_real_detector:
        detector = RealPersonFaceDetector(cfg.yolo_weights_path, cfg.face_task_path)
        print("[pipeline] using REAL YOLO26 + mediapipe detector")
    else:
        detector = StubPersonFaceDetector()
        print("[pipeline] using STUB detector — plumbing test only")

    calib = CalibrationTracker(cfg)
    drowsy = DrowsinessStateMachine(cfg, calib)
    yawn = YawnStateMachine(cfg)
    no_driver = NoDriverTracker(cfg)
    pitch_calib = PitchCalibrationTracker(cfg)
    distraction = DistractionStateMachine(cfg, pitch_calib)
    stats = RunStats(fps=src_fps)
    perf = PerfTimers()

    ws_server = None
    if cfg.enable_ws_server:
        ws_server = AlarmBroadcastServer(cfg.ws_host, cfg.ws_port)
        ws_server.start()
    alarm_tracker = AlarmStateTracker()
    _last_ws_publish = 0.0

    frame_idx = 0
    last_result = None
    run_start = time.time()
    _debug_saved = False
    _last_progress_print = run_start
    if cfg.debug_visualize_frame_index is not None:
        print(f"[debug] armed: will save verification artifacts on the first detector-call "
              f"frame >= {cfg.debug_visualize_frame_index} (~{cfg.debug_visualize_frame_index/src_fps:.2f}s)")

    while True:
        _t0 = time.time()
        ok, frame = cap.read()
        perf.read_total += time.time() - _t0
        if not ok:
            break
        # File: t is video-relative time, correct since every frame is
        # processed in sequence at the video's own rate. Stream: frame_idx
        # no longer tracks real elapsed time once we're grabbing "whatever's
        # latest" rather than a fixed cadence — our loop can spin faster or
        # slower than src_fps depending on our own processing speed, so
        # frame_idx/src_fps would drift from reality. Wall-clock time is the
        # only correct choice here, since the microsleep/PERCLOS/distraction
        # timers all compare against real elapsed seconds.
        t = time.time() if cap.is_stream else frame_idx / src_fps

        _t0 = time.time()
        brightness, variance = frame_brightness_stats(frame)  # every frame, cheap
        covered = camera_covered(brightness, variance, cfg)
        feed_age = cap.seconds_since_last_frame()
        stale = stream_stale(feed_age, cfg)
        perf.covered_check_total += time.time() - _t0

        if frame_idx % sampling_interval == 0 and not covered and not stale:  # expensive path, every Nth frame
            infer_start = time.time()
            last_result = detector.detect(frame, t)
            stats.record_detector_call(time.time() - infer_start)

            _t0 = time.time()
            no_driver.update(t, last_result.person_detected)
            if last_result.person_detected and last_result.ear is not None:
                drowsy.update(t, last_result.ear)
            if last_result.person_detected and last_result.mar is not None:
                yawn.update(t, last_result.mar)
            # Distraction updates independently of drowsy — deliberately NOT
            # gated on drowsy's calibration or alarm state, so a high-pitch
            # reading can never suppress or be suppressed by the EAR signal.
            if last_result.person_detected and last_result.pitch_proxy is not None:
                distraction.update(t, last_result.pitch_proxy)
            if calib.is_calibrated:
                stats.baseline_ear = calib.baseline_ear
                stats.record_ear_mar(last_result.ear, last_result.mar)
            if pitch_calib.is_calibrated:
                stats.baseline_pitch = pitch_calib.baseline_pitch
                stats.record_pitch(last_result.pitch_proxy)
            perf.state_update_total += time.time() - _t0

            if (cfg.debug_visualize_frame_index is not None and not _debug_saved
                    and frame_idx >= cfg.debug_visualize_frame_index):
                _save_debug_artifacts(frame, detector, calib, pitch_calib, last_result)
                _debug_saved = True
                print(f"[debug] captured at frame_idx={frame_idx} (t={t:.2f}s) — "
                      f"nearest detector-call frame at or after the requested index")

        # --- overlay ---
        stats.record_frame(covered, no_driver.no_driver_alarm,
                            drowsy.microsleep_alarm, drowsy.perclos_alarm,
                            distraction.distraction_alarm, stale)
        _t0 = time.time()
        if cfg.draw_face_contours and not covered:
            # Read the detector's most recently known ROI/landmarks — these
            # only update on detector-call frames, so on skipped frames this
            # naturally reuses the last real detection, same persistence
            # pattern as the EAR/MAR/pitch text below. getattr guards against
            # StubPersonFaceDetector, which has neither attribute.
            draw_landmark_contours(frame, getattr(detector, "last_roi_bounds", None),
                                    getattr(detector, "last_landmarks_xy", None))
        overlay_lines = [f"t={t:.1f}s  frame={frame_idx}"]
        if stale:
            overlay_lines.append(f"STREAM STALE ({feed_age:.1f}s since last frame)")
        elif covered:
            overlay_lines.append("CAMERA COVERED")
        else:
            overlay_lines.append(f"brightness={brightness:.0f} var={variance:.0f}")
            if no_driver.no_driver_alarm:
                overlay_lines.append("NO DRIVER DETECTED")
            if last_result and last_result.person_detected:
                if not (calib.is_calibrated and pitch_calib.is_calibrated):
                    overlay_lines.append("CALIBRATING...")
                else:
                    ear_txt = f"EAR={drowsy.current_ear:.3f}" if drowsy.current_ear is not None else "EAR=--"
                    mar_txt = f"MAR={yawn.current_mar:.3f}" if yawn.current_mar is not None else "MAR=--"
                    pitch_txt = f"PITCH={distraction.current_pitch:.3f}" if distraction.current_pitch is not None else "PITCH=--"
                    overlay_lines.append(ear_txt + "  " + mar_txt + "  " + pitch_txt)
                    if drowsy.microsleep_alarm:
                        overlay_lines.append("MICROSLEEP ALARM")
                    if drowsy.perclos_alarm:
                        overlay_lines.append("DROWSINESS TREND ALARM (PERCLOS)")
                    if yawn.yawn_alarm:
                        overlay_lines.append(f"YAWN ALARM (count={yawn.yawn_count})")
                    if distraction.distraction_alarm:
                        overlay_lines.append(f"DISTRACTION ALARM (count={distraction.distraction_count})")

        y = 25
        for line in overlay_lines:
            is_alert = any(kw in line for kw in ("ALARM", "COVERED", "NO DRIVER", "STALE"))
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        perf.overlay_total += time.time() - _t0

        # --- structured alarm/telemetry broadcast (separate from video overlay) ---
        if ws_server is not None:
            for cmd in ws_server.get_pending_commands():
                alarm_tracker.apply_command(cmd, t)

        if ws_server is not None and (t - _last_ws_publish) >= cfg.ws_broadcast_interval_sec:
            _last_ws_publish = t
            person_present = bool(last_result and last_result.person_detected)
            flags = {
                "stale": stale,
                "covered": covered,
                "no_driver": no_driver.no_driver_alarm,
                "microsleep": drowsy.microsleep_alarm,
                "perclos": drowsy.perclos_alarm,
                "yawn": yawn.yawn_alarm,
                "distraction": distraction.distraction_alarm,
            }
            alarms, events = alarm_tracker.update(t, flags)

            # Real (not fabricated) face/person bounding box, as normalized
            # percentages of frame size, so the frontend can draw a box that
            # actually tracks the detector's output instead of a static mock.
            face_box = None
            yolo_box = getattr(detector, "last_yolo_box", None)
            if yolo_box is not None and width and height:
                bx1, by1, bx2, by2 = yolo_box
                face_box = {
                    "x": round(100 * bx1 / width, 2),
                    "y": round(100 * by1 / height, 2),
                    "w": round(100 * (bx2 - bx1) / width, 2),
                    "h": round(100 * (by2 - by1) / height, 2),
                }
            signals = [
                {"key": "present", "label": "Driver Present", "active": person_present,
                 "level": "normal" if person_present else "critical"},
                {"key": "drowsy", "label": "Drowsiness",
                 "active": drowsy.microsleep_alarm or drowsy.perclos_alarm,
                 "level": "critical" if drowsy.microsleep_alarm else ("warning" if drowsy.perclos_alarm else "off"),
                 "value": f"EAR={drowsy.current_ear:.2f}" if drowsy.current_ear is not None else None},
                {"key": "yawn", "label": "Yawning", "active": yawn.yawn_alarm,
                 "level": "warning" if yawn.yawn_alarm else "off",
                 "value": f"count={yawn.yawn_count}" if yawn.yawn_count else None},
                {"key": "distracted", "label": "Distracted Driver", "active": distraction.distraction_alarm,
                 "level": "high" if distraction.distraction_alarm else "off",
                 "value": f"PITCH={distraction.current_pitch:.2f}" if distraction.current_pitch is not None else None},
                {"key": "covered", "label": "Camera Covered", "active": covered,
                 "level": "high" if covered else "off"},
            ]
            state = {
                "type": "state",
                "server_ts": time.time(),
                "video_t": round(t, 2),
                "frame": frame_idx,
                "calibrated": calib.is_calibrated and pitch_calib.is_calibrated,
                "person_detected": person_present,
                "metrics": {
                    "fps": round(src_fps, 1),
                    "ear": round(drowsy.current_ear, 3) if drowsy.current_ear is not None else None,
                    "mar": round(yawn.current_mar, 3) if yawn.current_mar is not None else None,
                    "pitch": round(distraction.current_pitch, 3) if distraction.current_pitch is not None else None,
                },
                "alarms": alarms,
                "events": events,
                "signals": signals,
                "face_box": face_box,
            }
            ws_server.publish(state)

        _t0 = time.time()
        writer.write(frame)
        perf.write_total += time.time() - _t0

        frame_idx += 1
        if time.time() - _last_progress_print >= 5.0:
            # Time-based, not frame-count-based: total_frames is 0/unknown
            # for a live stream, so the old "every total_frames//10 frames"
            # logic had nothing to divide by. This works identically for
            # files and streams, and is arguably more useful for a live
            # monitor anyway — elapsed wall-clock matters more than a frame
            # count when there's no defined end to count toward.
            suffix = f"{frame_idx}/{total_frames} frames" if total_frames else f"{frame_idx} frames (live)"
            print(f"[pipeline] {suffix} processed, {time.time()-run_start:.1f}s elapsed")
            _last_progress_print = time.time()

    cap.release()
    writer.release()
    total_time = time.time() - run_start
    # YOLO/landmark timers live on RealPersonFaceDetector itself (only that
    # class does real inference); pull them in now. Stub detector has neither
    # attribute, so both default to 0.0 and the report just shows 0% there.
    perf.yolo_total = getattr(detector, "_yolo_time_total", 0.0)
    perf.landmark_total = getattr(detector, "_landmark_time_total", 0.0)
    print(f"[pipeline] done. wrote {frame_idx} frames -> {cfg.output_video_path}")

    if cfg.debug_visualize_frame_index is not None and not _debug_saved:
        print(f"[debug] WARNING: requested frame index {cfg.debug_visualize_frame_index} "
              f"was never reached — the video only had {frame_idx} frames. Nothing was saved.")

    if cfg.print_report:
        print_report(stats, yawn.yawn_count, distraction.distraction_count,
                      cfg.output_video_path, sampling_interval)
        # YOLO/landmark sub-timers only exist on RealPersonFaceDetector — the
        # stub does no real inference, so perf.yolo_total/landmark_total stay
        # 0.0 in that case and print_performance_report still works, it just
        # won't show a meaningful split for those two rows.
        video_duration = frame_idx / src_fps if src_fps else 0.0
        print_performance_report(perf, video_duration, total_time, frame_idx, stats.detector_calls)


# ============================================================================
# SELF-TESTS — offline, synthetic data, no camera/YOLO/mediapipe needed
# ============================================================================
# Validates the MATH and STATE MACHINE LOGIC only, not whether the landmark
# indices above match a real mediapipe output (that needs a real model run).
# Toggle via RDASConfig.run_self_tests instead of a separate command.

def _make_eye_landmarks(openness: float):
    """openness=1.0 -> open eye hexagon, 0.0 -> fully flat (closed)."""
    half_h = 0.05 * openness
    return {
        LEFT_EYE_IDX[0]: (0.0, 0.0),
        LEFT_EYE_IDX[1]: (0.3, -half_h),
        LEFT_EYE_IDX[2]: (0.6, -half_h),
        LEFT_EYE_IDX[3]: (1.0, 0.0),
        LEFT_EYE_IDX[4]: (0.6, half_h),
        LEFT_EYE_IDX[5]: (0.3, half_h),
    }


def _make_mouth_landmarks(openness: float):
    half_h = 0.05 * openness
    return {
        MOUTH_TOP_IDX: (0.5, -half_h),
        MOUTH_BOTTOM_IDX: (0.5, half_h),
        MOUTH_LEFT_IDX: (0.0, 0.0),
        MOUTH_RIGHT_IDX: (1.0, 0.0),
    }


def _make_face_landmarks_for_pitch(nose_y_offset: float):
    """Eyes level at y=0, inter-eye distance fixed at 0.8, nose offset below
    the eye line by nose_y_offset -> pitch_proxy = nose_y_offset / 0.8."""
    return {
        LEFT_EYE_IDX[0]: (0.3, 0.0), LEFT_EYE_IDX[3]: (0.5, 0.0),
        RIGHT_EYE_IDX[0]: (-0.3, 0.0), RIGHT_EYE_IDX[3]: (-0.5, 0.0),
        NOSE_TIP_IDX: (0.0, nose_y_offset),
    }


def _test_ear_decreases_as_eye_closes():
    open_ear = eye_aspect_ratio(_make_eye_landmarks(1.0), LEFT_EYE_IDX)
    half_ear = eye_aspect_ratio(_make_eye_landmarks(0.5), LEFT_EYE_IDX)
    closed_ear = eye_aspect_ratio(_make_eye_landmarks(0.0), LEFT_EYE_IDX)
    assert open_ear > half_ear > closed_ear
    assert closed_ear == 0.0
    print(f"  EAR open={open_ear:.3f} half={half_ear:.3f} closed={closed_ear:.3f}  [PASS]")


def _test_mar_increases_as_mouth_opens():
    closed_mar = mouth_aspect_ratio(_make_mouth_landmarks(0.0))
    yawn_mar = mouth_aspect_ratio(_make_mouth_landmarks(1.5))
    assert yawn_mar > closed_mar
    print(f"  MAR closed={closed_mar:.3f} yawn={yawn_mar:.3f}  [PASS]")


def _test_blink_does_not_alarm_but_microsleep_does():
    cfg = RDASConfig(calibration_duration_sec=2.0, microsleep_continuous_closure_sec=1.5,
                      perclos_window_sec=60.0, perclos_threshold=0.15, ear_drop_ratio=0.75)
    calib = CalibrationTracker(cfg)
    dsm = DrowsinessStateMachine(cfg, calib)
    t, dt = 0.0, 1 / 30
    OPEN_EAR, CLOSED_EAR = 0.30, 0.15

    while t < cfg.calibration_duration_sec + dt:
        dsm.update(t, OPEN_EAR)
        t += dt
    assert calib.is_calibrated

    blink_end = t + 0.2
    while t < blink_end:
        dsm.update(t, CLOSED_EAR)
        t += dt
    assert dsm.microsleep_alarm is False, "200ms blink incorrectly triggered microsleep alarm"

    reopen_end = t + 1.0
    while t < reopen_end:
        dsm.update(t, OPEN_EAR)
        t += dt

    closure_end = t + 2.0
    fired_at = None
    while t < closure_end:
        dsm.update(t, CLOSED_EAR)
        if dsm.microsleep_alarm and fired_at is None:
            fired_at = t
        t += dt
    assert dsm.microsleep_alarm is True, "sustained 2s closure did not trigger microsleep alarm"
    print(f"  Blink (0.2s) -> no alarm [PASS]; sustained closure -> alarm fired "
          f"~{fired_at - (closure_end - 2.0):.2f}s in (threshold={cfg.microsleep_continuous_closure_sec}s)  [PASS]")


def _test_perclos_trend_alarm_without_single_long_closure():
    cfg = RDASConfig(calibration_duration_sec=1.0, microsleep_continuous_closure_sec=1.5,
                      perclos_window_sec=10.0, perclos_threshold=0.15, ear_drop_ratio=0.75)
    calib = CalibrationTracker(cfg)
    dsm = DrowsinessStateMachine(cfg, calib)
    t, dt = 0.0, 1 / 30
    OPEN_EAR, CLOSED_EAR = 0.30, 0.10

    while t < cfg.calibration_duration_sec + dt:
        dsm.update(t, OPEN_EAR)
        t += dt

    # closed 0.8s / open 1.2s repeating -> 40% closed, above threshold, but
    # no single closure reaches the 1.5s microsleep threshold
    end = t + 10.0
    while t < end:
        ear = CLOSED_EAR if (t % 2.0) < 0.8 else OPEN_EAR
        dsm.update(t, ear)
        t += dt

    assert dsm.microsleep_alarm is False, "no single closure should have reached 1.5s"
    assert dsm.perclos_alarm is True, "40% closure over the window should have tripped PERCLOS"
    print("  Repeated sub-threshold closures (40% of window) -> microsleep_alarm=False, "
          "perclos_alarm=True  [PASS]")


def _test_yawn_requires_minimum_duration():
    cfg = RDASConfig(mar_yawn_threshold=0.6, yawn_min_duration_sec=1.2)
    ysm = YawnStateMachine(cfg)
    t, dt = 0.0, 1 / 30

    for _ in range(int(0.4 * 30)):
        ysm.update(t, 0.8)
        t += dt
    assert ysm.yawn_alarm is False, "short mouth-open incorrectly counted as yawn"
    assert ysm.yawn_count == 0

    for _ in range(int(0.5 * 30)):
        ysm.update(t, 0.2)
        t += dt

    for _ in range(int(1.5 * 30)):
        ysm.update(t, 0.8)
        t += dt
    assert ysm.yawn_alarm is True
    assert ysm.yawn_count == 1
    print("  0.4s mouth-open -> not counted [PASS]; 1.5s sustained -> yawn_count=1  [PASS]")


def _test_no_driver_timeout():
    cfg = RDASConfig(no_driver_timeout_sec=5.0)
    tracker = NoDriverTracker(cfg)
    t, dt = 0.0, 1 / 30

    for _ in range(int(3 * 30)):
        tracker.update(t, person_detected=True)
        t += dt
    assert tracker.no_driver_alarm is False

    for _ in range(int(4 * 30)):
        tracker.update(t, person_detected=False)
        t += dt
    assert tracker.no_driver_alarm is False, "4s absence should not yet alarm (timeout=5s)"

    for _ in range(int(2 * 30)):
        tracker.update(t, person_detected=False)
        t += dt
    assert tracker.no_driver_alarm is True, "6s total absence should have alarmed"
    print("  4s absence -> no alarm [PASS]; 6s absence -> alarm  [PASS]")


def _test_camera_covered_logic():
    cfg = RDASConfig(brightness_threshold=25.0, variance_threshold=15.0)
    assert camera_covered(mean_brightness=5.0, std_brightness=2.0, cfg=cfg) is True
    assert camera_covered(mean_brightness=10.0, std_brightness=20.0, cfg=cfg) is False
    result = camera_covered(mean_brightness=200.0, std_brightness=3.0, cfg=cfg)
    assert result is False, "expected — known blind spot, not a bug"
    print("  Dark+flat -> covered [PASS]; dark+textured (tunnel) -> not covered [PASS]")
    print("  Bright+flat (reflective spoof) -> NOT caught by brightness+variance alone "
          "(known gap, see README)")


def _test_stream_stale_logic():
    cfg = RDASConfig(stream_stale_threshold_sec=2.0)
    assert stream_stale(0.5, cfg) is False, "0.5s since last frame should not be stale"
    assert stream_stale(1.9, cfg) is False, "just under threshold should not be stale"
    assert stream_stale(2.1, cfg) is True, "just over threshold should be stale"
    assert stream_stale(30.0, cfg) is True, "long-frozen feed should be stale"
    print("  0.5s/1.9s since last frame -> not stale [PASS]; 2.1s/30s -> stale [PASS]")


def _test_live_stream_reader_file_passthrough():
    """A file path must produce IDENTICAL behavior to a raw cv2.VideoCapture
    — is_stream False, no threading, no staleness concept. This is what
    protects every previous file-based test/validation in this project from
    silently changing behavior when LiveStreamReader was introduced."""
    cfg = RDASConfig()
    reader = LiveStreamReader("synthetic_input.mp4", cfg)
    assert reader.is_stream is False
    assert reader.seconds_since_last_frame() == 0.0
    ok, frame = reader.read()
    assert ok is True and frame is not None, "should read the first real frame of the file"
    reader.release()
    print("  File path -> is_stream=False, seconds_since_last_frame()=0.0, "
          "read() returns a real frame  [PASS]")


def _test_live_stream_reader_paces_to_source_rate():
    """Found via a real live-camera run, not designed in ahead of time: a
    naive read() that instantly returns whatever's cached lets the consumer
    loop outrun the camera's actual delivery rate and reprocess the same
    frame many times over (observed: ~113 'frames/sec' processed against a
    real 15fps source). read() must BLOCK until a genuinely new frame is
    signaled — but never hang forever if the source stalls."""
    import threading as _threading
    import numpy as _np
    cfg = RDASConfig(stream_stale_threshold_sec=1.0)
    reader = LiveStreamReader("synthetic_input.mp4", cfg)
    reader.is_stream = True  # exercise the stream code path without a real network source
    fake_frame = _np.zeros((10, 10, 3), dtype=np.uint8)
    reader._latest_frame = fake_frame
    reader._latest_ok = True
    reader._last_frame_time = time.monotonic()
    reader._frame_ready.clear()  # no NEW frame has arrived yet

    # no new frame available -> read() blocks for the full timeout, then
    # returns the cached frame anyway rather than hanging indefinitely
    t0 = time.time()
    ok, _ = reader.read()
    elapsed = time.time() - t0
    assert 0.9 < elapsed < 1.3, f"expected ~1.0s timeout block, got {elapsed:.2f}s"
    assert ok is True

    # a new frame signaled mid-wait -> read() returns promptly, not after
    # the full timeout
    def _deliver_after_delay():
        time.sleep(0.3)
        with reader._lock:
            reader._latest_frame = fake_frame
            reader._last_frame_time = time.monotonic()
        reader._frame_ready.set()

    reader._frame_ready.clear()
    t = _threading.Thread(target=_deliver_after_delay)
    t.start()
    t0 = time.time()
    reader.read()
    elapsed = time.time() - t0
    t.join()
    reader.release()
    assert 0.25 < elapsed < 0.6, f"expected ~0.3s (prompt return), got {elapsed:.2f}s"
    print("  No new frame -> read() blocks ~1.0s then returns cached frame (no hang)  [PASS]")
    print("  New frame signaled at 0.3s -> read() returns at ~0.3s, not the full timeout  [PASS]")


def _test_rtsp_publisher_file_passthrough():
    """Same guarantee as the reader test above, for the write side: a local
    file output path must be an unchanged cv2.VideoWriter underneath, not a
    behavior change introduced by adding RTSP publish support."""
    cfg = RDASConfig()
    writer = RTSPPublisher("rtsp_publisher_test_output.mp4", fps=24.0, width=64, height=48, cfg=cfg)
    assert writer.is_stream is False
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    writer.write(frame)  # should not raise
    writer.release()
    assert os.path.exists("rtsp_publisher_test_output.mp4"), "file should have been written"
    os.remove("rtsp_publisher_test_output.mp4")
    print("  File path -> is_stream=False, write()/release() behave as a plain "
          "cv2.VideoWriter  [PASS]")


def _test_rtsp_publisher_missing_ffmpeg_fails_clearly():
    """If ffmpeg isn't on PATH, publishing must fail with a clear, actionable
    RuntimeError — not a bare FileNotFoundError traceback with no context."""
    cfg = RDASConfig()
    import unittest.mock as mock
    with mock.patch("subprocess.Popen", side_effect=FileNotFoundError()):
        try:
            RTSPPublisher("rtsp://localhost:8554/test", fps=24.0, width=64, height=48, cfg=cfg)
            assert False, "should have raised RuntimeError when ffmpeg is missing"
        except RuntimeError as e:
            assert "ffmpeg" in str(e).lower()
    print("  Missing ffmpeg -> clear RuntimeError mentioning ffmpeg, not a bare traceback  [PASS]")


def _test_frame_brightness_stats_correctness():
    """Never directly tested before this function's implementation changed
    (cv2-based instead of numpy) — only camera_covered() was tested, via
    injected values, which doesn't exercise this function at all. A uniform
    gray frame should have near-zero std and mean close to that gray value;
    a half-black/half-white frame should have high std."""
    uniform = np.full((100, 100, 3), 128, dtype=np.uint8)
    mean, std = frame_brightness_stats(uniform)
    assert abs(mean - 128) < 2.0, f"expected mean near 128, got {mean}"
    assert std < 1.0, f"expected near-zero std on a uniform frame, got {std}"

    split = np.zeros((100, 100, 3), dtype=np.uint8)
    split[:, 50:] = 255
    _, split_std = frame_brightness_stats(split)
    assert split_std > 100, f"expected high std on a half-black/half-white frame, got {split_std}"

    print(f"  Uniform gray frame -> mean~{mean:.1f}, std={std:.2f} (near zero)  [PASS]")
    print(f"  Half-black/half-white frame -> std={split_std:.1f} (high)  [PASS]")


def _test_pitch_proxy_increases_as_head_pitches_down():
    neutral = head_pitch_proxy(_make_face_landmarks_for_pitch(0.3))
    pitched_down = head_pitch_proxy(_make_face_landmarks_for_pitch(0.5))
    assert pitched_down > neutral
    print(f"  pitch_proxy neutral={neutral:.3f} pitched_down={pitched_down:.3f}  [PASS]")


def _test_distraction_requires_sustained_duration():
    cfg = RDASConfig(calibration_duration_sec=2.0, distraction_pitch_ratio=1.15,
                      distraction_min_duration_sec=2.0)
    calib = PitchCalibrationTracker(cfg)
    dsm = DistractionStateMachine(cfg, calib)
    t, dt = 0.0, 1 / 24
    BASELINE, HIGH = 0.68, 0.85  # same real-measured values used in the stub

    while t < cfg.calibration_duration_sec + dt:
        dsm.update(t, BASELINE)
        t += dt
    assert calib.is_calibrated

    # brief 0.8s glance — shorter than the 2.0s minimum, should not alarm
    glance_end = t + 0.8
    while t < glance_end:
        dsm.update(t, HIGH)
        t += dt
    assert dsm.distraction_alarm is False, "brief glance incorrectly counted as distraction"

    back_end = t + 1.0
    while t < back_end:
        dsm.update(t, BASELINE)
        t += dt

    # sustained 2.5s -> should alarm
    sustained_end = t + 2.5
    while t < sustained_end:
        dsm.update(t, HIGH)
        t += dt
    assert dsm.distraction_alarm is True
    assert dsm.distraction_count == 1
    print("  0.8s glance -> no alarm [PASS]; 2.5s sustained -> distraction_count=1  [PASS]")


def _test_distraction_does_not_suppress_microsleep():
    """The architectural claim this whole feature rests on: a simultaneous
    high pitch and a real eye closure must both alarm. If either channel
    silently gated the other, this test would fail — that's the point."""
    cfg = RDASConfig(calibration_duration_sec=2.0, microsleep_continuous_closure_sec=1.5,
                      distraction_pitch_ratio=1.15, distraction_min_duration_sec=1.5,
                      ear_drop_ratio=0.75)
    ear_calib = CalibrationTracker(cfg)
    drowsy = DrowsinessStateMachine(cfg, ear_calib)
    pitch_calib = PitchCalibrationTracker(cfg)
    distraction = DistractionStateMachine(cfg, pitch_calib)

    t, dt = 0.0, 1 / 24
    OPEN_EAR, CLOSED_EAR = 0.30, 0.15
    BASELINE_PITCH, HIGH_PITCH = 0.68, 0.85

    while t < cfg.calibration_duration_sec + dt:
        drowsy.update(t, OPEN_EAR)
        distraction.update(t, BASELINE_PITCH)
        t += dt
    assert ear_calib.is_calibrated and pitch_calib.is_calibrated

    # eyes close AND pitch goes high at the same time, sustained 2s
    end = t + 2.0
    while t < end:
        drowsy.update(t, CLOSED_EAR)
        distraction.update(t, HIGH_PITCH)
        t += dt

    assert drowsy.microsleep_alarm is True, "microsleep alarm was suppressed by concurrent distraction"
    assert distraction.distraction_alarm is True, "distraction alarm was suppressed by concurrent closure"
    print("  Concurrent closed-eyes + high-pitch -> BOTH microsleep_alarm and "
          "distraction_alarm True simultaneously  [PASS] (confirms no cross-suppression)")


def _test_distraction_tolerates_brief_noise_dips():
    """Reproduces the real-footage failure mode found during validation:
    a noisy signal that dips below threshold for short stretches within an
    otherwise-sustained high-pitch period must still alarm, not get reset
    to zero duration on every dip."""
    cfg = RDASConfig(calibration_duration_sec=2.0, distraction_pitch_ratio=1.15,
                      distraction_min_duration_sec=2.0, distraction_gap_tolerance_sec=0.5)
    calib = PitchCalibrationTracker(cfg)
    dsm = DistractionStateMachine(cfg, calib)
    t, dt = 0.0, 1 / 24
    BASELINE, HIGH = 0.70, 0.85

    while t < cfg.calibration_duration_sec + dt:
        dsm.update(t, BASELINE)
        t += dt
    assert calib.is_calibrated

    # 3s of high pitch with three brief 0.2s dips below threshold scattered
    # through it — none of the dips individually exceed the 0.5s tolerance.
    schedule_end = t + 3.0
    dip_windows = [(t + 0.8, t + 1.0), (t + 1.6, t + 1.8), (t + 2.3, t + 2.5)]
    while t < schedule_end:
        in_dip = any(lo <= t < hi for lo, hi in dip_windows)
        dsm.update(t, BASELINE if in_dip else HIGH)
        t += dt

    assert dsm.distraction_alarm is True, "brief noise dips incorrectly reset accumulated duration"
    print("  3s high-pitch with three 0.2s noise dips -> still alarms "
          "(gap tolerance survives noise)  [PASS]")

    # now a genuine 1.0s drop (exceeds 0.5s tolerance) -> must actually reset
    drop_end = t + 1.0
    while t < drop_end:
        dsm.update(t, BASELINE)
        t += dt
    assert dsm.distraction_alarm is False, "a real 1.0s drop should have reset the alarm"
    print("  Genuine 1.0s drop (> 0.5s tolerance) -> correctly resets  [PASS]")


def run_self_tests():
    tests = [
        _test_ear_decreases_as_eye_closes,
        _test_mar_increases_as_mouth_opens,
        _test_blink_does_not_alarm_but_microsleep_does,
        _test_perclos_trend_alarm_without_single_long_closure,
        _test_yawn_requires_minimum_duration,
        _test_no_driver_timeout,
        _test_camera_covered_logic,
        _test_stream_stale_logic,
        _test_live_stream_reader_file_passthrough,
        _test_live_stream_reader_paces_to_source_rate,
        _test_rtsp_publisher_file_passthrough,
        _test_rtsp_publisher_missing_ffmpeg_fails_clearly,
        _test_frame_brightness_stats_correctness,
        _test_pitch_proxy_increases_as_head_pitches_down,
        _test_distraction_requires_sustained_duration,
        _test_distraction_tolerates_brief_noise_dips,
        _test_distraction_does_not_suppress_microsleep,
    ]
    print("Running self-tests...\n")
    for test in tests:
        print(f"{test.__name__}:")
        test()
    print(f"\nAll {len(tests)} test groups passed.\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    cfg = RDASConfig()
    if cfg.run_self_tests:
        run_self_tests()
    run(cfg)
