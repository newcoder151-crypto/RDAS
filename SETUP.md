# RDAS — merged backend + frontend setup

## Quick start (single command)

First time only — fetch MediaMTX + the FaceLandmarker model (both are
gitignored, and MediaMTX in particular wasn't part of this merged upload —
see "Run order" below for why):
```
python scripts/setup_assets.py
```
Then edit `backend/.env` — it was auto-created from `.env.example` the first
time you run anything below, but `RDAS_INPUT_URL` is still a placeholder
until you put your real camera's RTSP URL/credentials in it.

Bring up MediaMTX + backend + frontend together, in the right order, from one
command:
```
python run_all.py
```
(`start.bat` on Windows / `./start.sh` on macOS/Linux do the same thing.)
Ctrl+C stops all three. Run `python run_all.py --help` for flags — e.g.
`--skip-mediamtx` if you're already running it yourself, or `--skip-frontend`
for a backend-only run. See `run_all.py`'s docstring for the full list.

The sections below (manual run order, ports, known gaps) are still accurate
if you'd rather run each piece by hand.

## What changed

**Backend (`backend/stream_pipeline.py`)**
- Added `AlarmBroadcastServer` + `AlarmStateTracker`: a WebSocket server on its
  own background thread that broadcasts JSON alarm/telemetry state (~5x/sec)
  to any connected browser. This runs **in parallel with** the existing RTSP
  video overlay — it doesn't replace it.
- Added a real (detector-derived) face bounding box to that JSON, instead of
  a hardcoded mock box.
- Replaced hardcoded Windows paths and an exposed camera password with
  env-var-overridable config (`RDAS_INPUT_URL`, `RDAS_YOLO_WEIGHTS`, etc — see
  `backend/.env.example`).

**Frontend (`frontend/src/...`)**
- `hooks/use-rdas-socket.ts`: connects to the backend's WebSocket, with
  auto-reconnect (exponential backoff) and a mock-data fallback so the UI
  still demos meaningfully if the backend isn't running.
- `components/rdas/VideoWidget.tsx`: now plays the real MediaMTX HLS stream
  via `hls.js`, falling back to the old simulated placeholder only while
  disconnected.
- `components/rdas/ActiveAlarmsList.tsx`: **new** — lists every currently
  active alarm as its own row. Previously `AlarmPanel` only ever showed the
  single highest-priority alarm, so a second simultaneous alarm was invisible.
- `routes/index.tsx`: wired to live data instead of the static mock arrays in
  `lib/rdas-backend.ts` (that file's exports are now just the fallback/demo
  data, used until the first successful WebSocket connection).

## Your actual question: is the video overlay enough, or do you need something separate?

**You need something separate — the video overlay alone can't drive the dashboard.**
`stream_pipeline.py` originally only had one output: `cv2.putText()` burning
alarm text directly into video pixels, published out over RTSP. A browser
frontend has no way to read text back out of a video frame into a structured
`AlarmPanel` / `DriverStatusPanel` / `EventLogWidget` — those widgets need
actual JSON objects (`{code, title, level, since}`, etc.), not pixels.

That's what the new WebSocket server does: same underlying alarm state,
delivered as structured JSON alongside (not instead of) the video overlay.
Keep the video overlay too — it's useful for anyone watching a raw RTSP feed
without the web dashboard (e.g. in VLC), and it's a reasonable redundant
record if the WebSocket ever drops.

**One thing to flag honestly:** the frontend's original mock data included
alarm categories the backend doesn't actually detect — phone usage, smoking,
headphones. The real backend only implements: microsleep, PERCLOS drowsiness
trend, yawning, distraction (head-pitch proxy — a stand-in for
phone-use/gaze-off-road), no-driver, camera-covered, and stream-stale. I
wired the UI to exactly what the backend can produce rather than inventing
detectors for the rest. If you need phone/smoking/headphone detection, that's
additional model work (extra YOLO classes + a state machine each), not a
wiring problem.

## Run order

1. **MediaMTX** (from your original `code.zip`, `mediaMTX/mediamtx_v1.19.2_windows_amd64/`):
   ```
   mediamtx.exe
   ```
   Default ports: RTSP `:8554`, HLS `:8888`, WebRTC `:8889`. No config changes
   needed — HLS and permissive CORS (`hlsAllowOrigins: ["*"]`) are already on
   by default in `mediamtx.yml`.

2. **Backend**:
   ```
   cd backend
   pip install -r requirements.txt
   cp .env.example .env   # then edit RDAS_INPUT_URL to your real camera
   python stream_pipeline.py
   ```
   Confirm you see `[ws] alarm broadcast server listening on ws://...:8765`
   and `[pipeline] source fps=...` in the console.

3. **Frontend**:
   ```
   cd frontend
   npm install   # or bun install
   cp .env.example .env   # only needed if backend isn't on localhost
   npm run dev
   ```
   Open the printed local URL. Live Feed should show the real annotated video
   within a few seconds (HLS has ~1-2s latency); alarms/driver status/event
   log should update in near-real-time over the WebSocket, independent of
   video latency.

## Ports summary

| Service              | Port | Purpose                              |
|-----------------------|------|---------------------------------------|
| MediaMTX RTSP publish  | 8554 | `stream_pipeline.py` -> MediaMTX      |
| MediaMTX HLS           | 8888 | MediaMTX -> browser `<video>`         |
| MediaMTX WebRTC        | 8889 | (not used yet, lower-latency option)  |
| RDAS alarm WebSocket   | 8765 | `stream_pipeline.py` -> browser HUD   |

## Known gaps / next steps if you want to push further

- **WebRTC instead of HLS** would cut video latency to sub-second (MediaMTX
  already serves it on `:8889` via WHEP) — worth it if the 1-2s HLS delay
  matters for your use case, but it's more integration work in the frontend.
- **Acknowledge/Silence buttons** in `AlarmPanel` are still just UI — wire
  them to send a message back over the WebSocket if you want dispatcher-side
  acknowledgment to actually suppress/reset an alarm in the backend.
- **TLS/auth on the WebSocket** — right now it's open, unauthenticated
  `ws://`. Fine on a private in-cab network; add a token or `wss://` behind a
  reverse proxy before exposing it more broadly.
- The removed "Train Context" panel (train ID, route, driver name) was purely
  decorative mock data with no backend source — left out rather than
  re-faked. Wire it up if/when there's a real source (e.g. a train
  management system) to pull it from.
