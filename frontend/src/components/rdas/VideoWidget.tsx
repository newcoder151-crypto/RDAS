import { useEffect, useRef, useState } from "react";
import Hls from "hls.js";

export type FaceBox = { x: number; y: number; w: number; h: number } | null;

// Matches MediaMTX's default HLS output for whatever path stream_pipeline.py
// publishes to (RDASConfig.output_video_path = rtsp://localhost:8554/rdas
// -> MediaMTX also serves that same path as HLS on :8888 and WebRTC on :8889).
const DEFAULT_HLS_URL = (import.meta.env.VITE_RDAS_HLS_URL as string | undefined) ?? "http://localhost:8888/rdas/index.m3u8";

type Props = {
  fps: number;
  resolution: string;
  faceBox?: FaceBox;
  connected?: boolean;
  hlsUrl?: string;
};

export function VideoWidget({ fps, resolution, faceBox = null, connected = false, hlsUrl = DEFAULT_HLS_URL }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streamOk, setStreamOk] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const i = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(i);
  }, []);

  // Attach the real MediaMTX HLS stream. Falls back silently to the
  // simulated placeholder below if the stream isn't reachable yet (backend
  // not running, MediaMTX not started, or camera not connected).
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let hls: Hls | null = null;
    let cancelled = false;

    function markOk() {
      if (!cancelled) setStreamOk(true);
    }
    function markDown() {
      if (!cancelled) setStreamOk(false);
    }

    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      // Safari: native HLS support, no hls.js needed.
      video.src = hlsUrl;
      video.addEventListener("loadeddata", markOk);
      video.addEventListener("error", markDown);
      video.play().catch(() => {});
    } else if (Hls.isSupported()) {
      hls = new Hls({ maxLiveSyncPlaybackRate: 1.5 });
      hls.loadSource(hlsUrl);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        markOk();
        video.play().catch(() => {});
      });
      hls.on(Hls.Events.ERROR, (_evt, data) => {
        if (data.fatal) markDown();
      });
    } else {
      markDown();
    }

    return () => {
      cancelled = true;
      hls?.destroy();
    };
  }, [hlsUrl]);

  const showPlaceholder = !streamOk;

  return (
    <div className="panel flex flex-1 flex-col overflow-hidden">
      <div className="panel-header flex items-center justify-between">
        <span>Live Feed · Cab Camera 01</span>
        <span className="flex items-center gap-3 normal-case tracking-normal">
          <span className="mono text-[10px] text-muted-foreground">{resolution}</span>
          <span className={`mono text-[10px] ${streamOk ? "text-status-normal" : "text-status-warning"}`}>
            {streamOk ? "● LIVE" : "● NO SIGNAL"}
          </span>
          <span className="mono text-[10px] text-muted-foreground">{fps.toFixed(1)} FPS</span>
        </span>
      </div>

      <div className="relative flex-1 overflow-hidden bg-black">
        {/* Real video element — always mounted so hls.js can attach to it */}
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          className={`absolute inset-0 h-full w-full object-cover transition-opacity ${
            showPlaceholder ? "opacity-0" : "opacity-100"
          }`}
        />

        {/* Simulated placeholder — shown only until the real stream connects,
            so the HUD still demos meaningfully with the backend offline */}
        {showPlaceholder && (
          <>
            <div
              className="absolute inset-0"
              style={{
                backgroundImage:
                  "radial-gradient(ellipse at 50% 40%, oklch(0.35 0.02 240) 0%, oklch(0.14 0.01 240) 60%, oklch(0.08 0 0) 100%)",
              }}
            />
            <div
              className="absolute inset-0 opacity-30 mix-blend-screen"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(0deg, transparent 0 3px, oklch(1 0 0 / 0.04) 3px 4px)",
              }}
            />
            <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full opacity-40" preserveAspectRatio="xMidYMid slice">
              <ellipse cx="50" cy="42" rx="14" ry="18" fill="oklch(0.55 0.04 40)" />
              <path d="M20 100 Q20 68 50 66 Q80 68 80 100 Z" fill="oklch(0.3 0.03 250)" />
            </svg>
            <div
              className="pointer-events-none absolute inset-x-0 h-16 bg-gradient-to-b from-transparent via-primary/15 to-transparent"
              style={{ animation: "scan 4s linear infinite" }}
            />
            <div className="absolute inset-x-3 top-3 mono rounded-sm bg-black/60 px-2 py-1 text-center text-[10px] text-status-warning backdrop-blur">
              Waiting for backend stream ({hlsUrl}) — showing simulated feed
            </div>
          </>
        )}

        {/* Real face/person bounding box, driven by the backend's detector
            output over the WebSocket — not a hardcoded mock box */}
        {faceBox && (
          <div
            className="absolute border-2"
            style={{
              left: `${faceBox.x}%`,
              top: `${faceBox.y}%`,
              width: `${faceBox.w}%`,
              height: `${faceBox.h}%`,
              borderColor: "var(--status-normal)",
              boxShadow: "0 0 0 1px oklch(0 0 0 / 0.5), 0 0 12px -2px var(--status-normal)",
            }}
          >
            <div
              className="mono absolute -top-5 left-0 px-1.5 py-0.5 text-[10px] font-semibold"
              style={{ backgroundColor: "var(--status-normal)", color: "oklch(0.1 0 0)" }}
            >
              driver
            </div>
          </div>
        )}

        {/* HUD corners */}
        <div className="pointer-events-none absolute inset-3">
          {["top-0 left-0", "top-0 right-0", "bottom-0 left-0", "bottom-0 right-0"].map((p) => (
            <div key={p} className={`absolute h-4 w-4 border-primary/60 ${p}
              ${p.includes("top") ? "border-t-2" : "border-b-2"}
              ${p.includes("left") ? "border-l-2" : "border-r-2"}`} />
          ))}
        </div>

        {/* Bottom HUD */}
        <div className="absolute inset-x-3 bottom-3 flex items-center justify-between">
          <div className="mono flex gap-4 rounded-sm bg-black/60 px-2 py-1 text-[10px] text-primary/90 backdrop-blur">
            <span>YOLO26 + MediaPipe</span>
            <span>{connected ? "TELEMETRY LIVE" : "TELEMETRY OFFLINE"}</span>
          </div>
          <div className="mono rounded-sm bg-black/60 px-2 py-1 text-[10px] text-primary/90 backdrop-blur">
            {String(tick).padStart(6, "0")}
          </div>
        </div>
      </div>
    </div>
  );
}
