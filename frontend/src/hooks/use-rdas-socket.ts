import { useEffect, useRef, useState } from "react";
import type { Alarm } from "@/components/rdas/AlarmPanel";
import type { DriverSignal } from "@/components/rdas/DriverStatusPanel";
import type { LogEvent } from "@/components/rdas/EventLogWidget";
import { initialAlarms, initialEvents, initialSignals } from "@/lib/rdas-backend";

// The default matches AlarmBroadcastServer's default host/port in
// stream_pipeline.py (RDASConfig.ws_host / ws_port). Override at build time
// with VITE_RDAS_WS_URL if the backend runs on a different host/port
// (e.g. ws://192.168.1.50:8765 for a backend on another machine).
const DEFAULT_WS_URL = (import.meta.env.VITE_RDAS_WS_URL as string | undefined) ?? "ws://localhost:8765";

const NO_ALARM: Alarm = {
  code: "NOMINAL",
  title: "All Systems Normal",
  detail: "No active alarms",
  level: "normal",
  since: "00:00:00",
};

export type RdasMetrics = {
  fps: number | null;
  ear: number | null;
  mar: number | null;
  pitch: number | null;
};

export type RdasSocketState = {
  /** True while a live WebSocket connection is open. */
  connected: boolean;
  /** True until the very first successful connection — i.e. we're still showing demo/mock data. */
  usingMockData: boolean;
  alarms: Alarm[];
  signals: DriverSignal[];
  events: LogEvent[];
  metrics: RdasMetrics | null;
  calibrated: boolean;
  personDetected: boolean;
  /** Video-relative timestamp (seconds) reported by the backend, if connected. */
  videoT: number | null;
  /** Real detector-derived bounding box (normalized %), or null if no person detected. */
  faceBox: { x: number; y: number; w: number; h: number } | null;
  /**
   * Sends a command back to AlarmBroadcastServer (backend/stream_pipeline.py),
   * e.g. { action: "ack", code: "yawn" } or
   * { action: "silence", code: "yawn", duration_sec: 60 }. No-op (with a
   * console warning) if the socket isn't currently open — there's nothing
   * useful to queue client-side for a dispatcher action like this.
   */
  sendCommand: (cmd: Record<string, unknown>) => void;
};

export function useRdasSocket(url: string = DEFAULT_WS_URL): RdasSocketState {
  const [connected, setConnected] = useState(false);
  const [everConnected, setEverConnected] = useState(false);
  const [alarms, setAlarms] = useState<Alarm[]>(initialAlarms);
  const [signals, setSignals] = useState<DriverSignal[]>(initialSignals);
  const [events, setEvents] = useState<LogEvent[]>(initialEvents);
  const [metrics, setMetrics] = useState<RdasMetrics | null>(null);
  const [calibrated, setCalibrated] = useState(false);
  const [personDetected, setPersonDetected] = useState(false);
  const [videoT, setVideoT] = useState<number | null>(null);
  const [faceBox, setFaceBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    let cancelled = false;

    function connect() {
      if (cancelled) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch {
        scheduleRetry();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setConnected(true);
        setEverConnected(true);
      };

      ws.onmessage = (evt) => {
        let msg: any;
        try {
          msg = JSON.parse(evt.data);
        } catch {
          return; // malformed frame — drop it, wait for the next one
        }
        if (msg.type !== "state") return;

        if (Array.isArray(msg.alarms)) setAlarms(msg.alarms.length ? msg.alarms : [NO_ALARM]);
        if (Array.isArray(msg.signals)) setSignals(msg.signals);
        if (Array.isArray(msg.events)) {
          setEvents(
            msg.events.map((e: any) => ({
              id: String(e.id),
              ts: e.ts,
              event: e.message ?? e.event,
              confidence: 1,
              level: e.level,
            })),
          );
        }
        if (msg.metrics) setMetrics(msg.metrics);
        setCalibrated(!!msg.calibrated);
        setPersonDetected(!!msg.person_detected);
        if (typeof msg.video_t === "number") setVideoT(msg.video_t);
        setFaceBox(msg.face_box ?? null);
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        scheduleRetry();
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    function scheduleRetry() {
      if (cancelled) return;
      const delay = Math.min(10000, 1000 * 2 ** retryRef.current);
      retryRef.current += 1;
      timerRef.current = setTimeout(connect, delay);
    }

    connect();
    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, [url]);

  function sendCommand(cmd: Record<string, unknown>) {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn("[rdas-socket] cannot send command — WebSocket is not open", cmd);
      return;
    }
    ws.send(JSON.stringify(cmd));
  }

  return {
    connected,
    usingMockData: !everConnected,
    alarms: alarms.length ? alarms : [NO_ALARM],
    signals,
    events,
    metrics,
    calibrated,
    personDetected,
    videoT,
    faceBox,
    sendCommand,
  };
}
