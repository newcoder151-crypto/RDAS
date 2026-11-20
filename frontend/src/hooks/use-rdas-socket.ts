import { useEffect, useRef, useState } from "react";
import type { Alarm } from "@/components/rdas/AlarmPanel";
import type { DriverSignal } from "@/components/rdas/DriverStatusPanel";
import type { LogEvent } from "@/components/rdas/EventLogWidget";
import { initialAlarms, initialEvents, initialSignals } from "@/lib/rdas-backend";
import { ALARM_SOUND_CONFIG } from "@/lib/alarm-sound-config";

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
  /**
   * Acknowledge one alarm. Marks it acked locally right away (so the sound
   * stops instantly) and tells the backend; the backend's next state frame
   * then confirms it.
   */
  acknowledge: (code: string) => void;
  /** Acknowledge every currently active, unacknowledged alarm. */
  acknowledgeAll: () => void;
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
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // code -> time (ms) the dispatcher hit ACK. Covers the short window where a
  // state frame already in flight from the backend still says acked=false.
  const pendingAcksRef = useRef<Map<string, number>>(new Map());
  // Acknowledged alarms disappear from the live views after a short delay (they
  // stay in the event log). ackTimersRef holds one removal timer per alarm code,
  // and hiddenCodes is the set whose timer has fired.
  const ackTimersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  const [hiddenCodes, setHiddenCodes] = useState<Set<string>>(new Set());

  function trackAcked(code: string) {
    if (ackTimersRef.current.has(code)) return;
    const timer = setTimeout(() => {
      setHiddenCodes((prev) => new Set(prev).add(code));
    }, ALARM_SOUND_CONFIG.ackedVisibleMs);
    ackTimersRef.current.set(code, timer);
  }

  function untrackAcked(code: string) {
    const timer = ackTimersRef.current.get(code);
    if (timer) clearTimeout(timer);
    ackTimersRef.current.delete(code);
    setHiddenCodes((prev) => {
      if (!prev.has(code)) return prev;
      const next = new Set(prev);
      next.delete(code);
      return next;
    });
  }

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

        if (Array.isArray(msg.alarms)) {
          const now = Date.now();
          const pending = pendingAcksRef.current;
          const incoming: Alarm[] = msg.alarms.map((a: Alarm) => {
            const ackedAt = pending.get(a.code);
            if (ackedAt === undefined) return a;
            if (a.acked || now - ackedAt > ALARM_SOUND_CONFIG.ackGraceMs) {
              pending.delete(a.code); // backend caught up (or grace expired): trust the backend
              return a;
            }
            return { ...a, acked: true };
          });
          // forget pending acks for alarms that are no longer active
          for (const code of [...pending.keys()]) {
            if (!incoming.some((a) => a.code === code)) pending.delete(code);
          }
          // Start / reset the on-screen countdown for acknowledged alarms.
          for (const a of incoming) {
            if (a.acked) trackAcked(a.code);
            else untrackAcked(a.code); // fresh occurrence of a previously acked alarm
          }
          for (const code of [...ackTimersRef.current.keys()]) {
            if (!incoming.some((a) => a.code === code)) untrackAcked(code); // alarm cleared
          }
          setAlarms(incoming.length ? incoming : [NO_ALARM]);
        }
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
      ackTimersRef.current.forEach((t) => clearTimeout(t));
      ackTimersRef.current.clear();
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

  function acknowledge(code: string) {
    pendingAcksRef.current.set(code, Date.now());
    trackAcked(code);
    setAlarms((prev) => prev.map((a) => (a.code === code ? { ...a, acked: true } : a)));
    sendCommand({ action: "ack", code });
  }

  function acknowledgeAll() {
    alarms.filter((a) => a.level !== "normal" && !a.acked).forEach((a) => acknowledge(a.code));
  }

  const visibleAlarms = alarms.filter((a) => !hiddenCodes.has(a.code));

  return {
    connected,
    usingMockData: !everConnected,
    alarms: visibleAlarms.length ? visibleAlarms : [NO_ALARM],
    signals,
    events,
    metrics,
    calibrated,
    personDetected,
    videoT,
    faceBox,
    sendCommand,
    acknowledge,
    acknowledgeAll,
  };
}
