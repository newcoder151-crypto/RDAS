import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { TopStatusBar } from "@/components/rdas/TopStatusBar";
import { VideoWidget } from "@/components/rdas/VideoWidget";
import { DriverStatusPanel } from "@/components/rdas/DriverStatusPanel";
import { AlarmPanel } from "@/components/rdas/AlarmPanel";
import { ActiveAlarmsList } from "@/components/rdas/ActiveAlarmsList";
import { EventLogWidget } from "@/components/rdas/EventLogWidget";
import { highestAlarm } from "@/lib/rdas-backend";
import { useRdasSocket } from "@/hooks/use-rdas-socket";

export const Route = createFileRoute("/")({
  component: MainWindow,
});

type TabKey = "live" | "telemetry";

function MainWindow() {
  const [clock, setClock] = useState("--:--:--");
  const [tab, setTab] = useState<TabKey>("live");
  const rdas = useRdasSocket();

  useEffect(() => {
    const update = () => setClock(new Date().toISOString().slice(11, 19));
    update();
    const i = setInterval(update, 1000);
    return () => clearInterval(i);
  }, []);

  const currentAlarm = useMemo(() => highestAlarm(rdas.alarms), [rdas.alarms]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-foreground">
      <TopStatusBar
        clock={clock}
        connections={[
          { label: "Camera", status: rdas.personDetected ? "online" : "degraded", meta: rdas.connected ? "1080p" : "n/a" },
          { label: "AI Model", status: rdas.calibrated ? "online" : "degraded", meta: rdas.calibrated ? "calibrated" : "calibrating" },
          { label: "Backend", status: rdas.connected ? "online" : "offline", meta: rdas.connected ? "ws" : "retrying" },
          { label: "Alarm Feed", status: rdas.usingMockData ? "offline" : "online", meta: rdas.usingMockData ? "demo data" : "live" },
        ]}
      />

      <nav className="flex items-center gap-1 border-b border-panel-border bg-panel-header/40 px-3">
        <TabButton active={tab === "live"} onClick={() => setTab("live")}>
          Live Feed
        </TabButton>
        <TabButton active={tab === "telemetry"} onClick={() => setTab("telemetry")}>
          Telemetry
        </TabButton>
      </nav>

      {tab === "live" ? (
        <main className="flex min-h-0 flex-1 flex-col gap-3 p-3">
          <div className="flex min-h-0 flex-1 flex-col">
            <VideoWidget
              fps={rdas.metrics?.fps ?? 0}
              resolution="1920×1080"
              faceBox={rdas.faceBox}
              connected={rdas.connected}
            />
          </div>
          <div className="grid shrink-0 grid-cols-1 gap-3 lg:grid-cols-[1fr_320px]">
            <AlarmPanel
              alarm={currentAlarm}
              disabled={!rdas.connected}
              onAcknowledge={() => rdas.sendCommand({ action: "ack", code: currentAlarm.code })}
              onSilence={() => rdas.sendCommand({ action: "silence", code: currentAlarm.code, duration_sec: 60 })}
            />
            <ActiveAlarmsList alarms={rdas.alarms} />
          </div>
        </main>
      ) : (
        <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-auto p-3 lg:grid-cols-[1fr_360px]">
          <div className="flex min-h-0 min-w-0 flex-col gap-3">
            <div className="flex min-h-[240px] flex-1 flex-col">
              <EventLogWidget events={rdas.events} />
            </div>
          </div>
          <aside className="flex min-h-0 flex-col gap-3">
            <DriverStatusPanel signals={rdas.signals} />
            <div className="panel p-3">
              <div className="mono mb-2 text-[10px] uppercase tracking-widest text-muted-foreground">
                System Metrics
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <Metric label="FPS" value={rdas.metrics?.fps != null ? rdas.metrics.fps.toFixed(1) : "--"} />
                <Metric label="EAR" value={rdas.metrics?.ear != null ? rdas.metrics.ear.toFixed(3) : "--"} />
                <Metric label="MAR" value={rdas.metrics?.mar != null ? rdas.metrics.mar.toFixed(3) : "--"} />
                <Metric label="Pitch" value={rdas.metrics?.pitch != null ? rdas.metrics.pitch.toFixed(3) : "--"} />
                <Metric label="Calibrated" value={rdas.calibrated ? "yes" : "no"} />
                <Metric label="Video t" value={rdas.videoT != null ? `${rdas.videoT.toFixed(1)}s` : "--"} />
              </div>
            </div>
            {rdas.usingMockData && (
              <div className="panel p-3 text-sm text-status-warning">
                Showing demo data — no WebSocket connection to the backend yet. Start
                <code className="mono mx-1 rounded-sm bg-panel-header/60 px-1">stream_pipeline.py</code>
                and confirm <code className="mono">VITE_RDAS_WS_URL</code> matches its host/port.
              </div>
            )}
          </aside>
        </main>
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`mono relative px-4 py-2 text-[11px] uppercase tracking-widest transition-colors ${
        active
          ? "text-primary"
          : "text-muted-foreground hover:text-foreground"
      }`}
    >
      {children}
      {active && (
        <span className="absolute inset-x-2 -bottom-px h-0.5 bg-primary" />
      )}
    </button>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-panel-border bg-panel-header/40 px-2.5 py-1.5">
      <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mono text-base font-semibold">{value}</div>
    </div>
  );
}
