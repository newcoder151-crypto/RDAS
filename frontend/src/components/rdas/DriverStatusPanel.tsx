import { StatusDot } from "./StatusDot";

export type DriverSignal = {
  key: string;
  label: string;
  active: boolean;
  level: "normal" | "warning" | "high" | "critical" | "off";
  value?: string;
};

export function DriverStatusPanel({ signals }: { signals: DriverSignal[] }) {
  return (
    <div className="panel flex flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>Driver Status</span>
        <span className="mono text-[10px] normal-case tracking-normal text-status-normal">● live</span>
      </div>
      <div className="divide-y divide-panel-border">
        {signals.map((s) => (
          <div
            key={s.key}
            className={`flex items-center justify-between px-3 py-2.5 transition-colors ${
              s.active && s.level !== "normal" ? "bg-panel-header/60" : ""
            }`}
          >
            <div className="flex items-center gap-3">
              <StatusDot level={s.active ? s.level : "off"} pulse={s.active && s.level !== "normal"} />
              <div className="leading-tight">
                <div className="text-sm font-medium">{s.label}</div>
                {s.value && (
                  <div className="mono text-[10px] text-muted-foreground">{s.value}</div>
                )}
              </div>
            </div>
            <span
              className={`mono text-[10px] uppercase tracking-widest ${
                s.active
                  ? s.level === "normal"
                    ? "text-status-normal"
                    : s.level === "warning"
                    ? "text-status-warning"
                    : s.level === "high"
                    ? "text-status-high"
                    : "text-status-critical"
                  : "text-muted-foreground"
              }`}
            >
              {s.active ? (s.level === "normal" ? "OK" : "ALERT") : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
