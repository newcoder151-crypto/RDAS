import type { Alarm, AlarmLevel } from "./AlarmPanel";
import { StatusDot } from "./StatusDot";

const levelText: Record<AlarmLevel, string> = {
  normal: "text-status-normal",
  warning: "text-status-warning",
  high: "text-status-high",
  critical: "text-status-critical",
};

/**
 * Shows every alarm that is currently active, each as its own row — the
 * AlarmPanel above only ever shows the single highest-priority alarm, so a
 * second simultaneous alarm (e.g. yawning while also distracted) would
 * otherwise be invisible in the UI even though the backend is tracking both.
 */
export function ActiveAlarmsList({ alarms }: { alarms: Alarm[] }) {
  const active = alarms.filter((a) => a.level !== "normal");

  return (
    <div className="panel flex flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>Active Alarms</span>
        <span className="mono text-[10px] normal-case tracking-normal text-muted-foreground">
          {active.length} active
        </span>
      </div>
      {active.length === 0 ? (
        <div className="px-3 py-4 text-center text-sm text-muted-foreground">No active alarms</div>
      ) : (
        <div className="divide-y divide-panel-border">
          {active.map((a) => (
            <div key={a.code} className="flex items-center justify-between gap-3 px-3 py-2.5">
              <div className="flex min-w-0 items-center gap-3">
                <StatusDot level={a.level} pulse={a.level === "critical"} />
                <div className="min-w-0 leading-tight">
                  <div className="truncate text-sm font-medium">{a.title}</div>
                  <div className="mono truncate text-[10px] text-muted-foreground">{a.detail}</div>
                </div>
              </div>
              <div className="shrink-0 text-right">
                <div className={`mono text-[10px] uppercase tracking-widest ${levelText[a.level]}`}>
                  {a.level}
                </div>
                <div className="mono text-[10px] text-muted-foreground">since {a.since}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
