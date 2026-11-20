import type { Alarm, AlarmLevel } from "./AlarmPanel";
import { StatusDot } from "./StatusDot";

const levelText: Record<AlarmLevel, string> = {
  normal: "text-status-normal",
  warning: "text-status-warning",
  high: "text-status-high",
  critical: "text-status-critical",
};

/**
 * Shows every alarm that is currently active, each as its own row with its own
 * ACK button — the AlarmPanel above only ever shows the single highest-priority
 * alarm, so a second simultaneous alarm (e.g. yawning while also distracted)
 * would otherwise be invisible in the UI even though the backend tracks both.
 *
 * The acoustic alarm sounds while any row here is unacknowledged and stops as
 * soon as the last one is acknowledged.
 */
export function ActiveAlarmsList({
  alarms,
  onAcknowledge,
  onAcknowledgeAll,
  disabled = false,
  sound,
}: {
  alarms: Alarm[];
  onAcknowledge?: (code: string) => void;
  onAcknowledgeAll?: () => void;
  /** Disable all ack buttons, e.g. no live backend connection. */
  disabled?: boolean;
  sound?: { sounding: boolean; muted: boolean; blocked: boolean; onToggle: () => void };
}) {
  const active = alarms.filter((a) => a.level !== "normal");
  const unackedCount = active.filter((a) => !a.acked).length;

  const soundLabel = sound?.blocked ? "ENABLE SOUND" : sound?.muted ? "MUTED" : sound?.sounding ? "SOUNDING" : "SOUND ON";
  const soundTone = sound?.blocked
    ? "border-status-warning text-status-warning"
    : sound?.muted
      ? "border-panel-border text-muted-foreground"
      : sound?.sounding
        ? "border-status-critical text-status-critical"
        : "border-panel-border text-foreground";

  return (
    <div className="panel flex flex-col">
      <div className="panel-header flex items-center justify-between gap-2">
        <span>Active Alarms</span>
        <div className="flex items-center gap-2">
          {sound && (
            <button
              onClick={sound.onToggle}
              title={sound.blocked ? "Browser is blocking audio — click to enable alarm sound" : "Mute / unmute alarm sound"}
              className={`mono rounded-sm border px-2 py-0.5 text-[10px] normal-case tracking-normal ${soundTone} ${
                sound.sounding ? "animate-pulse" : ""
              }`}
            >
              {sound.muted || sound.blocked ? "🔇" : "🔊"} {soundLabel}
            </button>
          )}
          {onAcknowledgeAll && (
            <button
              onClick={onAcknowledgeAll}
              disabled={disabled || unackedCount === 0}
              className={`mono rounded-sm border border-panel-border px-2 py-0.5 text-[10px] normal-case tracking-normal ${
                disabled || unackedCount === 0
                  ? "cursor-not-allowed opacity-40"
                  : "text-foreground hover:bg-muted"
              }`}
            >
              ACK ALL
            </button>
          )}
          <span className="mono text-[10px] normal-case tracking-normal text-muted-foreground">
            {active.length} active
          </span>
        </div>
      </div>
      {active.length === 0 ? (
        <div className="px-3 py-4 text-center text-sm text-muted-foreground">No active alarms</div>
      ) : (
        <div className="divide-y divide-panel-border">
          {active.map((a) => {
            const ackDisabled = disabled || !!a.acked || !onAcknowledge;
            return (
              <div
                key={a.code}
                className={`flex items-center justify-between gap-3 px-3 py-2.5 ${a.acked ? "opacity-60" : ""}`}
              >
                <div className="flex min-w-0 items-center gap-3">
                  <StatusDot level={a.level} pulse={a.level === "critical" && !a.acked} />
                  <div className="min-w-0 leading-tight">
                    <div className="truncate text-sm font-medium">{a.title}</div>
                    <div className="mono truncate text-[10px] text-muted-foreground">{a.detail}</div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-3">
                  <div className="text-right">
                    <div className={`mono text-[10px] uppercase tracking-widest ${levelText[a.level]}`}>
                      {a.level}
                    </div>
                    <div className="mono text-[10px] text-muted-foreground">since {a.since}</div>
                  </div>
                  <button
                    onClick={() => onAcknowledge?.(a.code)}
                    disabled={ackDisabled}
                    className={`mono w-12 rounded-sm px-2 py-1 text-[10px] font-semibold transition-opacity ${
                      a.acked
                        ? "cursor-default border border-panel-border text-muted-foreground"
                        : `bg-primary text-primary-foreground ${
                            ackDisabled ? "cursor-not-allowed opacity-40" : "hover:opacity-90"
                          }`
                    }`}
                  >
                    {a.acked ? "ACKED" : "ACK"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
