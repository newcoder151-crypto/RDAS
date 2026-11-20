export type AlarmLevel = "normal" | "warning" | "high" | "critical";

export type Alarm = {
  code: string;
  title: string;
  detail: string;
  level: AlarmLevel;
  since: string;
  /** Set by the backend once a dispatcher has hit ACKNOWLEDGE for this alarm's current occurrence. */
  acked?: boolean;
};

const levelStyles: Record<AlarmLevel, { bg: string; ring: string; text: string; badge: string }> = {
  normal: {
    bg: "bg-status-normal/10",
    ring: "ring-status-normal/40",
    text: "text-status-normal",
    badge: "bg-status-normal text-black",
  },
  warning: {
    bg: "bg-status-warning/12",
    ring: "ring-status-warning/40",
    text: "text-status-warning",
    badge: "bg-status-warning text-black",
  },
  high: {
    bg: "bg-status-high/15",
    ring: "ring-status-high/50",
    text: "text-status-high",
    badge: "bg-status-high text-black",
  },
  critical: {
    bg: "bg-status-critical/20",
    ring: "ring-status-critical/60",
    text: "text-status-critical",
    badge: "bg-status-critical text-white",
  },
};

export function AlarmPanel({
  alarm,
  onAcknowledge,
  onSilence,
  disabled = false,
}: {
  alarm: Alarm;
  /** Called when ACKNOWLEDGE is clicked. Omit to hide the button's behavior entirely (it'll just render disabled). */
  onAcknowledge?: () => void;
  /** Called when SILENCE 60s is clicked. */
  onSilence?: () => void;
  /** Force both buttons disabled regardless of alarm state — e.g. no live backend connection. */
  disabled?: boolean;
}) {
  const s = levelStyles[alarm.level];
  const blink = alarm.level === "critical" && !alarm.acked;
  const isRealAlarm = alarm.level !== "normal";
  const ackDisabled = disabled || !isRealAlarm || !!alarm.acked || !onAcknowledge;
  const silenceDisabled = disabled || !isRealAlarm || !onSilence;
  return (
    <div className={`panel overflow-hidden`}>
      <div className="panel-header flex items-center justify-between">
        <span>Current Alarm · Highest Priority</span>
        <span className="mono text-[10px] normal-case tracking-normal text-muted-foreground">
          since {alarm.since}
        </span>
      </div>
      <div
        className={`relative flex items-center gap-6 px-6 py-5 ring-1 ring-inset ${s.bg} ${s.ring}`}
        style={blink ? { animation: "blink-critical 1.2s ease-in-out infinite" } : undefined}
      >
        <div className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-sm ${s.badge}`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="h-8 w-8">
            {alarm.level === "normal" ? (
              <path d="M5 12l5 5L20 7" />
            ) : (
              <>
                <path d="M12 3l10 18H2z" />
                <path d="M12 10v5M12 18h.01" />
              </>
            )}
          </svg>
        </div>
        <div className="min-w-0 flex-1">
          <div className="mono flex items-center gap-3 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span className={`rounded-sm px-1.5 py-0.5 ${s.badge}`}>{alarm.level}</span>
            <span>{alarm.code}</span>
            {alarm.acked && (
              <span className="rounded-sm border border-current px-1.5 py-0.5 text-muted-foreground">
                ACKED
              </span>
            )}
          </div>
          <div className={`mt-1 truncate text-3xl font-bold tracking-tight ${s.text}`}>
            {alarm.title}
          </div>
          <div className="mt-1 truncate text-sm text-muted-foreground">{alarm.detail}</div>
        </div>
        <div className="hidden shrink-0 flex-col items-end gap-2 md:flex">
          <button
            onClick={onAcknowledge}
            disabled={ackDisabled}
            className={`mono rounded-sm px-4 py-2 text-xs font-semibold transition-opacity ${s.badge} ${
              ackDisabled ? "cursor-not-allowed opacity-40" : "hover:opacity-90"
            }`}
          >
            {alarm.acked ? "ACKNOWLEDGED" : "ACKNOWLEDGE"}
          </button>
          <button
            onClick={onSilence}
            disabled={silenceDisabled}
            className={`mono rounded-sm border border-panel-border px-4 py-2 text-xs text-muted-foreground transition-colors ${
              silenceDisabled ? "cursor-not-allowed opacity-40" : "hover:text-foreground"
            }`}
          >
            SILENCE 60s
          </button>
        </div>
      </div>
    </div>
  );
}
