import type { AlarmLevel } from "./AlarmPanel";

export type LogEvent = {
  id: string;
  ts: string;
  event: string;
  confidence: number;
  level: AlarmLevel;
};

const levelText: Record<AlarmLevel, string> = {
  normal: "text-status-normal",
  warning: "text-status-warning",
  high: "text-status-high",
  critical: "text-status-critical",
};

export function EventLogWidget({ events }: { events: LogEvent[] }) {
  return (
    <div className="panel flex min-h-0 flex-col">
      <div className="panel-header flex items-center justify-between">
        <span>Event Log</span>
        <span className="mono text-[10px] normal-case tracking-normal text-muted-foreground">
          {events.length} events · auto-scroll
        </span>
      </div>
      <div className="grid grid-cols-[140px_1fr_100px_90px] gap-2 border-b border-panel-border bg-panel-header/40 px-3 py-1.5 mono text-[10px] uppercase tracking-widest text-muted-foreground">
        <span>Timestamp</span>
        <span>Event</span>
        <span className="text-right">Confidence</span>
        <span className="text-right">Level</span>
      </div>
      <div className="min-h-0 flex-1 overflow-auto">
        {events.map((e) => (
          <div
            key={e.id}
            className="grid grid-cols-[140px_1fr_100px_90px] gap-2 border-b border-panel-border/60 px-3 py-1.5 text-sm hover:bg-panel-header/40"
          >
            <span className="mono text-xs text-muted-foreground">{e.ts}</span>
            <span className="truncate">{e.event}</span>
            <span className="mono text-right text-xs">
              {(e.confidence * 100).toFixed(1)}%
            </span>
            <span className={`mono text-right text-[10px] uppercase tracking-widest ${levelText[e.level]}`}>
              {e.level}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
