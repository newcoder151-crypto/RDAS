import { StatusDot } from "./StatusDot";

type Conn = { label: string; status: "online" | "degraded" | "offline"; meta?: string };

const statusToLevel = (s: Conn["status"]) =>
  s === "online" ? "normal" : s === "degraded" ? "warning" : "critical";

export function TopStatusBar({
  connections,
  clock,
}: {
  connections: Conn[];
  clock: string;
}) {
  return (
    <header className="flex items-center justify-between border-b border-panel-border bg-panel-header/60 px-4 py-2 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex h-7 w-7 items-center justify-center rounded-sm bg-primary/15 text-primary">
          <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" stroke="currentColor" strokeWidth="2">
            <path d="M4 4h16v12H4z" />
            <path d="M8 20h8M12 16v4" />
            <circle cx="8.5" cy="12.5" r="1" fill="currentColor" />
            <circle cx="15.5" cy="12.5" r="1" fill="currentColor" />
          </svg>
        </div>
        <div className="leading-tight">
          <div className="mono text-[11px] tracking-[0.25em] text-muted-foreground">RDAS · v2.4.1</div>
          <div className="text-sm font-semibold">Railway Driver Assistance System</div>
        </div>
      </div>

      <div className="flex items-center gap-5">
        {connections.map((c) => (
          <div key={c.label} className="flex items-center gap-2">
            <StatusDot level={statusToLevel(c.status)} pulse={c.status === "online"} />
            <div className="leading-tight">
              <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                {c.label}
              </div>
              <div className="mono text-xs">
                {c.status.toUpperCase()}
                {c.meta && <span className="ml-1 text-muted-foreground">· {c.meta}</span>}
              </div>
            </div>
          </div>
        ))}
        <div className="ml-3 border-l border-panel-border pl-4">
          <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">UTC</div>
          <div className="mono text-xs">{clock}</div>
        </div>
      </div>
    </header>
  );
}
