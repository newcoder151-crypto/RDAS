type Level = "normal" | "warning" | "high" | "critical" | "off";

const colorMap: Record<Level, string> = {
  normal: "bg-status-normal text-status-normal",
  warning: "bg-status-warning text-status-warning",
  high: "bg-status-high text-status-high",
  critical: "bg-status-critical text-status-critical",
  off: "bg-status-off text-status-off",
};

export function StatusDot({ level, pulse = false }: { level: Level; pulse?: boolean }) {
  const [bg] = colorMap[level].split(" ");
  return (
    <span className="relative inline-flex h-2.5 w-2.5">
      {pulse && level !== "off" && (
        <span
          className={`absolute inset-0 rounded-full ${bg} opacity-60 animate-ping`}
        />
      )}
      <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${bg}`} />
    </span>
  );
}
