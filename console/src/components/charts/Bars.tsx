import { cn } from "@/lib/utils";

/** Horizontal comparison bars. `better` says which direction wins so colour and ranking stay honest. */
export function HBars({ rows, format = (v: number) => v.toFixed(2), better = "lower", highlight }: {
  rows: { label: string; value: number; note?: string }[]; format?: (v: number) => string; better?: "lower" | "higher"; highlight?: string }) {
  const max = Math.max(...rows.map((r) => r.value), 1e-9);
  const best = (better === "lower" ? Math.min : Math.max)(...rows.map((r) => r.value));
  return (
    <ul className="space-y-2">
      {rows.map((r) => (
        <li key={r.label} className="grid grid-cols-[minmax(9rem,38%)_1fr_4rem] items-center gap-3 text-sm">
          <span className={cn("leading-snug", r.label === highlight && "font-semibold")}>{r.label}</span>
          <div className="h-5 rounded-sm bg-muted" role="img" aria-label={`${r.label} ${format(r.value)}`}>
            <div className={cn("h-full rounded-sm", r.value === best ? "bg-primary" : r.label === highlight ? "bg-teal" : "bg-slate-400 dark:bg-slate-500")} style={{ width: `${Math.max(2, (r.value / max) * 100)}%` }} />
          </div>
          <span className="text-right tabular font-medium">{format(r.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Calibration: for each quantile, the share of actuals that fell at or below it. A perfect model sits on the diagonal. */
export function Calibration({ observed }: { observed: number[] }) {
  const W = 260, H = 220, p = 30;
  const X = (v: number) => p + v * (W - p - 10), Y = (v: number) => H - p - v * (H - p - 10);
  const nominal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-xs" role="img" aria-label="Calibration of forecast quantiles">
      <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} stroke="var(--chart-baseline)" strokeDasharray="4 3" />
      <path d={observed.map((o, i) => `${i ? "L" : "M"}${X(nominal[i])},${Y(o)}`).join(" ")} fill="none" stroke="var(--chart-forecast)" strokeWidth="2.25" />
      {observed.map((o, i) => <circle key={i} cx={X(nominal[i])} cy={Y(o)} r="3.5" fill="var(--chart-forecast)"><title>{`Forecast ${nominal[i] * 100}% level: ${(o * 100).toFixed(0)}% of actuals fell below`}</title></circle>)}
      {[0, 0.5, 1].map((v) => <g key={v}><text x={X(v)} y={H - 10} textAnchor="middle" fontSize="10" fill="var(--muted-foreground)">{v * 100}%</text><text x={p - 6} y={Y(v) + 3} textAnchor="end" fontSize="10" fill="var(--muted-foreground)">{v * 100}%</text></g>)}
      <text x={W / 2} y={H - 0} textAnchor="middle" fontSize="9" fill="var(--muted-foreground)">What the forecast said</text>
      <text transform={`translate(8 ${H / 2}) rotate(-90)`} textAnchor="middle" fontSize="9" fill="var(--muted-foreground)">What actually happened</text>
    </svg>
  );
}

export function Sparkline({ y, className }: { y: number[]; className?: string }) {
  if (y.length < 2) return null;
  const W = 120, H = 28, max = Math.max(...y, 1e-9), min = Math.min(...y, 0);
  const d = y.map((v, i) => `${i ? "L" : "M"}${(i / (y.length - 1)) * W},${H - 2 - ((v - min) / (max - min || 1)) * (H - 4)}`).join(" ");
  return <svg viewBox={`0 0 ${W} ${H}`} className={className ?? "h-7 w-28"} aria-hidden><path d={d} fill="none" stroke="var(--chart-forecast)" strokeWidth="1.75" /></svg>;
}
