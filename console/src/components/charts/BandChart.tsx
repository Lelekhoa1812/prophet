import { useMemo, useState } from "react";
import { dayShort } from "@/lib/utils";

export interface BandSeries {
  history: { x: string[]; y: number[] };
  forecast?: { x: string[]; q10: number[]; q50: number[]; q90: number[]; q20?: number[]; q80?: number[] };
  baselines?: { name: string; x: string[]; y: number[] }[];
  actual?: { x: string[]; y: (number | null)[] };         // what really happened after the forecast
  truth?: { x: string[]; y: number[] };                   // twin only: true underlying demand
  events?: { start: string; end: string; label: string; kind?: string }[];
  unit?: string;
}

/** Hand-built SVG chart: history line, forecast median, 10-90 band, baselines, actuals, truth, event shading.
 *  Dates are positioned on a real time axis, hover shows every series at that date. */
export function BandChart({ s, height = 280, label }: { s: BandSeries; height?: number; label?: string }) {
  const W = 760, H = height, m = { l: 48, r: 14, t: 12, b: 26 };
  const [hover, setHover] = useState<number | null>(null);
  const t = (iso: string) => new Date(iso + "T00:00:00").getTime();
  const model = useMemo(() => {
    const xs = [...s.history.x, ...(s.forecast?.x ?? []), ...(s.truth?.x ?? [])].map(t);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const ys = [...s.history.y, ...(s.forecast?.q90 ?? []), ...(s.forecast?.q10 ?? []), ...(s.truth?.y ?? []), ...((s.actual?.y.filter((v) => v != null) as number[]) ?? [])];
    const y1 = Math.max(1, ...ys) * 1.08;
    return { x0, x1: x1 === x0 ? x0 + 1 : x1, y1 };
  }, [s]);
  const X = (iso: string) => m.l + ((t(iso) - model.x0) / (model.x1 - model.x0)) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - (v / model.y1) * (H - m.t - m.b);
  const line = (xs: string[], ys: (number | null)[]) => xs.map((x, i) => (ys[i] == null ? "" : `${i && ys[i - 1] != null ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i] as number).toFixed(1)}`)).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * model.y1);
  const all = useMemo(() => {
    const set = new Set<string>([...s.history.x, ...(s.forecast?.x ?? [])]);
    return [...set].sort();
  }, [s]);
  const idx = hover == null ? null : all[hover];
  const at = (xs: string[] | undefined, ys: (number | null)[] | undefined, d: string) => { const i = xs?.indexOf(d) ?? -1; return i >= 0 && ys ? ys[i] : null; };
  const fmt = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString("en-AU", { maximumFractionDigits: v < 10 ? 1 : 0 }));
  const band = s.forecast && s.forecast.x.length
    ? `${s.forecast.x.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(s.forecast!.q90[i]).toFixed(1)}`).join(" ")} ${[...s.forecast.x].reverse().map((x, i) => `L${X(x).toFixed(1)},${Y(s.forecast!.q10[s.forecast!.x.length - 1 - i]).toFixed(1)}`).join(" ")} Z`
    : null;
  return (
    <figure className="w-full" aria-label={label ?? "Demand history and forecast"}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full select-none" role="img"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const target = model.x0 + ((px - m.l) / (W - m.l - m.r)) * (model.x1 - model.x0);
          let best = 0, bd = Infinity;
          all.forEach((d, i) => { const dd = Math.abs(t(d) - target); if (dd < bd) { bd = dd; best = i; } });
          setHover(best);
        }}>
        {ticks.map((v, i) => (
          <g key={i}><line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="var(--chart-grid)" strokeWidth="1" />
            <text x={m.l - 6} y={Y(v) + 4} textAnchor="end" fontSize="10" fill="var(--muted-foreground)" className="tabular">{fmt(v)}</text></g>
        ))}
        {(s.events ?? []).map((e, i) => {
          const a = Math.max(m.l, X(e.start)), b = Math.min(W - m.r, Math.max(X(e.end), a + 3));
          return b > a ? <g key={i}><rect x={a} y={m.t} width={b - a} height={H - m.t - m.b} fill="var(--chart-truth)" opacity="0.13" /><title>{e.label}</title></g> : null;
        })}
        {s.forecast && s.forecast.x.length > 0 && <line x1={X(s.forecast.x[0])} x2={X(s.forecast.x[0])} y1={m.t} y2={H - m.b} stroke="var(--muted-foreground)" strokeDasharray="3 3" strokeWidth="1" />}
        {band && <path d={band} fill="var(--chart-band)" />}
        {(s.baselines ?? []).map((b) => <path key={b.name} d={line(b.x, b.y)} fill="none" stroke="var(--chart-baseline)" strokeWidth="1.25" strokeDasharray="4 3" />)}
        {s.truth && <path d={line(s.truth.x, s.truth.y)} fill="none" stroke="var(--chart-truth)" strokeWidth="1.5" strokeDasharray="1 3" strokeLinecap="round" />}
        <path d={line(s.history.x, s.history.y)} fill="none" stroke="var(--chart-history)" strokeWidth="1.75" strokeLinejoin="round" />
        {s.forecast && <path d={line(s.forecast.x, s.forecast.q50)} fill="none" stroke="var(--chart-forecast)" strokeWidth="2.25" />}
        {s.actual && s.actual.x.map((x, i) => s.actual!.y[i] != null && <circle key={i} cx={X(x)} cy={Y(s.actual!.y[i] as number)} r="3.5" fill="var(--chart-history)" stroke="var(--background)" strokeWidth="1.5" />)}
        {[0, 0.5, 1].map((f, i) => { const d = new Date(model.x0 + f * (model.x1 - model.x0)).toISOString().slice(0, 10); return <text key={i} x={m.l + f * (W - m.l - m.r)} y={H - 8} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"} fontSize="10" fill="var(--muted-foreground)">{dayShort(d)}</text>; })}
        {idx && <line x1={X(idx)} x2={X(idx)} y1={m.t} y2={H - m.b} stroke="var(--foreground)" strokeWidth="1" opacity="0.4" />}
      </svg>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <Key c="var(--chart-history)" t="Actual demand" />{s.forecast && <Key c="var(--chart-forecast)" t="TimesFM median" bold />}{s.forecast && <Key c="var(--chart-band)" t="10–90% band" box />}
        {(s.baselines?.length ?? 0) > 0 && <Key c="var(--chart-baseline)" t="Baseline" dash />}{s.truth && <Key c="var(--chart-truth)" t="True underlying demand" dash />}
        {(s.events?.length ?? 0) > 0 && <Key c="var(--chart-truth)" t="Known event" box faint />}
        {s.unit && <span className="ml-auto">{s.unit}</span>}
      </div>
      {idx && (
        <div className="mt-2 rounded-md border bg-card px-3 py-2 text-xs tabular" aria-live="polite">
          <span className="font-semibold">{dayShort(idx)}</span>
          {at(s.history.x, s.history.y, idx) != null && <span className="ml-3">Actual {fmt(at(s.history.x, s.history.y, idx))}</span>}
          {s.forecast && at(s.forecast.x, s.forecast.q50, idx) != null && <span className="ml-3 text-[var(--chart-forecast)]">Median {fmt(at(s.forecast.x, s.forecast.q50, idx))} (range {fmt(at(s.forecast.x, s.forecast.q10, idx))}–{fmt(at(s.forecast.x, s.forecast.q90, idx))})</span>}
          {s.actual && at(s.actual.x, s.actual.y, idx) != null && <span className="ml-3">Actually {fmt(at(s.actual.x, s.actual.y, idx))}</span>}
          {s.truth && at(s.truth.x, s.truth.y, idx) != null && <span className="ml-3 text-[var(--chart-truth)]">True {fmt(at(s.truth.x, s.truth.y, idx))}</span>}
        </div>
      )}
    </figure>
  );
}
const Key = ({ c, t, bold, dash, box, faint }: { c: string; t: string; bold?: boolean; dash?: boolean; box?: boolean; faint?: boolean }) => (
  <span className="inline-flex items-center gap-1.5">
    {box ? <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: c, opacity: faint ? 0.4 : 1 }} /> : <span className="inline-block w-5" style={{ borderTop: `${bold ? 3 : 2}px ${dash ? "dashed" : "solid"} ${c}` }} />}
    {t}
  </span>
);
