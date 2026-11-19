import { useMemo, useState } from "react";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { BandChart } from "@/components/charts/BandChart";
import { HBars } from "@/components/charts/Bars";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState, InlineAlert, Input, PageHeader, Select, Skeleton, Tabs, TabsList, TabsTrigger } from "@/components/ui";
import { money, pct } from "@/lib/utils";

// Presets for the history the model sees (periods = weeks or days). "All" caps at the context the model uses.
const HIST_PRESETS: Record<"W" | "D", { v: number; l: string }[]> = {
  W: [{ v: 13, l: "3 months" }, { v: 26, l: "6 months" }, { v: 52, l: "1 year" }, { v: 104, l: "2 years" }, { v: 208, l: "4 years" }, { v: 400, l: "All" }],
  D: [{ v: 56, l: "8 weeks" }, { v: 91, l: "13 weeks" }, { v: 182, l: "6 months" }, { v: 364, l: "1 year" }, { v: 2000, l: "All" }],
};
const HORIZON_MAX = { W: 12, D: 28 } as const;

/** Forecast explorer: pick a product and a date, see the history, the TimesFM band, the baselines and what happened next. */
export function Explorer() {
  const { dataset, reveal, revealed } = useApp();
  const [q, setQ] = useState("");
  const { data: prods } = useApi<any[]>(`/products?dataset=${dataset}&limit=40${q ? `&q=${encodeURIComponent(q)}` : ""}`);
  const [pid, setPid] = useState<string | null>(null);
  const product = pid ?? prods?.[0]?.product_id ?? null;
  const { data: card } = useApi<any>(`/datasets/${dataset}/summary`);
  const first = card?.card?.first ?? "2025-01-01", last = card?.card?.last ?? "2026-10-01";
  const [freq, setFreq] = useState<"W" | "D">("W");
  const [histW, setHistW] = useState(400), [histD, setHistD] = useState(2000);
  const [horW, setHorW] = useState(4), [horD, setHorD] = useState(14);
  const hist = freq === "W" ? histW : histD;
  const hor = freq === "W" ? horW : horD;
  const setHist = freq === "W" ? setHistW : setHistD;
  const setHor = freq === "W" ? setHorW : setHorD;
  const { data: bts } = useApi<any[]>(`/backtests?dataset=${dataset}`);
  const testResults = bts?.[0]?.results;
  const [asOf, setAsOf] = useState<string>("");
  const defaultAsOf = useMemo(() => { const d = new Date(last); d.setDate(d.getDate() - 56); return d.toISOString().slice(0, 10); }, [last]);
  const at = asOf || defaultAsOf;
  const { data, error, loading, reload } = useApi<any>(product ? `/forecast?dataset=${dataset}&product=${product}&as_of=${at}&freq=${freq}&horizon=${hor}&history=${hist}` : null);
  const label = (pidv: string) => reveal ? (revealed[pidv] ?? pidv) : pidv;
  const actualAfter = data?.actual_future as (number | null)[] | undefined;
  const ev = data?.events ?? [];
  return (
    <>
      <PageHeader eyebrow="Developer and operator" title="Forecast explorer"
        description="Choose a product and a date. The chart uses only data before that date for the forecast, then shows what really happened. The shaded areas are known events on the twin." />
      <div className="grid grid-cols-[minmax(0,1fr)] gap-5 lg:grid-cols-[18rem_minmax(0,1fr)]">
        <Card className="h-fit">
          <CardHeader><CardTitle className="text-sm">Product</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Input placeholder="Search name, code or category" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search products" />
            <ul className="max-h-[26rem] space-y-0.5 overflow-y-auto pr-1 text-sm" role="listbox" aria-label="Products">
              {(prods ?? []).map((p) => (
                <li key={p.product_id}>
                  <button role="option" aria-selected={product === p.product_id} onClick={() => setPid(p.product_id)}
                    className={`w-full cursor-pointer rounded px-2 py-1.5 text-left ${product === p.product_id ? "bg-accent font-semibold" : "hover:bg-menu-hover"}`}>
                    <span className="block truncate">{label(p.product_id)} · {p.category}</span>
                    <span className="block text-xs text-muted-foreground tabular">{money(p.revenue, true)} revenue · shelf {p.shelf_life_days || "–"} d</span>
                  </button>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
        <div className="min-w-0 space-y-5">
          <div className="flex flex-wrap items-end gap-3">
            <Tabs value={freq} onValueChange={(v) => setFreq(v as "W" | "D")}>
              <TabsList><TabsTrigger value="W">Weekly</TabsTrigger><TabsTrigger value="D">Daily</TabsTrigger></TabsList>
            </Tabs>
            <label className="space-y-1 text-xs text-muted-foreground">Forecast made on
              <Input type="date" className="w-44" min={first} max={last} value={at} onChange={(e) => setAsOf(e.target.value)} /></label>
            <div className="space-y-1" role="group" aria-label="History the model sees">
              <p className="text-xs text-muted-foreground">History the model sees</p>
              <div className="flex flex-wrap gap-1">
                {HIST_PRESETS[freq].map((h) => (
                  <button key={h.v} type="button" aria-pressed={hist === h.v} onClick={() => setHist(h.v)}
                    className={`h-8 cursor-pointer rounded-md border px-2.5 text-xs ${hist === h.v ? "border-primary bg-accent font-semibold" : "hover:bg-menu-hover"}`}>{h.l}</button>
                ))}
              </div>
            </div>
            <label className="w-56 space-y-1 text-xs text-muted-foreground">Look ahead: <span className="font-medium text-foreground tabular">{hor} {freq === "W" ? (hor === 1 ? "week" : "weeks") : (hor === 1 ? "day" : "days")}</span>
              <input type="range" min={1} max={HORIZON_MAX[freq]} step={1} value={hor} onChange={(e) => setHor(Number(e.target.value))} className="w-full accent-[var(--primary)]" aria-label="Forecast horizon" />
            </label>
            <p className="pb-2 text-xs text-muted-foreground">Data available {first} to {last}</p>
          </div>
          {error && <InlineAlert title="The forecast could not run" onRetry={reload}>{error}. The model loads on first use, which can take about 80 seconds.</InlineAlert>}
          {loading && !data && <Skeleton className="h-96" />}
          {!product && <EmptyState title="Pick a product">Search or select one on the left.</EmptyState>}
          {data && (
            <Card>
              <CardHeader>
                <CardTitle>{label(data.product)} · {data.name}</CardTitle>
                <CardDescription>{freq === "W" ? "Weekly units ordered" : "Daily units ordered"}, forecast from {data.as_of}, horizon {data.forecast.x.length} {freq === "W" ? "weeks" : "days"}.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-5">
                <BandChart height={300} label="Forecast with bands" s={{
                  history: data.history, forecast: { x: data.forecast.x, q10: data.forecast.q.q10, q50: data.forecast.q.q50, q90: data.forecast.q.q90 },
                  baselines: [{ name: "ma8", x: data.forecast.x, y: data.baselines.ma8?.point ?? [] }],
                  actual: actualAfter ? { x: data.forecast.x, y: actualAfter } : undefined,
                  truth: data.truth_expected ?? undefined,
                  events: ev.map((e: any) => ({ start: e.start_date, end: e.end_date, label: `${e.kind}: ${e.note}`, kind: e.kind })),
                  unit: freq === "W" ? "units / week" : "units / day",
                }} />
                <div className="grid grid-cols-[minmax(0,1fr)] gap-5 2xl:grid-cols-2">
                  <div>
                    <p className="mb-2 text-sm font-medium">Next {freq === "W" ? "weeks" : "days"}: TimesFM against baselines</p>
                    <div className="overflow-x-auto"><table className="w-full min-w-[30rem] text-sm tabular">
                      <thead className="text-xs text-muted-foreground"><tr><th className="text-left font-medium">Period</th><th className="text-right font-medium">TimesFM median</th><th className="text-right font-medium">10–90%</th><th className="text-right font-medium">Moving avg</th><th className="text-right font-medium">Actual</th></tr></thead>
                      <tbody>
                        {data.forecast.x.map((x: string, i: number) => (
                          <tr key={x} className="border-t"><td className="whitespace-nowrap py-1 pr-2">{x}</td><td className="whitespace-nowrap text-right font-medium text-[var(--chart-forecast)]">{Math.round(data.forecast.point[i]).toLocaleString("en-AU")}</td>
                            <td className="whitespace-nowrap text-right text-muted-foreground">{Math.round(data.forecast.q.q10[i]).toLocaleString("en-AU")} to {Math.round(data.forecast.q.q90[i]).toLocaleString("en-AU")}</td>
                            <td className="whitespace-nowrap text-right">{data.baselines.ma8 ? Math.round(data.baselines.ma8.point[i]).toLocaleString("en-AU") : "–"}</td>
                            <td className="whitespace-nowrap text-right font-medium">{actualAfter?.[i] != null ? Math.round(actualAfter[i]).toLocaleString("en-AU") : "–"}</td></tr>
                        ))}
                      </tbody>
                    </table></div>
                  </div>
                  <div>
                    <p className="mb-2 text-sm font-medium">How the forecast compares</p>
                    {/* Motivation vs Logic
                        Motivation: a signed sum lets over- and under-forecasts cancel, so a bad forecast can look good.
                        Logic: compare the sum of absolute weekly errors, then express it as a share of actual units. */}
                    <HBars better="lower" format={(v) => `${v.toLocaleString("en-AU", { maximumFractionDigits: 0 })} units`}
                      rows={[{ label: "TimesFM 3", value: absErr(actualAfter, data.forecast.point) }, { label: "Moving avg (8 wk)", value: absErr(actualAfter, data.baselines.ma8?.point ?? []) }]} />
                    <p className="mt-2 text-xs text-muted-foreground">Sum of absolute weekly errors over the horizon, where actuals exist. Lower is better. {(() => { const a = actualAfter?.filter((v) => v != null) as number[] | undefined; const tot = a?.reduce((x, y) => x + y, 0) ?? 0; return tot ? `TimesFM is ${pct(absErr(actualAfter, data.forecast.point) / tot)} of actual units off; moving average ${pct(absErr(actualAfter, data.baselines.ma8?.point ?? []) / tot)}.` : "No actuals yet for this window."; })()}</p>
                    {ev.length > 0 && <div className="mt-4 rounded-md border border-dashed p-2"><p className="text-xs font-medium">Labelled events in the twin, for checking</p>
                      <p className="text-[11px] text-muted-foreground">These are the twin's planted shocks. The forecast cannot see them in advance, so they show where it should react, not what it knew.</p>
                      <ul className="mt-1 space-y-1 text-xs">{ev.map((e: any) => <li key={e.event_id}><span className="tabular">{String(e.start_date).slice(0, 10)}</span> · {e.note}</li>)}</ul></div>}
                  </div>
                </div>
                {data.partial_week && (
                  <p className="rounded-md border p-3 text-xs text-muted-foreground">
                    The week of {data.partial_week.week_start} is only partly known ({data.partial_week.known_days} days, {Math.round(data.partial_week.known_units).toLocaleString("en-AU")} units so far). The forecast starts the following Monday, so the history stops at the last full week.
                  </p>
                )}
                {data.model_saw && <ModelSaw saw={data.model_saw} freq={freq} hor={hor} results={testResults} />}
                <p className="text-xs text-muted-foreground">Price and stock are not part of the forecast; it uses order history only. Planned promotions are known in advance and can be added as a covariate (see the developer page).</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

function absErr(actual: (number | null)[] | undefined, pred: number[]) {
  if (!actual) return 0;
  return actual.reduce<number>((a, v, i) => a + (v == null ? 0 : Math.abs(v - (pred[i] ?? 0))), 0);
}

/** Explains what the forecast was built from, and what it cannot see. Plain language first, numbers second. */
function ModelSaw({ saw, freq, hor, results }: { saw: any; freq: "W" | "D"; hor: number; results: any }) {
  const unit = freq === "W" ? "weeks" : "days";
  const tf = results?.models?.timesfm;
  const coverage = tf?.head?.coverage;
  const byH: number[] | undefined = tf?.by_horizon;
  return (
    <div className="grid gap-5 rounded-md border bg-menu-hover p-4 md:grid-cols-2">
      <div className="space-y-2 text-sm">
        <p className="font-semibold">What the model saw</p>
        <p>{saw.periods} {unit}, {saw.from} to {saw.to}{saw.seasonal_years ? ` (${saw.seasonal_years} of a year)` : ""}.</p>
        {saw.one_year_sensitivity != null && (
          <p className="text-xs">If the window were cut to the last year, the next weeks would move by about <span className="font-semibold tabular">{pct(saw.one_year_sensitivity)}</span> on average. {saw.one_year_sensitivity > 0.05 ? "That is large enough to question the forecast." : "The answer is stable to the window."}</p>
        )}
        {saw.derived && <p className="text-xs text-muted-foreground">Daily values are the weekly forecast split by this product's weekday pattern, so they agree with the briefing.</p>}
        {freq === "W" && !saw.full_year && (
          <p className="rounded bg-amber-50 p-2 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-100">
            Less than a full year. The model cannot learn yearly peaks such as Christmas or Easter, so expect a flatter forecast. Widen the history to see whether the peak is in the data.
          </p>
        )}
        <p className="text-xs text-muted-foreground">Not used by this forecast: promotions, school and public holidays, price and stock. Those are tested separately on the Proof page.</p>
      </div>
      <div className="space-y-2 text-sm">
        <p className="font-semibold">How to read the shaded band</p>
        <p className="text-xs">The band is the 10 to 90 percent range. In the test, {coverage != null ? <>about <span className="font-semibold tabular">{pct(coverage)}</span></> : "most"} of real outcomes landed inside it, which is close to the 80 percent it aims for.</p>
        {freq === "W" && byH && byH.length > 0 && (
          <div>
            <p className="text-xs font-medium">Typical error by week ahead (test set, lower is better)</p>
            <ul className="mt-1 space-y-1 text-xs tabular">
              {byH.map((v, i) => (
                <li key={i} className={`flex items-center gap-2 ${i + 1 === hor ? "font-semibold" : ""}`}>
                  <span className="w-16">Week {i + 1}</span>
                  <span className="h-2 rounded bg-[var(--chart-forecast)]" style={{ width: `${Math.min(100, v * 200)}%`, maxWidth: "70%" }} />
                  <span>{pct(v)}</span>
                </li>
              ))}
            </ul>
            <p className="mt-1 text-xs text-muted-foreground">Error grows the further ahead you look. Plan on the near weeks; treat distant weeks as a range.</p>
          </div>
        )}
      </div>
    </div>
  );
}

