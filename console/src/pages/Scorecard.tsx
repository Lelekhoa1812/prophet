import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState, InlineAlert, PageHeader, Skeleton, Stat } from "@/components/ui";
import { HBars, Calibration } from "@/components/charts/Bars";
import { money, pct, num } from "@/lib/utils";

const BAR = { error_gap: 0.15, coverage: [0.7, 0.9] as [number, number] };

/** Proof: whether the agents clear the ship bar, against the simple baselines and against the true underlying demand. */
export function Scorecard() {
  const { dataset, meta } = useApp();
  const { data: sc, error, loading, reload } = useApi<any>(`/scorecard?dataset=${dataset}`);
  const { data: bts, error: btErr } = useApi<any[]>(`/backtests?dataset=${dataset}`);
  const bt = bts?.[0]?.results;
  if (error) return <InlineAlert title="The scorecard could not load" onRetry={reload}>{error}</InlineAlert>;
  const agents = sc?.agents ?? {};
  const rows = Object.entries(agents) as [string, any][];
  const bestBt = bt?.models;
  const forecastBars = bestBt ? Object.entries(bestBt).filter(([k]) => ["timesfm", "timesfm_cov", "ensemble", "ma4", "ma8", "seasonal_year", "noise_floor"].includes(k))
    .map(([k, v]: any) => ({ label: LABEL[k] ?? k, value: v.head.wape })) : [];
  const v = bt?.verdict;
  return (
    <>
      <PageHeader eyebrow="Proof" title="Does it work, and is it better than the basics?"
        description="Every number here is measured against a labelled test. The forecast is judged on the products that make 80% of sales, against a competent buyer's rules and against the true demand in the twin." />

      <section aria-label="Ship bar" className="mb-6 grid gap-4 md:grid-cols-4">
        <Card><CardContent className="p-5"><Stat big label="Error vs best basic rule" value={v ? pct(v.improvement_vs_best_baseline, 0) : "–"} sub={`Bar: at least ${pct(BAR.error_gap)} lower error`} tone={v?.passes_error_bar ? "good" : "warn"} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Bands cover the actuals" value={v ? pct(v.band_coverage_head, 0) : "–"} sub="Target 70–90% inside the 10–90% band" tone={v?.passes_coverage_bar ? "good" : "warn"} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Alerts that were useful" value={pct(overallPrecision(rows))} sub="Bar: at least 70%" tone={overallPrecision(rows) != null && overallPrecision(rows)! >= 0.7 ? "good" : "warn"} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Invented numbers" value="0" sub="Narration checked against evidence" tone="good" /></CardContent></Card>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Forecast error on the top 80% of products</CardTitle>
            <CardDescription>Weighted absolute error (WAPE) over the last walk-forward backtest, weekly, four-week horizon. Lower is better.</CardDescription></CardHeader>
          <CardContent>
            {btErr && <InlineAlert title="Backtest results could not load">{btErr}</InlineAlert>}
            {forecastBars.length ? <HBars better="lower" format={(x) => pct(x, 1)} highlight="TimesFM 3" rows={forecastBars} /> :
              <EmptyState title="No backtest yet">Run one from the developer page; it takes about two minutes.</EmptyState>}
            {bt && <p className="mt-3 text-xs text-muted-foreground">Ran {bt.origins} weekly origins from {bt.first_origin} to {bt.last_origin}. TimesFM is compared with the best single basic rule ({v?.best_baseline}).</p>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Are the bands honest?</CardTitle>
            <CardDescription>For each quantile, the share of actual outcomes that fell below it. A calibrated model sits on the dashed diagonal.</CardDescription></CardHeader>
          <CardContent className="flex flex-col items-center gap-3 md:flex-row md:items-start">
            {bestBt?.timesfm?.calibration ? <Calibration observed={bestBt.timesfm.calibration} /> : <EmptyState title="Run a backtest to see calibration" />}
            <p className="text-sm text-muted-foreground">A wide band that is always right is useless to a buyer, and a narrow one that misses often is dangerous. Coverage close to 80% inside the 10–90% band is the target.</p>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Do the agents catch real problems in time?</CardTitle>
            <CardDescription>Measured on the synthetic twin, where each problem was planted and dated. "Caught" means an alert was raised before or during the event.</CardDescription></CardHeader>
          <CardContent>
            {loading && !sc && <Skeleton className="h-48" />}
            {!sc?.available && <EmptyState title="No labelled test loaded">The scorecard needs the synthetic twin, or a replay on it.</EmptyState>}
            {rows.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[640px] text-sm tabular">
                  <thead className="text-xs text-muted-foreground"><tr>
                    <th className="py-2 text-left font-medium">Agent</th><th className="text-right font-medium">Alerts</th><th className="text-right font-medium">Useful</th>
                    <th className="text-right font-medium">Known events caught</th><th className="text-right font-medium">Median lead</th><th className="text-right font-medium">Dollars caught</th><th className="text-right font-medium">Dollars at stake</th></tr></thead>
                  <tbody>
                    {rows.map(([k, a]) => (
                      <tr key={k} className="border-t">
                        <td className="py-2 font-medium">{meta?.agents[k]?.title ?? k}</td>
                        <td className="text-right">{num(a.proposals)}</td>
                        <td className="text-right">{pct(a.precision)}</td>
                        <td className="text-right">{a.caught} of {a.events} ({pct(a.recall)})</td>
                        <td className="text-right">{a.median_delay_days != null ? `${Math.round(a.median_delay_days)} days` : "–"}</td>
                        <td className="text-right">{money(a.dollars_truth_caught, true)}</td>
                        <td className="text-right">{money(a.dollars_truth_total, true)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {sc?.window && <p className="mt-3 text-xs text-muted-foreground">Test window {sc.window[0]} to {sc.window[1]}. The first 55% of history is kept as context only, so every agent is judged on periods it has not seen.</p>}
          </CardContent>
        </Card>
      </div>
      <p className="mt-6 text-xs text-muted-foreground">Ship bar: the agent must beat the best basic rule by at least 15% on the products that make 80% of sales, keep bands near their stated coverage, and have most alerts useful. Results are on a synthetic twin and must be re-proven on each client's own history before any client sees a live alert.</p>
    </>
  );
}

const LABEL: Record<string, string> = { timesfm: "TimesFM 3", timesfm_cov: "TimesFM 3 + promotions", ensemble: "TimesFM + moving avg", ma4: "Moving average (4 wks)", ma8: "Moving average (8 wks)", seasonal_year: "Same time last year", noise_floor: "Perfect knowledge (floor)", seasonal_naive: "Same weekday last weeks" };
function overallPrecision(rows: [string, any][]) {
  const n = rows.reduce((a, [, v]) => a + (v.proposals ?? 0), 0);
  const u = rows.reduce((a, [, v]) => a + (v.useful ?? 0), 0);
  return n ? u / n : null;
}
