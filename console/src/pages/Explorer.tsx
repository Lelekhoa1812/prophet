import { useMemo, useState } from "react";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { BandChart } from "@/components/charts/BandChart";
import { HBars } from "@/components/charts/Bars";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState, InlineAlert, Input, PageHeader, Select, Skeleton, Tabs, TabsList, TabsTrigger } from "@/components/ui";
import { money, pct } from "@/lib/utils";

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
  const [asOf, setAsOf] = useState<string>("");
  const defaultAsOf = useMemo(() => { const d = new Date(last); d.setDate(d.getDate() - 56); return d.toISOString().slice(0, 10); }, [last]);
  const at = asOf || defaultAsOf;
  const { data, error, loading, reload } = useApi<any>(product ? `/forecast?dataset=${dataset}&product=${product}&as_of=${at}&freq=${freq}&horizon=${freq === "W" ? 4 : 14}&history=${freq === "W" ? 52 : 120}` : null);
  const label = (pidv: string) => reveal ? (revealed[pidv] ?? pidv) : pidv;
  const actualAfter = data?.actual_future as (number | null)[] | undefined;
  const ev = data?.events ?? [];
  return (
    <>
      <PageHeader eyebrow="Developer and operator" title="Forecast explorer"
        description="Choose a product and a date. The chart uses only data before that date for the forecast, then shows what really happened. The shaded areas are known events on the twin." />
      <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
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
        <div className="space-y-5">
          <div className="flex flex-wrap items-end gap-3">
            <Tabs value={freq} onValueChange={(v) => setFreq(v as "W" | "D")}>
              <TabsList><TabsTrigger value="W">Weekly</TabsTrigger><TabsTrigger value="D">Daily</TabsTrigger></TabsList>
            </Tabs>
            <label className="space-y-1 text-xs text-muted-foreground">Forecast made on
              <Input type="date" className="w-44" min={first} max={last} value={at} onChange={(e) => setAsOf(e.target.value)} /></label>
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
                <div className="grid gap-5 md:grid-cols-2">
                  <div>
                    <p className="mb-2 text-sm font-medium">Next {freq === "W" ? "weeks" : "days"}: TimesFM against baselines</p>
                    <table className="w-full text-sm tabular">
                      <thead className="text-xs text-muted-foreground"><tr><th className="text-left font-medium">Period</th><th className="text-right font-medium">TimesFM median</th><th className="text-right font-medium">10–90%</th><th className="text-right font-medium">Moving avg</th><th className="text-right font-medium">Actual</th></tr></thead>
                      <tbody>
                        {data.forecast.x.map((x: string, i: number) => (
                          <tr key={x} className="border-t"><td className="py-1">{x}</td><td className="text-right font-medium text-[var(--chart-forecast)]">{data.forecast.point[i]}</td>
                            <td className="text-right text-muted-foreground">{data.forecast.q.q10[i]}–{data.forecast.q.q90[i]}</td>
                            <td className="text-right">{data.baselines.ma8?.point[i] ?? "–"}</td>
                            <td className="text-right font-medium">{actualAfter?.[i] ?? "–"}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <p className="mb-2 text-sm font-medium">How the forecast compares</p>
                    <HBars better="lower" format={(v) => v.toFixed(1)} rows={[{ label: "TimesFM 3", value: Math.abs(sumDiff(actualAfter, data.forecast.point)) }, { label: "Moving avg", value: Math.abs(sumDiff(actualAfter, data.baselines.ma8?.point ?? [])) }]} />
                    <p className="mt-2 text-xs text-muted-foreground">Gap in units over the horizon, where actuals exist (lower is better).</p>
                    {ev.length > 0 && <div className="mt-4"><p className="text-xs font-medium">Known events on this product</p>
                      <ul className="mt-1 space-y-1 text-xs">{ev.map((e: any) => <li key={e.event_id}>{e.start_date} · {e.note}</li>)}</ul></div>}
                  </div>
                </div>
                <p className="text-xs text-muted-foreground">Price and stock are not part of the forecast; it uses order history only. Planned promotions are known in advance and can be added as a covariate (see the developer page).</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

function sumDiff(actual: (number | null)[] | undefined, pred: number[]) {
  if (!actual) return 0;
  return actual.reduce<number>((a, v, i) => a + (v == null ? 0 : (v - (pred[i] ?? 0))), 0);
}
