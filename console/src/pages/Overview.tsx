import { Link } from "react-router-dom";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Card, CardContent, EmptyState, InlineAlert, PageHeader, Skeleton, Stat } from "@/components/ui";
import { money, pct, num } from "@/lib/utils";
import { HBars } from "@/components/charts/Bars";

interface Scorecard { agents: Record<string, any>; window: string[]; dataset: string; available?: boolean; proposals_total?: number; dollars_truth_caught?: number }

/** Executive view: is it working, what does it protect, what is the honest caveat. */
export function Overview() {
  const { dataset, meta } = useApp();
  const { data: sc, error, loading, reload } = useApi<Scorecard>(`/scorecard?dataset=${dataset}`);
  const { data: summary } = useApi<any>(`/datasets/${dataset}/summary`);
  if (error) return <InlineAlert title="The scorecard could not load" onRetry={reload}>{error}</InlineAlert>;
  const isTwin = dataset === "twin";
  const agents = sc?.agents ?? {};
  const caught = Object.values(agents).reduce((a: number, v: any) => a + (v.dollars_truth_caught ?? 0), 0) as number;
  const total = Object.values(agents).reduce((a: number, v: any) => a + (v.dollars_truth_total ?? 0), 0) as number;
  const recallAll = total ? caught / total : null;
  const precision = (() => { const ps = Object.values(agents).filter((v: any) => v.precision != null) as any[]; const n = ps.reduce((a, v) => a + v.proposals, 0); return n ? ps.reduce((a, v) => a + v.useful, 0) / n : null; })();
  const days = sc?.window?.length === 2 ? Math.max(1, Math.round((+new Date(sc.window[1]) - +new Date(sc.window[0])) / 864e5) + 1) : null;
  const perDay = days && sc?.proposals_total != null ? Math.round(sc.proposals_total / days) : null;
  return (
    <>
      <PageHeader eyebrow="Executive view" title="Four agents that watch the business and ask before they act"
        description="They read sales, stock, lots, purchases and prices every day, use TimesFM 3 for the forecast, and only put a proposal in front of a person with the evidence. Nothing is sent or changed without an OK." />

      <section aria-label="Headline numbers" className="mb-6 grid gap-4 md:grid-cols-4">
        <Card><CardContent className="p-5"><Stat big label="Known problems flagged in time" value={pct(recallAll)} sub={isTwin ? "Of the problems planted in the twin, weighted by dollars. Flagged is not the same as fixed." : "Needs labelled history to measure"} tone={recallAll != null && recallAll >= 0.6 ? "good" : "default"} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Alerts that were useful" value={pct(precision)} sub="Share of proposals that matched a real event in the test" tone={precision != null && precision >= 0.6 ? "good" : "warn"} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Dollars flagged in time" value={money(caught, true)} sub={`Of ${money(total, true)} at stake in planted problems. Nobody acted on them in the test, so nothing is saved yet.`} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat big label="Proposals raised" value={num(sc?.proposals_total)} sub={`${perDay != null ? `About ${perDay} a day across all agents. ` : ""}${sc?.window?.join(" to ") ?? ""}`} /></CardContent></Card>
      </section>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardContent className="space-y-4 p-5">
            <h2 className="text-base font-semibold">How often each agent flags a planted problem in time</h2>
            {loading && !sc && <Skeleton className="h-40" />}
            {Object.keys(agents).length > 0 && (
              <HBars better="higher" format={(v) => pct(v)} highlight="" rows={Object.entries(agents).map(([k, v]: any) => ({ label: meta?.agents[k]?.title ?? k, value: v.recall ?? 0 }))} />
            )}
            <p className="text-xs text-muted-foreground">Each bar is one agent. The Proof page breaks this down by problem type, because an average hides the weak spots.</p>
            <div className="flex flex-wrap gap-2 pt-2">
              <Link to="/briefing" className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground">See a morning briefing</Link>
              <Link to="/scorecard" className="rounded-md border px-3 py-1.5 text-sm hover:bg-menu-hover">Read the proof</Link>
              <Link to="/time" className="rounded-md border px-3 py-1.5 text-sm hover:bg-menu-hover">Run the time machine</Link>
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardContent className="space-y-3 p-5">
            <h2 className="text-base font-semibold">What this rests on</h2>
            <ul className="space-y-2 text-sm">
              <li><span className="font-medium">Model:</span> TimesFM 3.0 with quantile bands, run on Apple silicon (MLX).</li>
              <li><span className="font-medium">Test data:</span> {isTwin ? "a synthetic twin of a meat wholesaler, 24 months, with known shocks labelled. Its results show whether the method catches real problems." : "real staging orders, about eight weeks. Good for plumbing, too short to prove seasonality."}</li>
              <li><span className="font-medium">Against the basics:</span> forecasts are scored against simple moving averages on the Proof page, so you can see where TimesFM does and does not help.</li>
              <li><span className="font-medium">Licence:</span> TimesFM 3.0 weights are non-commercial. No client production use until that is resolved.</li>
            </ul>
            {summary?.operations && (
              <p className="rounded-md bg-menu-hover p-3 text-xs">Current buying policy in the twin: waste {pct(summary.operations.waste_pct, 1)} of sales value, fill rate {pct(summary.operations.fill_rate, 1)}. These are the numbers agents must beat.</p>
            )}
            {!meta?.datasets?.[dataset] && <EmptyState title="No data loaded for this dataset" />}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
