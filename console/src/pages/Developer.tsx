import { useState } from "react";
import { useApi, post } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState, InlineAlert, PageHeader, Select, Stat, Skeleton } from "@/components/ui";
import { HBars } from "@/components/charts/Bars";
import { pct } from "@/lib/utils";
import { Rollout } from "@/components/Rollout";

/** Developer view: model status, licence, runs, backtests, jobs, cost ledger, and the API surface. */
export function Developer() {
  const { meta, dataset } = useApp();
  const { data: bts, reload: reloadBts } = useApi<any[]>(`/backtests?dataset=${dataset}`);
  const { data: jobs, reload: reloadJobs } = useApi<any[]>(`/jobs`);
  const [freq, setFreq] = useState("W");
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setRunning(true); setErr(null);
    try { await post("/backtests/run", { dataset, freq, horizon: freq === "W" ? 4 : 14, n_origins: freq === "W" ? 26 : 60 }); reloadJobs(); setTimeout(() => { reloadBts(); reloadJobs(); }, 3000); }
    catch (e: any) { setErr(String(e.message ?? e)); } finally { setRunning(false); }
  };
  const latest = bts?.[0];
  return (
    <>
      <PageHeader eyebrow="Developer" title="Models, runs and the API"
        description="Everything the other pages show is produced here. Backtests and replays run as background jobs; results are stored, so reloading never loses work." />
      <section className="mb-6 grid gap-4 md:grid-cols-4" aria-label="Model status">
        <Card><CardContent className="p-5"><Stat label="Model" value="TimesFM 3.0" sub={meta?.model.backend} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat label="Weights" value={meta?.model.loaded ? "Loaded" : "On first use"} sub={meta?.model.checkpoint} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat label="Narration spend" value={`$${(meta?.bedrock.spent_usd ?? 0).toFixed(3)}`} sub={`Hard cap $${meta?.bedrock.cap_usd ?? 30} (${meta?.bedrock.model})`} /></CardContent></Card>
        <Card><CardContent className="p-5"><Stat label="Licence" value={<Badge variant="warning">Unresolved</Badge>} sub="Non-commercial weights: no client production" /></CardContent></Card>
      </section>

      <Rollout />

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Walk-forward backtest</CardTitle>
            <CardDescription>Forecast each past origin using only data before it, then score against what happened. Compares TimesFM 3 with the baselines and with perfect knowledge.</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Select aria-label="Frequency" value={freq} onChange={(e) => setFreq(e.target.value)} className="h-9">
                <option value="W">Weekly, 4-week horizon, 26 origins</option><option value="D">Daily, 14-day horizon, 60 origins</option>
              </Select>
              <Button onClick={run} loading={running}>Run backtest</Button>
            </div>
            {err && <InlineAlert title="Backtest did not start">{err}</InlineAlert>}
            {latest ? (
              <>
                <p className="text-xs text-muted-foreground">Last run {new Date(latest.created_at).toLocaleString("en-AU")} · {latest.results.origins} origins · {latest.results.seconds}s</p>
                <HBars better="lower" format={(v) => pct(v, 1)} highlight="TimesFM 3" rows={Object.entries(latest.results.models).map(([k, v]: any) => ({ label: k, value: v.head.wape }))} />
                <p className="text-xs text-muted-foreground">WAPE on the top 80% of products, lower is better. Verdict vs best basic rule: {pct(latest.results.verdict?.improvement_vs_best_baseline, 1)}.</p>
              </>
            ) : <EmptyState title="No backtest yet">Run one above.</EmptyState>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Background jobs</CardTitle><CardDescription>Latest first. Jobs survive page changes; results are stored.</CardDescription></CardHeader>
          <CardContent>
            {!jobs && <Skeleton className="h-32" />}
            {jobs?.length === 0 && <EmptyState title="No jobs yet" />}
            <ul className="divide-y text-sm">
              {jobs?.map((j) => (
                <li key={j.id} className="flex items-center justify-between gap-3 py-2">
                  <span><span className="font-medium">{j.kind}</span> <span className="text-xs text-muted-foreground">{j.id}</span></span>
                  <span className="flex items-center gap-2 text-xs">
                    <Badge variant={j.status === "done" ? "success" : j.status === "error" ? "error" : "info"}>{j.status}</Badge>
                    {j.status === "running" && <span className="tabular">{Math.round((j.progress ?? 0) * 100)}%</span>}
                    {j.error && <span className="text-red-600">{j.error}</span>}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>API</CardTitle><CardDescription>The console is a thin client. Every number can be pulled programmatically.</CardDescription></CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <thead className="text-xs text-muted-foreground"><tr><th className="py-1.5 text-left font-medium">Method and path</th><th className="text-left font-medium">What it does</th></tr></thead>
              <tbody className="font-mono text-xs">
                {API.map(([m, p, d]) => <tr key={p + m} className="border-t"><td className="py-1.5"><span className="mr-2 rounded bg-accent px-1.5 py-0.5 font-semibold">{m}</span>{p}</td><td className="font-sans text-sm text-muted-foreground">{d}</td></tr>)}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </>
  );
}

const API: [string, string, string][] = [
  ["GET", "/api/meta", "Model, datasets, agents, spend, ship bar"],
  ["GET", "/api/datasets/{ds}/summary", "Revenue, categories, operating rates"],
  ["GET", "/api/agents?dataset=", "Agent configuration, stats, ladder status"],
  ["PUT", "/api/agents/{agent}?dataset=", "Enable, owner, quiet hours, cooldown, autonomy, thresholds"],
  ["GET", "/api/proposals?dataset=&agent=&status=", "Proposals with evidence, ranked by money at stake"],
  ["POST", "/api/proposals/{id}/decision", "Approve, adjust or dismiss (dismissal can create a lesson)"],
  ["POST", "/api/proposals/{id}/narrate", "Templated explanation, plus optional Haiku wording checked against evidence"],
  ["GET", "/api/lessons?dataset=", "Lessons the business has taught the agents"],
  ["POST", "/api/lessons/{id}/undo", "Remove a lesson at once"],
  ["GET", "/api/forecast?dataset=&product=&as_of=", "TimesFM forecast with bands, baselines, actuals and truth"],
  ["POST", "/api/backtests/run", "Start a walk-forward backtest job"],
  ["POST", "/api/replay/run", "Start a full replay job with scoring"],
  ["POST", "/api/time/step", "Advance the time machine up to 14 days"],
  ["GET", "/api/briefing?dataset=&owner=", "The morning briefing for a date and owner"],
  ["GET", "/api/scorecard?dataset=", "Latest replay scorecard"],
  ["GET", "/api/tenant?dataset=", "Client economics (review cost, effectiveness, alerts per owner), go-live date, rollout phase, learning log"],
  ["POST", "/api/tenant?dataset=", "Change the economics, go-live date or shadow length"],
  ["GET", "/api/policy/frontier?dataset=", "What each alerts-per-owner budget buys under the client's economics"],
  ["GET", "/api/readiness?agent=&dataset=", "Replay one agent on this client's recent history and say if it is safe to switch on"],
  ["GET", "/api/drift?dataset=", "Forecast calibration by volume class over the latest eight weeks"],
  ["POST", "/api/tenant/learn?dataset=", "Turn approvals and dismissals into bounded priority weights"],
  ["GET", "/api/jobs/{id}", "Progress and result of a background job"],
];
