import { useEffect, useState } from "react";
import { api, post, useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, InlineAlert, Input, Skeleton } from "@/components/ui";
import { pct } from "@/lib/utils";

/** Tenant set-up: the three business facts that drive alerts, the rollout phase, the readiness check and the drift check. */
export function Rollout() {
  const { dataset, meta } = useApp();
  const { data: t, reload } = useApi<any>(`/tenant?dataset=${dataset}`);
  const [cost, setCost] = useState(""), [eff, setEff] = useState(""), [budget, setBudget] = useState(""), [goLive, setGoLive] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ready, setReady] = useState<Record<string, any>>({});
  const [drift, setDrift] = useState<any>(null);
  const [frontier, setFrontier] = useState<any>(null);
  const [fac, setFac] = useState("");
  useEffect(() => {
    if (!t) return;
    setCost(String(t.economics.review_cost)); setEff(String(t.economics.effectiveness)); setBudget(String(t.economics.budget_per_owner)); setFac(String(t.economics.false_alarm_cost)); setGoLive(t.go_live ?? "");
  }, [t]);
  const run = async (key: string, fn: () => Promise<void>) => {
    setBusy(key); setErr(null);
    try { await fn(); } catch (e: any) { setErr(String(e.message ?? e)); } finally { setBusy(null); }
  };
  const save = () => run("save", async () => {
    await post(`/tenant?dataset=${dataset}`, { review_cost: Number(cost), effectiveness: Number(eff), budget_per_owner: Number(budget), false_alarm_cost: Number(fac), go_live: goLive });
    reload();
  });
  const floor = Number(eff) > 0 ? Number(cost) / Number(eff) : null;
  const agents = Object.entries(meta?.agents ?? {});
  const tone = (v: string) => (v === "ready" ? "success" : v === "shadow_first" ? "warning" : "error") as "success" | "warning" | "error";
  const label = (v: string) => (v === "ready" ? "Ready to switch on" : v === "shadow_first" ? "Start in shadow" : "Not ready");
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Client set-up and rollout</CardTitle>
        <CardDescription>Three business facts decide which alerts people see. Nothing else is hand-set. Before an agent goes live it is checked on this client's own history, then runs hidden for the shadow period.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {!t && <Skeleton className="h-24" />}
        {t && (
          <>
            <div className="grid gap-4 md:grid-cols-5">
              <label className="text-sm">Cost to review one alert ($)<Input inputMode="decimal" value={cost} onChange={(e) => setCost(e.target.value)} /></label>
              <label className="text-sm">Share of the loss a reviewed alert saves<Input inputMode="decimal" value={eff} onChange={(e) => setEff(e.target.value)} /></label>
              <label className="text-sm">Cost of acting on a false alarm ($)<Input inputMode="decimal" value={fac} onChange={(e) => setFac(e.target.value)} /></label>
              <label className="text-sm">Alerts per owner each morning<Input inputMode="numeric" value={budget} onChange={(e) => setBudget(e.target.value)} /></label>
              <label className="text-sm">Go-live date (blank = demo)<Input type="date" value={goLive} onChange={(e) => setGoLive(e.target.value)} /></label>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <Button onClick={save} loading={busy === "save"}>Save</Button>
              {floor !== null && <span>An alert must be worth at least about <b>${floor.toFixed(0)}</b> in expected loss to be shown, and more when it is less likely to happen. The rest wait in the digest.</span>}
              <Badge variant={t.phase.phase === "live" ? "success" : t.phase.phase === "demo" ? "neutral" : "warning"}>{t.phase.label}</Badge>
            </div>
            {err && <InlineAlert title="That did not work">{err}</InlineAlert>}

            <div>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold">What each alert budget buys</h3>
                <Button variant="outline" size="sm" loading={busy === "frontier"} onClick={() => run("frontier", async () => setFrontier(await api(`/policy/frontier?dataset=${dataset}`)))}>Show the trade-off</Button>
              </div>
              {!frontier && <p className="text-xs text-muted-foreground">More alerts catch more problems but cost attention. This replays the saved candidates under your settings, so it takes a few seconds.</p>}
              {frontier && (
                <>
                  <div className="overflow-x-auto">
                  <table className="w-full min-w-[640px] text-sm">
                    <thead className="text-xs text-muted-foreground"><tr><th className="py-1 text-left font-medium">Per owner a day</th><th className="text-right font-medium">Shown a day</th><th className="text-right font-medium">Stockouts flagged</th><th className="text-right font-medium">Stockout dollars flagged</th><th className="text-right font-medium">Stockout alerts useful</th><th className="text-right font-medium">Expiry dollars flagged</th></tr></thead>
                    <tbody>{frontier.rows.map((r: any) => (
                      <tr key={r.budget_per_owner} className={`border-t ${r.budget_per_owner === t.economics.budget_per_owner ? "bg-accent/40" : ""}`}>
                        <td className="py-1.5">{r.budget_per_owner}</td><td className="tabular text-right">{r.shown_per_day}</td>
                        <td className="tabular text-right">{pct(r.agents.shortfall?.recall, 0)}</td><td className="tabular text-right">{pct(r.agents.shortfall?.dollar_recall, 0)}</td>
                        <td className="tabular text-right">{pct(r.agents.shortfall?.precision, 0)}</td><td className="tabular text-right">{pct(r.agents.freshness?.dollar_recall, 0)}</td>
                      </tr>))}</tbody>
                  </table>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{frontier.caveat} Synthetic twin, so this shows the method, not a client result.</p>
                </>
              )}
            </div>

            <div>
              <h3 className="mb-2 text-sm font-semibold">Readiness check on this client's history</h3>
              <ul className="divide-y text-sm">
                {agents.map(([k, a]: any) => {
                  const r = ready[k];
                  return (
                    <li key={k} className="flex flex-wrap items-start justify-between gap-3 py-2">
                      <div className="min-w-0 flex-1">
                        <span className="font-medium">{a.title}</span>
                        {r && (
                          <div className="mt-1 text-xs text-muted-foreground">
                            {r.alerts_shown_per_day != null && <>About {r.alerts_shown_per_day} alerts shown a day. </>}
                            {r.precision != null && <>{pct(r.precision, 0)} useful. </>}
                            {r.recall != null && <>Caught {r.caught} of {r.events} recorded problems. </>}
                            {r.reasons.concat(r.notes).map((x: string) => <span key={x} className="block">{x}</span>)}
                            <span className="block italic">{r.caveat}</span>
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        {r && <Badge variant={tone(r.verdict)}>{label(r.verdict)}</Badge>}
                        <Button variant="outline" size="sm" loading={busy === k}
                          onClick={() => run(k, async () => { setReady({ ...ready, [k]: await api(`/readiness?agent=${k}&dataset=${dataset}&weeks=13`) }); })}>Check</Button>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <p className="mt-1 text-xs text-muted-foreground">Replays the last 13 weeks under the settings above. It can take a minute.</p>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-semibold">Forecast health by volume class</h3>
                <Button variant="outline" size="sm" loading={busy === "drift"} onClick={() => run("drift", async () => setDrift(await api(`/drift?dataset=${dataset}`)))}>Run drift check</Button>
              </div>
              {!drift && <EmptyState title="Not run yet">Compares the forecast range with what happened over the last eight weeks.</EmptyState>}
              {drift && (
                <>
                  <table className="w-full text-sm">
                    <thead className="text-xs text-muted-foreground"><tr><th className="py-1 text-left font-medium">Sellers</th><th className="text-right font-medium">Actuals inside the range (aim 80%)</th><th className="text-right font-medium">Actuals below the median (aim 50%)</th><th className="text-right font-medium">Status</th></tr></thead>
                    <tbody>{Object.entries(drift.by_volume_class).map(([k, v]: any) => (
                      <tr key={k} className="border-t"><td className="py-1.5 capitalize">{k}</td><td className="tabular text-right">{pct(v.coverage_10_90, 0)}</td><td className="tabular text-right">{pct(v.share_below_median, 0)}</td>
                        <td className="text-right"><Badge variant={v.status === "ok" ? "success" : "warning"}>{v.status === "ok" ? "Healthy" : "Drifting"}</Badge></td></tr>))}</tbody>
                  </table>
                  <p className="mt-1 text-xs text-muted-foreground">{drift.action}</p>
                </>
              )}
            </div>

            <div>
              <h3 className="mb-1 text-sm font-semibold">What the system has learned</h3>
              {t.learning_log.length === 0
                ? <p className="text-xs text-muted-foreground">Nothing yet. As operators approve or dismiss alerts, the system learns which kinds (for example a product category) it can trust. Priorities move by at most 10% a step, never outside 0.5 to 2 times, and every change is listed here. Nothing disappears: a distrusted kind drops down the list into the digest.</p>
                : <ul className="text-xs">{t.learning_log.map((l: any) => <li key={l.at + (l.key ?? l.agent)}>{l.at.slice(0, 10)}: {(l.key ?? l.agent).replace("|", " · ")} priority {l.old} to {l.new} ({pct(l.basis.precision, 0)} of {l.basis.useful + l.basis.not_useful} reviewed were useful)</li>)}</ul>}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
