import { useState } from "react";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, InlineAlert, Skeleton } from "@/components/ui";
import { money, pct } from "@/lib/utils";

/** What the alerts of the stored replay could be worth if people act on a share of them. Expected value, never observed savings. */
export function WhatIf() {
  const { dataset } = useApp();
  const [acc, setAcc] = useState(0.5);
  const { data, error } = useApi<any>(`/policy/whatif?dataset=${dataset}&acceptance=${acc}`);
  return (
    <Card>
      <CardHeader>
        <CardTitle>What the alerts could be worth</CardTitle>
        <CardDescription>Expected value on a synthetic replay, under the assumptions you set. It is not money saved, and no client has acted on these alerts.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-muted-foreground">Share of shown alerts a person acts on</span>
          {[0.25, 0.5, 0.75, 1].map((v) => <Button key={v} size="sm" variant={v === acc ? "default" : "outline"} aria-pressed={v === acc} onClick={() => setAcc(v)}>{pct(v, 0)}</Button>)}
        </div>
        {error && <InlineAlert tone="warning" title="No replay to value yet">{String(error)}</InlineAlert>}
        {!data && !error && <Skeleton className="h-24 w-full" />}
        {data && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[34rem] text-sm">
                <thead className="text-xs text-muted-foreground"><tr><th className="py-1 text-left font-medium">Agent</th><th className="text-right font-medium">Alerts shown</th><th className="text-right font-medium">Alert-level claim (counts repeats)</th><th className="text-right font-medium">One claim per problem</th><th className="text-right font-medium">Real problem dollars (all)</th><th className="text-right font-medium">Expected net</th></tr></thead>
                <tbody>{data.agents.map((a: any) => (
                  <tr key={a.agent} className="border-t">
                    <td className="py-1.5 capitalize">{a.agent}</td>
                    <td className="tabular text-right">{a.shown.toLocaleString()}</td>
                    {a.value_measured
                      ? <><td className="tabular text-right">{money(a.claimed_dollars, true)}</td>
                        <td className="tabular text-right">{a.claimed_dollars_per_problem != null ? `${money(a.claimed_dollars_per_problem, true)} (${a.problems_alerted.toLocaleString()})` : "–"}</td><td className="tabular text-right">{money(a.truth_dollars_total, true)}</td>
                        <td className={`tabular text-right font-medium ${a.net < 0 ? "text-destructive" : ""}`}>{money(a.net, true)}</td></>
                      : <td colSpan={4} className="text-right text-muted-foreground">Value not measured: no recorded dollars for this agent{a.claimed_dollars_per_problem != null ? ` (claims ${money(a.claimed_dollars_per_problem, true)} across ${a.problems_alerted} problems)` : ""}</td>}
                  </tr>))}
                  <tr className="border-t font-semibold"><td className="py-1.5">Measured agents</td><td /><td /><td /><td /><td className="tabular text-right">{money(data.total.net, true)}</td></tr>
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground">
              {data.break_even_acceptance != null
                ? <>Across the measured agents the alerts pay for themselves once people act on <b>{pct(data.break_even_acceptance, 0)}</b> of them. </>
                : <>Under these costs the measured alerts do not pay for themselves at any acceptance. </>}
              The alert-level claim counts one problem again on every day it is alerted, so it overstates; the per-problem claim keeps the largest alert for each problem (alerts on the same item within 10 days). Never add up alert-level dollars.
            </p>
            <Badge variant="outline">Evidence: {data.evidence}</Badge>
            <p className="text-xs text-muted-foreground">{data.caveat}</p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
