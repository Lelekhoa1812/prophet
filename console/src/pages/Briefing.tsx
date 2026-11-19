import { useState } from "react";
import { useApi } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ProposalCard, AGENT_LABEL } from "@/components/ProposalCard";
import { Card, CardContent, EmptyState, InlineAlert, PageHeader, Select, Skeleton, Stat } from "@/components/ui";
import { money } from "@/lib/utils";

/** The 5 am briefing: what needs you, in order of money at stake, with the rest of the week's open items. */
export function Briefing() {
  const { dataset } = useApp();
  const [owner, setOwner] = useState("");
  const [asOf, setAsOf] = useState("");
  const q = `/briefing?dataset=${dataset}${owner ? `&owner=${encodeURIComponent(owner)}` : ""}${asOf ? `&as_of=${asOf}` : ""}`;
  const { data, error, loading, reload } = useApi<any>(q);
  return (
    <>
      <PageHeader eyebrow="Operator view · 5 am briefing" title={data?.headline ?? "Morning briefing"}
        description="Each item says what changed, why, what it is worth, and the one decision you need to make. Nothing has been sent or changed."
        actions={<>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">Owner
            <Select value={owner} onChange={(e) => setOwner(e.target.value)} className="h-8 text-xs">
              <option value="">Everyone</option><option value="buyer">Buyer</option><option value="sales">Sales</option><option value="account_manager">Account manager</option><option value="order_desk">Order desk</option><option value="pricing">Pricing</option>
            </Select></label>
        </>} />
      {error && <InlineAlert title="The briefing could not load" onRetry={reload}>{error}</InlineAlert>}
      {loading && !data && <div className="space-y-3"><Skeleton className="h-40" /><Skeleton className="h-40" /></div>}
      {data && (
        <div className="grid gap-6 lg:grid-cols-4">
          <section className="space-y-4 lg:col-span-3" aria-label="Needs you">
            <div className="grid grid-cols-3 gap-4 rounded-lg border bg-card p-4">
              <Stat label="Needs you" value={data.needs_you.length} />
              <Stat label="In play" value={money(data.dollars_in_play, true)} />
              <Stat label="Briefing date" value={data.as_of} />
            </div>
            {data.phase?.phase === "shadow" && <InlineAlert tone="info" title="Shadow mode">{data.phase.label}. {data.shadow_hidden} alert{data.shadow_hidden === 1 ? " was" : "s were"} recorded for scoring only.</InlineAlert>}
            {data.policy && data.needs_you.length > 0 && (
              <p className="text-xs text-muted-foreground">Shown because each is worth at least {money(data.policy.score_floor)} in expected loss (more when less likely), within {data.policy.budget_per_owner} per owner.{data.below_floor > 0 ? ` ${data.below_floor} smaller one${data.below_floor === 1 ? " was" : "s were"} kept out.` : ""}</p>
            )}
            {data.needs_you.length === 0 && data.phase?.phase !== "shadow" && <EmptyState title="Nothing needs you on this date">Pick another date with the time machine, or check the open items on the right.</EmptyState>}
            {data.needs_you.map((p: any) => <ProposalCard key={p.proposal_id} p={p} onDecided={reload} />)}
            {data.digest?.length > 0 && (
              <Card aria-label="Digest of items not shown"><CardContent className="space-y-2 p-4 text-sm">
                <p className="font-semibold">Digest: {data.digest.reduce((a: number, g: any) => a + g.count, 0)} more, not shown this morning</p>
                <p className="text-xs text-muted-foreground">Each owner sees their top {data.policy?.budget_per_owner ?? 5} by expected dollars. Order-entry stops are never digested. Everything here stays open in Proposals.</p>
                <ul className="space-y-1 text-xs">
                  {data.digest.map((g: any) => (
                    <li key={g.owner + g.agent} className="flex justify-between gap-3 border-t pt-1">
                      <span>{g.owner.replace("_", " ")} · {AGENT_LABEL[g.agent] ?? g.agent}: {g.count}</span>
                      <span className="tabular-nums">{money(g.dollars, true)} at stake</span>
                    </li>
                  ))}
                </ul>
              </CardContent></Card>
            )}
          </section>
          <aside className="space-y-3 lg:col-span-1" aria-label="Still open">
            <h2 className="text-sm font-semibold">Still open from the last week</h2>
            {data.still_open.length === 0 && <p className="text-xs text-muted-foreground">Nothing older is waiting.</p>}
            {data.still_open.map((p: any) => (
              <Card key={p.proposal_id}><CardContent className="p-3">
                <p className="text-sm font-medium leading-snug">{p.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{p.as_of} · {money(p.dollars_at_stake)} at stake</p>
              </CardContent></Card>
            ))}
            <Card><CardContent className="p-3 text-xs text-muted-foreground">
              Counts by agent today: {Object.entries(data.by_agent).map(([k, v]) => `${k} ${v}`).join(" · ")}
            </CardContent></Card>
          </aside>
        </div>
      )}
    </>
  );
}
