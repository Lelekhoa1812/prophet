import { useEffect, useState } from "react";
import { useApi, post, api } from "@/lib/api";
import { useApp } from "@/lib/context";
import { ProposalCard } from "@/components/ProposalCard";
import { Button, Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState, InlineAlert, PageHeader, Stat } from "@/components/ui";
import { money } from "@/lib/utils";

/** Time machine: move the business's clock forward a day at a time. Every enabled agent runs as it would have that morning. */
export function TimeMachine() {
  const { dataset } = useApp();
  const { data: st, error, reload } = useApi<any>(`/time/state?dataset=${dataset}`);
  const [cursor, setCursor] = useState<string | null>(null);
  const [items, setItems] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (st) setCursor(st.cursor ?? st.start); }, [st]);
  const step = async (days: number) => {
    if (!cursor) return;
    setBusy(true); setErr(null);
    const to = new Date(cursor + "T00:00:00"); to.setDate(to.getDate() + days);
    try {
      const r = await post<any>("/time/step", { dataset, to: to.toISOString().slice(0, 10) });
      setCursor(r.cursor);
      setItems((prev) => [...r.new, ...prev].sort((a, b) => b.dollars_at_stake - a.dollars_at_stake).slice(0, 80));
    } catch (e: any) { setErr(String(e.message ?? e)); } finally { setBusy(false); }
  };
  const reset = async () => { await post(`/time/reset?dataset=${dataset}`); setItems([]); reload(); };
  if (error) return <InlineAlert title="The time machine could not load" onRetry={reload}>{error}</InlineAlert>;
  const dataTo = st?.end;
  return (
    <>
      <PageHeader eyebrow="Replay" title="Time machine: watch the agents run, day by day"
        description="Step the business forward. Each day, every enabled agent looks only at what was known that morning, and raises what it would have raised. Dismiss or approve, and the lessons carry forward."
        actions={<Button variant="outline" size="sm" onClick={reset}>Restart</Button>} />
      <div className="grid gap-5 lg:grid-cols-[20rem_1fr]">
        <Card className="h-fit">
          <CardHeader><CardTitle>This morning</CardTitle><CardDescription>Test window {st?.start} to {st?.end}</CardDescription></CardHeader>
          <CardContent className="space-y-4">
            <Stat big label="Date" value={cursor ?? "…"} sub={cursor && dataTo && cursor >= dataTo ? "End of the test window" : "Each step runs every enabled agent"} />
            <div className="grid grid-cols-3 gap-2">
              <Button variant="outline" onClick={() => step(1)} loading={busy} disabled={busy || !cursor || (dataTo && cursor >= dataTo)}>+1 day</Button>
              <Button variant="outline" onClick={() => step(7)} disabled={busy || !cursor || (dataTo && cursor >= dataTo)}>+1 week</Button>
              <Button onClick={() => step(14)} disabled={busy || !cursor || (dataTo && cursor >= dataTo)}>+2 weeks</Button>
            </div>
            {err && <p role="alert" className="text-xs text-red-600">{err}</p>}
            <p className="text-xs text-muted-foreground">Results accumulate on the right, biggest money first. Nothing is raised twice for the same issue inside its cooldown.</p>
            <div className="rounded-md bg-menu-hover p-3 text-xs">
              <p className="font-medium">What the agent could see</p>
              <p className="mt-1 text-muted-foreground">Orders delivered before this morning, stock and use-by dates on hand, purchase orders already raised, and prices already keyed. Nothing from the future.</p>
            </div>
          </CardContent>
        </Card>
        <section aria-label="Raised so far" className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">{items.length} raised so far{items.length ? `, ${money(items.reduce((a, p) => a + p.dollars_at_stake, 0), true)} at stake` : ""}</p>
          </div>
          {items.length === 0 && <EmptyState title="Step forward to begin">Each day the agents read the business and raise what they would have raised that morning.</EmptyState>}
          <div className="grid gap-4 xl:grid-cols-2">
            {items.map((p) => <ProposalCard key={p.proposal_id} p={p} compact onDecided={() => undefined} />)}
          </div>
        </section>
      </div>
    </>
  );
}
