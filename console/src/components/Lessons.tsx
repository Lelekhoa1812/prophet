import { useState } from "react";
import { useApi, post, api } from "@/lib/api";
import { Button, Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState } from "@/components/ui";

/** Every lesson the business has taught the agents, with an undo. Rule from FoodFlow: learning is visible and reversible. */
export function Lessons({ dataset, onChange }: { dataset: string; onChange?: () => void }) {
  const { data, reload } = useApi<any[]>(`/lessons?dataset=${dataset}`);
  const [who, setWho] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const undo = async (id: string) => { await post(`/lessons/${id}/undo`); reload(); onChange?.(); };
  // Motivation vs Logic: a lesson for a whole agent is inactive until a different person approves it (design rule 5).
  // The name is sent as X-Operator; the API records it as unverified because the lab has no login.
  const approve = async (id: string) => {
    setErr(null);
    try { await api(`/lessons/${id}/approve`, { method: "POST", headers: { "Content-Type": "application/json", "X-Operator": who } }); reload(); onChange?.(); }
    catch (e: any) { setErr(String(e.message ?? e)); }
  };
  const pending = (l: any) => l.effect.approval?.status === "pending" && !l.undone_at;
  return (
    <Card>
      <CardHeader>
        <CardTitle>What the agents have learned</CardTitle>
        <CardDescription>Each dismissal with a reason becomes a lesson. It stops repeat alerts for that customer or product for a set period. A lesson for a whole agent waits for a second person to approve it. Undo removes any lesson at once.</CardDescription>
      </CardHeader>
      <CardContent>
        {(!data || data.length === 0) && <EmptyState title="No lessons yet">Dismiss an alert with a reason and it will appear here.</EmptyState>}
        {data?.some(pending) && (
          <div className="mb-3 flex flex-wrap items-center gap-2 rounded-md border p-3 text-sm">
            <label htmlFor="approver" className="font-medium">Your name, to approve a whole-agent lesson</label>
            <input id="approver" value={who} onChange={(e) => setWho(e.target.value)} className="h-9 w-48 rounded-md border bg-background px-2" />
            <span className="text-xs text-muted-foreground">Not a login. The name is recorded as unverified and must differ from the person who asked.</span>
          </div>
        )}
        {err && <p role="alert" className="mb-2 text-sm text-destructive">{err}</p>}
        <ul className="divide-y">
          {data?.map((l) => (
            <li key={l.lesson_id} className="flex flex-wrap items-center justify-between gap-3 py-2.5 text-sm">
              <div className="min-w-0">
                <p className="font-medium">{l.scope_type === "agent" ? `${l.agent}: every alert up to $${Math.round(l.effect.max_dollars ?? 0)}` : `${l.agent}: ${l.scope_id}`}</p>
                <p className="text-xs text-muted-foreground">Suppress for {l.effect.suppress_days} days · reason: {String(l.effect.reason).replace(/_/g, " ")} · {l.active ? "active" : l.undone_at ? "undone" : "waiting for approval"}</p>
              </div>
              <div className="flex gap-2">
                {pending(l) && <Button size="sm" disabled={!who.trim()} onClick={() => approve(l.lesson_id)}>Approve</Button>}
                {!l.undone_at && <Button size="sm" variant="outline" onClick={() => undo(l.lesson_id)}>Undo</Button>}
              </div>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
