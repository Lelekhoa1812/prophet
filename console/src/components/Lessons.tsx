import { useApi, post } from "@/lib/api";
import { Button, Card, CardContent, CardHeader, CardTitle, CardDescription, EmptyState } from "@/components/ui";

/** Every lesson the business has taught the agents, with an undo. Rule from FoodFlow: learning is visible and reversible. */
export function Lessons({ dataset, onChange }: { dataset: string; onChange?: () => void }) {
  const { data, reload } = useApi<any[]>(`/lessons?dataset=${dataset}`);
  const undo = async (id: string) => { await post(`/lessons/${id}/undo`); reload(); onChange?.(); };
  return (
    <Card>
      <CardHeader>
        <CardTitle>What the agents have learned</CardTitle>
        <CardDescription>Each dismissal with a reason becomes a lesson. It stops repeat alerts for that customer or product for a set period. Undo removes it at once.</CardDescription>
      </CardHeader>
      <CardContent>
        {(!data || data.length === 0) && <EmptyState title="No lessons yet">Dismiss an alert with a reason and it will appear here.</EmptyState>}
        <ul className="divide-y">
          {data?.map((l) => (
            <li key={l.lesson_id} className="flex flex-wrap items-center justify-between gap-3 py-2.5 text-sm">
              <div className="min-w-0">
                <p className="font-medium">{l.agent}: {l.scope_id}</p>
                <p className="text-xs text-muted-foreground">Suppress for {l.effect.suppress_days} days · reason: {String(l.effect.reason).replace(/_/g, " ")} · {l.active ? "active" : "undone"}</p>
              </div>
              {l.active && <Button size="sm" variant="outline" onClick={() => undo(l.lesson_id)}>Undo</Button>}
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
