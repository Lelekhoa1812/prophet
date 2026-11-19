import { useState } from "react";
import { useApi, put, type AgentInfo } from "@/lib/api";
import { useApp } from "@/lib/context";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, CardDescription, InlineAlert, Input, PageHeader, Select, Switch, Skeleton, Stat } from "@/components/ui";
import { LADDER_TEXT } from "@/components/ProposalCard";
import { Lessons } from "@/components/Lessons";
import { pct } from "@/lib/utils";

const TRIGGERS: Record<string, string> = {
  shortfall: "Each morning, before the briefing. Re-checks when a PO is raised, received or late.",
  freshness: "Each morning, and when a lot is received. Looks ahead to the use-by date.",
  customer: "Mondays for patterns (slowdowns, lapses). Every day for today's order lines (keying).",
  margin: "Mondays for cost drift. Every day for today's order lines (price below floor).",
};
const LADDER_ORDER = ["explains", "suggests", "prepares", "acts", "runs"];

/** Agent control: enable, owner, quiet hours, trigger, thresholds, autonomy ladder, and what it has learned. */
const POLICY_AGENTS = ["shortfall", "freshness"];
const POLICY_KEYS = ["risk_threshold", "min_dollars", "min_share_at_risk"];

export function Agents() {
  const { dataset } = useApp();
  const { data, error, loading, reload } = useApi<AgentInfo[]>(`/agents?dataset=${dataset}`);
  return (
    <>
      <PageHeader eyebrow="Control" title="Agents, triggers and autonomy"
        description="Each agent has an owner, quiet hours and a cooldown. Autonomy only grows when the evidence says it has earned it, and it steps back after a miss." />
      {error && <InlineAlert title="Agents could not load" onRetry={reload}>{error}</InlineAlert>}
      {loading && !data && <div className="grid gap-4 lg:grid-cols-2"><Skeleton className="h-72" /><Skeleton className="h-72" /></div>}
      <div className="grid gap-5 lg:grid-cols-2">
        {data?.map((a) => <AgentCard key={a.key} a={a} dataset={dataset} onSaved={reload} />)}
      </div>
      <section className="mt-8" aria-label="Learned lessons"><Lessons dataset={dataset} onChange={reload} /></section>
    </>
  );
}

function AgentCard({ a, dataset, onSaved }: { a: AgentInfo; dataset: string; onSaved: () => void }) {
  const [cfg, setCfg] = useState(a.config);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const dirty = JSON.stringify(cfg) !== JSON.stringify(a.config);
  const save = async () => {
    setSaving(true); setNote(null);
    try { await put(`/agents/${a.key}?dataset=${dataset}`, cfg); setNote("Saved."); onSaved(); }
    catch (e: any) { setNote(`Not saved: ${e.message}`); } finally { setSaving(false); }
  };
  const L = a.ladder;
  const s = a.stats ?? {};
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle>{a.title}</CardTitle>
            <CardDescription>{a.summary}</CardDescription>
          </div>
          <label className="flex items-center gap-2 text-xs"><Switch checked={cfg.enabled} onCheckedChange={(v) => setCfg({ ...cfg, enabled: v })} aria-label={`${a.title} enabled`} />{cfg.enabled ? "On" : "Off"}</label>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3 rounded-md bg-menu-hover p-3">
          <Stat label="Useful" value={pct(s.precision)} tone={s.precision != null && s.precision >= 0.6 ? "good" : "default"} />
          <Stat label="Caught" value={pct(s.recall)} />
          <Stat label="Median lead" value={s.median_delay_days != null ? `${Math.round(s.median_delay_days)} d` : "–"} sub="before the event" />
        </div>
        <p className="text-xs text-muted-foreground"><span className="font-medium text-foreground">Trigger:</span> {TRIGGERS[a.key]}</p>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <label className="space-y-1"><span className="text-xs text-muted-foreground">Owner</span>
            <Input value={cfg.owner} onChange={(e) => setCfg({ ...cfg, owner: e.target.value })} /></label>
          <label className="space-y-1"><span className="text-xs text-muted-foreground">Cooldown (days)</span>
            <Input type="number" min={0} max={60} value={cfg.cooldown_days} onChange={(e) => setCfg({ ...cfg, cooldown_days: Number(e.target.value) })} /></label>
          <label className="space-y-1"><span className="text-xs text-muted-foreground">Quiet from</span>
            <Input type="time" value={cfg.quiet_start} onChange={(e) => setCfg({ ...cfg, quiet_start: e.target.value })} /></label>
          <label className="space-y-1"><span className="text-xs text-muted-foreground">Quiet until</span>
            <Input type="time" value={cfg.quiet_end} onChange={(e) => setCfg({ ...cfg, quiet_end: e.target.value })} /></label>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium">Autonomy</p>
          <div className="flex flex-wrap items-center gap-1" role="radiogroup" aria-label="Autonomy level">
            {LADDER_ORDER.map((k) => (
              <button key={k} role="radio" aria-checked={cfg.autonomy_level === k} onClick={() => setCfg({ ...cfg, autonomy_level: k })}
                className={`rounded-full border px-2.5 py-1 text-xs cursor-pointer ${cfg.autonomy_level === k ? "border-primary bg-accent font-semibold" : "hover:bg-menu-hover"}`}>{LADDER_TEXT[k]}</button>
            ))}
          </div>
          <div className="rounded-md border p-3 text-xs">
            <p><span className="font-medium">Current: {LADDER_TEXT[cfg.autonomy_level]}.</span>{L.next_label ? ` Next: ${L.next_label}.` : " Top of the ladder."}</p>
            {L.criteria.length > 0 && (
              <ul className="mt-2 space-y-1">
                {L.criteria.map((c) => (
                  <li key={c.label} className="flex items-center justify-between gap-2">
                    <span>{c.label}</span>
                    <span className="tabular">{c.value == null ? "–" : typeof c.value === "number" && c.value <= 1 && c.label.toLowerCase().includes("precision") ? pct(c.value) : String(c.value)} / target {c.target ?? "–"} <Badge variant={c.met ? "success" : "neutral"}>{c.met ? "met" : "not yet"}</Badge></span>
                  </li>
                ))}
              </ul>
            )}
            {L.demote && <p className="mt-2 text-amber-700 dark:text-amber-300">Recent precision has fallen. Step back one level until it recovers.</p>}
          </div>
        </div>

        <details className="text-xs">
          <summary className="cursor-pointer font-medium">Thresholds (advanced)</summary>
          {POLICY_AGENTS.includes(a.key) && <p className="mt-2 text-muted-foreground">How many alerts this agent raises is set once for the client under Developer, "Client set-up and rollout". The chance and dollar gates were removed on purpose.</p>}
          <div className="mt-2 grid grid-cols-2 gap-2">
            {Object.entries(a.defaults).filter(([k]) => !(POLICY_AGENTS.includes(a.key) && POLICY_KEYS.includes(k))).map(([k, v]) => (
              <label key={k} className="space-y-1"><span className="text-muted-foreground">{k.replace(/_/g, " ")}</span>
                <Input type="number" step="any" value={cfg.params[k] ?? v} onChange={(e) => setCfg({ ...cfg, params: { ...cfg.params, [k]: Number(e.target.value) } })} /></label>
            ))}
          </div>
        </details>

        <div className="flex flex-wrap items-center gap-3 border-t pt-3">
          <Button size="sm" onClick={save} loading={saving} disabled={!dirty || saving}>Save changes</Button>
          {dirty && <Button size="sm" variant="ghost" onClick={() => setCfg(a.config)}>Discard</Button>}
          {note && <span className="text-xs text-muted-foreground" aria-live="polite">{note}</span>}
          <span className="ml-auto text-xs text-muted-foreground">Operator decisions: {Object.entries(a.operator_decisions).map(([k, v]) => `${k} ${v}`).join(", ") || "none yet"}</span>
        </div>
      </CardContent>
    </Card>
  );
}
