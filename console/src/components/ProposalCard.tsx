import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Check, Pencil, X, ChevronRight, Sparkles } from "lucide-react";
import { Badge, Button, Card, CardContent, SEVERITY } from "@/components/ui";
import { BandChart } from "@/components/charts/BandChart";
import { post, type Proposal } from "@/lib/api";
import { useApp } from "@/lib/context";
import { money, pct } from "@/lib/utils";

export const AGENT_LABEL: Record<string, string> = { shortfall: "Shortfall", freshness: "Freshness", customer: "Customer", margin: "Margin" };
export const LADDER_TEXT: Record<string, string> = { explains: "Explains", suggests: "Suggests", prepares: "Prepares for your OK", acts: "Acts within limits", runs: "Runs the workflow" };

/** One proposal, shown the same way everywhere: what, why, how sure, what it's worth, what to do, and the evidence. */
export function ProposalCard({ p, onDecided, compact = false }: { p: Proposal; onDecided?: () => void; compact?: boolean }) {
  const { reveal, revealed, remember } = useApp();
  const [busy, setBusy] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [narr, setNarr] = useState<string | null>(null);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiNote, setAiNote] = useState<string | null>(null);
  const codes = useMemo(() => p.entity_id.split("|").filter((c) => /^(C|P)-\d{4}$/.test(c)), [p.entity_id]);
  useEffect(() => { if (reveal && codes.length) remember(codes); }, [reveal, codes, remember]);
  const sev = SEVERITY[p.severity];
  const text = (s: string) => (reveal ? s.replace(/\b(C|P)-\d{4}\b/g, (m) => (revealed[m] ? `${revealed[m]} (${m})` : m)) : s);

  const decide = async (decision: "approve" | "adjust" | "dismiss", reason?: string) => {
    setBusy(decision); setErr(null);
    try {
      await post(`/proposals/${p.proposal_id}/decision`, { decision, reason, decided_by: "reviewer" });
      setDone(decision === "approve" ? "Approved" : decision === "adjust" ? "Sent back to adjust" : "Dismissed. The agent will learn from this, and you can undo it.");
      onDecided?.();
    } catch (e: any) { setErr(String(e.message ?? e)); } finally { setBusy(null); }
  };

  const aiExplain = async () => {
    setAiBusy(true); setAiNote(null);
    try {
      const r = await post<any>(`/proposals/${p.proposal_id}/narrate`, { mode: "bedrock" });
      if (r.bedrock?.ok && r.bedrock.text) setNarr(r.bedrock.text);
      else if (r.bedrock?.rejected_numbers?.length) setAiNote(`Not used: the draft quoted figures that are not in the evidence (${r.bedrock.rejected_numbers.join(", ")}). Showing the template.`);
      else setAiNote(`Not written: ${r.bedrock?.error ?? "unknown error"}. Showing the template.`);
    } catch (e: any) { setAiNote(String(e.message ?? e)); } finally { setAiBusy(false); }
  };

  const plain = narr ?? p.narration_templated ?? "";
  const s = p.evidence.series;
  return (
    <Card className="overflow-hidden">
      <CardContent className="space-y-3 p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-1 flex flex-wrap items-center gap-1.5">
              <Badge variant={sev.variant}>{sev.label}</Badge>
              <Badge variant="brand">{AGENT_LABEL[p.agent] ?? p.agent}</Badge>
              <Badge variant="outline">{LADDER_TEXT[p.autonomy_level] ?? p.autonomy_level}</Badge>
              <span className="text-xs text-muted-foreground tabular">{p.as_of}</span>
            </div>
            <h2 className="text-base font-semibold leading-snug">{text(p.title)}</h2>
          </div>
          <div className="text-right">
            <p className="eyebrow">At stake</p>
            <p className="text-xl font-semibold tabular">{money(p.dollars_at_stake)}</p>
            {p.useful_rate == null && <p className="text-xs text-muted-foreground">Confidence {pct(p.confidence)}</p>}
            {p.useful_rate != null && <p className="text-xs text-muted-foreground">Alerts scored like this were useful {pct(p.useful_rate, 0)} of the time on this client's history</p>}
          </div>
        </div>

        <p className="text-sm leading-relaxed">{text(plain)}</p>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="ghost" size="sm" onClick={aiExplain} loading={aiBusy} disabled={aiBusy}>
            <Sparkles className="h-3.5 w-3.5" />Write it with AI (Haiku)
          </Button>
          <span className="text-[11px] text-muted-foreground">Any figure it can't source from the evidence is rejected.</span>
        </div>
        {aiNote && <p className="text-xs text-amber-700 dark:text-amber-300">{aiNote}</p>}

        {!compact && s && (
          <div className="rounded-md border bg-background p-3">
            <p className="mb-1 text-xs font-medium text-muted-foreground">{s.label}</p>
            <BandChart height={180} clip label={s.label} s={{ history: s.history, forecast: s.forecast, unit: s.unit }} />
          </div>
        )}

        {!compact && p.evidence.facts && (
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1.5 text-sm sm:grid-cols-2">
            {p.evidence.facts.map((f) => (
              <div key={f.label} className="flex items-baseline justify-between gap-3 border-b border-dashed py-1">
                <dt className="text-muted-foreground">{f.label}</dt>
                <dd className="text-right font-medium tabular">{text(String(f.value))} <span className="text-xs font-normal text-muted-foreground">{f.unit}</span></dd>
              </div>
            ))}
          </dl>
        )}

        {!compact && p.evidence.targets && p.evidence.targets.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">Who usually buys this (last 8 weeks)</p>
            <ul className="grid gap-1 text-sm sm:grid-cols-2">
              {p.evidence.targets.map((t: any) => (
                <li key={t.customer_id} className="flex justify-between gap-2 rounded bg-menu-hover px-2 py-1">
                  <span className="truncate">{reveal && revealed[t.customer_id] ? revealed[t.customer_id] : t.name}</span>
                  <span className="text-xs tabular text-muted-foreground">{t.qty_8w} units · last {t.days_since_last}d ago</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {!compact && p.evidence.why && <p className="text-xs text-muted-foreground"><span className="font-medium">Why:</span> {text(p.evidence.why)}</p>}
        {!compact && p.evidence.method && <p className="text-xs text-muted-foreground"><span className="font-medium">How it was worked out:</span> {p.evidence.method}</p>}

        {done ? (
          <p className="flex items-center gap-2 text-sm font-medium text-primary"><Check className="h-4 w-4" />{done}</p>
        ) : (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button variant="cta" size="sm" onClick={() => decide("approve")} loading={busy === "approve"} disabled={!!busy}><Check className="h-3.5 w-3.5" />Approve</Button>
            <Button variant="outline" size="sm" onClick={() => decide("adjust")} loading={busy === "adjust"} disabled={!!busy}><Pencil className="h-3.5 w-3.5" />Adjust</Button>
            <DismissMenu onPick={(r) => decide("dismiss", r)} disabled={!!busy} />
            <Link to={`/proposals/${p.proposal_id}`} aria-label={`Open proposal: ${text(p.title)}`} className="ml-auto inline-flex min-h-6 min-w-6 items-center gap-1 py-0.5 text-sm text-primary hover:underline">Open <ChevronRight className="h-3.5 w-3.5" /></Link>
          </div>
        )}
        {err && <p role="alert" className="text-xs text-red-600">{err}</p>}
      </CardContent>
    </Card>
  );
}

function DismissMenu({ onPick, disabled }: { onPick: (reason: string) => void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const reasons: [string, string][] = [["one_off", "One-off, won't repeat"], ["known_promo", "Known promotion"], ["wrong_alert", "Wrong alert"], ["not_useful", "Not useful now"]];
  return (
    <div className="relative">
      <Button variant="ghost" size="sm" onClick={() => setOpen((v) => !v)} aria-expanded={open} disabled={disabled}><X className="h-3.5 w-3.5" />Dismiss</Button>
      {open && (
        <div role="menu" className="absolute left-0 top-9 z-20 w-56 rounded-md border bg-card p-1 shadow-[var(--menu-shadow)]">
          <p className="px-2 py-1 text-xs text-muted-foreground">Why? The agent learns from this, and you can undo it.</p>
          {reasons.map(([k, l]) => (
            <button key={k} role="menuitem" onClick={() => { setOpen(false); onPick(k); }} className="block w-full cursor-pointer rounded px-2 py-1.5 text-left text-sm hover:bg-menu-hover">{l}</button>
          ))}
        </div>
      )}
    </div>
  );
}
