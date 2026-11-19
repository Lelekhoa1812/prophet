import { PageHeader, Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui";

/** How it works: the data path, the trigger path, the proposal lifecycle, and the production design on AWS. Diagrams are hand-built SVG. */
export function Method() {
  return (
    <>
      <PageHeader eyebrow="Method" title="How the agents work, end to end"
        description="A question first, then the machinery. Each diagram shows one part of the flow, from data to a decision." />

      <svg width="0" height="0" className="absolute" aria-hidden>
        <defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted-foreground)" /></marker></defs>
      </svg>
      <Card className="mb-6">
        <CardHeader><CardTitle>1 · From the business to a forecast</CardTitle><CardDescription>Only data that existed on the decision day is used.</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <svg viewBox="0 0 960 170" className="w-full min-w-[720px]" role="img" aria-label="Data flows from orders, stock and purchases into weekly series, then TimesFM, then bands">
            <Box x={10} y={40} w={170} h={70} t="Orders, stock, lots, POs" s="Read-only snapshot" />
            <Arrow x1={180} x2={214} y={75} />
            <Box x={214} y={40} w={170} h={70} t="Weekly series" s="Per product and customer, complete weeks only" />
            <Arrow x1={384} x2={418} y={75} />
            <Box x={418} y={40} w={170} h={70} t="TimesFM 3" s="Quantiles 10 to 90" accent />
            <Arrow x1={588} x2={622} y={75} />
            <Box x={622} y={40} w={170} h={70} t="Bands and baselines" s="Compared with moving averages" />
            <Arrow x1={792} x2={826} y={75} />
            <Box x={826} y={40} w={124} h={70} t="Agent" s="Money at stake" />
          </svg>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader><CardTitle>2 · When an agent wakes up</CardTitle><CardDescription>Scheduled, event-driven and on request. Quiet hours and cooldowns apply to all three.</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <svg viewBox="0 0 960 230" className="w-full min-w-[720px]" role="img" aria-label="Triggers: schedule, events and requests feed the governor, which gates proposals">
            <Box x={10} y={20} w={200} h={52} t="Schedule" s="05:00 daily; Mondays for patterns" />
            <Box x={10} y={89} w={200} h={52} t="Events" s="PO raised, received, late; price changed" />
            <Box x={10} y={158} w={200} h={52} t="Request" s="Asked from the briefing or explorer" />
            <Arrow x1={210} x2={260} y={46} /><Arrow x1={210} x2={260} y={115} /><Arrow x1={210} x2={260} y={184} />
            <Box x={260} y={70} w={210} h={90} t="Agent run" s="Reads only what was known on that day" accent />
            <Arrow x1={470} x2={520} y={115} />
            <Box x={520} y={70} w={200} h={90} t="Governor" s="Dedupe · cooldown · lessons · quiet hours" />
            <Arrow x1={720} x2={770} y={115} />
            <Box x={770} y={70} w={180} h={90} t="Briefing" s="Owner gets one decision, with evidence" />
          </svg>
        </CardContent>
      </Card>

      <Card className="mb-6">
        <CardHeader><CardTitle>3 · What happens to a proposal</CardTitle><CardDescription>Nothing leaves the lab. In FoodFlow, each state maps to a draft that a person confirms.</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <svg viewBox="0 0 960 150" className="w-full min-w-[720px]" role="img" aria-label="Proposal states: open, then approved, adjusted or dismissed; a dismissal can become a lesson that can be undone">
            <Box x={10} y={55} w={150} h={50} t="Open" s="Waiting for an owner" accent />
            <Arrow x1={160} x2={220} y={80} />
            <Box x={220} y={10} w={170} h={44} t="Approved" s="Draft confirmed" />
            <Box x={220} y={68} w={170} h={44} t="Adjusted" s="Changed, then confirmed" />
            <Box x={220} y={126} w={170} h={44} t="Dismissed" s="With a reason" />
            <Arrow x1={390} x2={450} y={126} />
            <Box x={450} y={100} w={220} h={44} t="Lesson" s="Stops repeats for a set period" />
            <Arrow x1={670} x2={730} y={122} />
            <Box x={730} y={100} w={220} h={44} t="Undo" s="Removes the lesson at once" />
          </svg>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>4 · Production shape on AWS (proposed)</CardTitle><CardDescription>Ap-southeast-2 only, behind the Axis model gateway. Not built yet; this lab proves the method first.</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <svg viewBox="0 0 960 200" className="w-full min-w-[720px]" role="img" aria-label="Production: FoodFlow emits events, a forecast service runs TimesFM, Axis governs proposals">
            <Box x={10} y={60} w={170} h={64} t="FoodFlow" s="Orders, stock, POs, prices" />
            <Arrow x1={180} x2={230} y={92} />
            <Box x={230} y={60} w={180} h={64} t="Event feed" s="Read models, no direct writes" />
            <Arrow x1={410} x2={460} y={92} />
            <Box x={460} y={40} w={190} h={104} t="Forecast service" s="ECS, batch-first, TimesFM 3 (licence first)" accent />
            <Arrow x1={650} x2={700} y={92} />
            <Box x={700} y={60} w={250} h={64} t="Axis proposals" s="Governed, evidence, undo, human OK" />
          </svg>
        </CardContent>
      </Card>
    </>
  );
}

function Box({ x, y, w, h, t, s, accent }: { x: number; y: number; w: number; h: number; t: string; s: string; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={accent ? "var(--accent)" : "var(--card)"} stroke={accent ? "var(--primary)" : "var(--border)"} strokeWidth={accent ? 1.5 : 1} />
      <text x={x + 12} y={y + 24} fontSize="13" fontWeight="600" fill="var(--foreground)">{t}</text>
      <text x={x + 12} y={y + 42} fontSize="11" fill="var(--muted-foreground)">{s}</text>
    </g>
  );
}
function Arrow({ x1, x2, y }: { x1: number; x2: number; y: number }) {
  return <line x1={x1} x2={x2} y1={y} y2={y} stroke="var(--muted-foreground)" strokeWidth="1.5" markerEnd="url(#arr)" />;
}
