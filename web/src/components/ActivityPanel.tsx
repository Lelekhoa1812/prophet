import {
  Bot,
  ChevronDown,
  ChevronRight,
  Code2,
  File,
  FileJson,
  FileSpreadsheet,
  FileText,
  Image as ImageIcon,
  PanelRightClose,
  Sparkles,
  Wrench,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { isRunning, useActivity, type ActivityTarget, type AgentRow, type CodeRun, type FileGroup, type OutputFile } from "../lib/activity";
import { formatDuration, humanise } from "../lib/format";
import { cn, formatBytes } from "../lib/utils";
import { AgentSessionView } from "./AgentSession";
import { IconButton, Spinner } from "./ui";

// Plain words for the states a task, a code run or a sub-agent can be in.
const STATE: Record<string, { label: string; tone: "ember" | "ok" | "bad" | "sand" }> = {
  running: { label: "Working", tone: "ember" },
  created: { label: "Starting", tone: "ember" },
  started: { label: "Running", tone: "ember" },
  done: { label: "Done", tone: "ok" },
  met: { label: "Done", tone: "ok" },
  completed: { label: "Done", tone: "ok" },
  partial: { label: "Partly done", tone: "ember" },
  not_met: { label: "Not finished", tone: "sand" },
  failed: { label: "Failed", tone: "bad" },
  stalled: { label: "Stuck", tone: "bad" },
  stopped: { label: "Stopped", tone: "sand" },
  cancelled: { label: "Stopped", tone: "sand" },
  checkpoint: { label: "Paused", tone: "ember" },
  waiting_user: { label: "Needs your answer", tone: "ember" },
};

export function stateOf(status: string) {
  return STATE[status] ?? { label: humanise(status), tone: "sand" as const };
}

export function StateDot({ status }: { status: string }) {
  const s = stateOf(status);
  return (
    <span
      aria-hidden="true"
      className={cn(
        "mt-1.5 size-2 shrink-0 rounded-full",
        s.tone === "ember" && "bg-ember motion-safe:animate-pulse",
        s.tone === "ok" && "bg-ok",
        s.tone === "bad" && "bg-bad",
        s.tone === "sand" && "bg-sand-2",
      )}
    />
  );
}

export function agentName(a: { agent_id: string; lane: boolean }) {
  const n = a.agent_id.replace(/^a/, "");
  return a.lane ? `Lane ${n}` : `Sub-agent ${n}`;
}

function remembered(key: string, fallback: boolean): boolean {
  try {
    const v = window.localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}

function Section({ id, title, count, icon, children, defaultOpen = true }: { id: string; title: string; count?: number; icon: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const key = `prophet.activity.${id}`;
  const [open, setOpen] = useState(() => remembered(key, defaultOpen));
  const toggle = () => {
    setOpen((o) => {
      try {
        window.localStorage.setItem(key, o ? "0" : "1");
      } catch {
        /* storage blocked */
      }
      return !o;
    });
  };
  return (
    <section className="border-b border-sand-2/15 last:border-b-0">
      <h3>
        <button
          type="button"
          aria-expanded={open}
          onClick={toggle}
          className="flex min-h-10 w-full items-center gap-2 px-3 text-left text-caption font-semibold text-sand-0 hover:bg-night-2"
        >
          {open ? <ChevronDown aria-hidden="true" className="size-4 text-sand-2" /> : <ChevronRight aria-hidden="true" className="size-4 text-sand-2" />}
          <span className="text-sand-2">{icon}</span>
          <span className="flex-1">{title}</span>
          {count !== undefined && <span className="rounded-md bg-night-2 px-1.5 text-micro font-normal text-sand-1 tabular-nums">{count}</span>}
        </button>
      </h3>
      {open && <div className="px-3 pb-3">{children}</div>}
    </section>
  );
}

function CodeRunRow({ run }: { run: CodeRun }) {
  const s = stateOf(run.status);
  return (
    <li className="flex items-start gap-2 py-1">
      <StateDot status={run.status} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-caption text-sand-0" title={run.title}>{run.title}</p>
        <p className="text-micro text-sand-2">
          {s.label} · {run.agent === "lead" ? "main thread" : agentName({ agent_id: run.agent, lane: false }).toLowerCase()} · step {run.step}
        </p>
      </div>
    </li>
  );
}

function AgentRowButton({ agent, onOpen }: { agent: AgentRow; onOpen: () => void }) {
  const s = stateOf(agent.status);
  return (
    <li>
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left hover:bg-night-2 focus-visible:bg-night-2"
      >
        <StateDot status={agent.status} />
        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-2">
            <span className="text-caption font-semibold text-sand-0">{agentName(agent)}</span>
            <span className="shrink-0 text-micro text-sand-2">{s.label}</span>
          </span>
          <span className="line-clamp-2 text-micro text-sand-1">{agent.task || humanise(agent.agent_type)}</span>
          <span className="text-micro text-sand-2">
            {humanise(agent.agent_type)} · {agent.tool_calls} tool {agent.tool_calls === 1 ? "call" : "calls"}
          </span>
        </span>
        <ChevronRight aria-hidden="true" className="mt-1 size-4 shrink-0 text-sand-2" />
      </button>
    </li>
  );
}

function fileIcon(f: OutputFile) {
  const cls = "size-4 shrink-0 text-sand-2";
  if (f.kind === "chart") return <ImageIcon aria-hidden="true" className={cls} />;
  if (f.kind === "table") return <FileSpreadsheet aria-hidden="true" className={cls} />;
  if (f.kind === "document") return <FileText aria-hidden="true" className={cls} />;
  if (f.kind === "data") return <FileJson aria-hidden="true" className={cls} />;
  return <File aria-hidden="true" className={cls} />;
}

function FileGroups({ groups }: { groups: FileGroup[] }) {
  if (groups.length === 0) return <p className="text-micro text-sand-2">No files yet.</p>;
  return (
    <div className="grid gap-3">
      {groups.map((g) => (
        <div key={g.mission_id}>
          <p className="mb-1 line-clamp-2 text-micro font-semibold text-sand-1" title={g.topic}>{g.topic}</p>
          <ul className="grid">
            {g.files.map((f) => (
              <li key={`${f.run_id}/${f.name}`}>
                <a
                  href={f.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex min-h-8 items-center gap-2 rounded-md px-1.5 text-caption text-sand-0 hover:bg-night-2"
                  title={f.name}
                >
                  {fileIcon(f)}
                  <span className="min-w-0 flex-1 truncate">{f.name}</span>
                  {f.bytes != null && <span className="shrink-0 text-micro text-sand-2 tabular-nums">{formatBytes(f.bytes)}</span>}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

// The right-hand panel: what the current round is doing (the main thread, code runs, sub-agents and research lanes), the
// files kept so far without duplicates and grouped by question, and the tools and skills this round used. A sub-agent opens
// its brief and its whole session in place.
export function ActivityPanel({ target, onClose }: { target: ActivityTarget | null; onClose: () => void }) {
  const { data, error } = useActivity(target);
  const [openAgent, setOpenAgent] = useState<string | null>(null);
  const [allRuns, setAllRuns] = useState(false);
  const missionId = data?.main?.mission_id ?? null;

  if (openAgent && missionId) {
    return <AgentSessionView missionId={missionId} agentId={openAgent} onBack={() => setOpenAgent(null)} onClose={onClose} />;
  }

  const main = data?.main ?? null;
  const running = (data?.code ?? []).filter((r) => isRunning(r.status));
  const finished = (data?.code ?? []).filter((r) => !isRunning(r.status)).reverse();
  const shownFinished = allRuns ? finished : finished.slice(0, 5);
  const agents = data?.agents ?? [];
  const liveAgents = agents.filter((a) => isRunning(a.status)).length;
  const fileCount = (data?.files ?? []).reduce((n, g) => n + g.files.length, 0);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-sand-2/20 px-3 py-2">
        <h2 className="text-caption font-semibold text-sand-0">Activity</h2>
        <IconButton label="Hide activity" icon={<PanelRightClose aria-hidden="true" className="size-5" />} onClick={onClose} />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {!target && <p className="p-3 text-caption text-sand-2">Open a notebook or a forecast to see its activity.</p>}
        {target && !data && !error && (
          <p className="flex items-center gap-2 p-3 text-caption text-sand-2"><Spinner className="size-4" /> Loading</p>
        )}
        {error && <p role="alert" className="p-3 text-caption text-bad">{error}</p>}
        {data && (
          <>
            <Section id="tasks" title="Background tasks" count={(main && isRunning(main.status) ? 1 : 0) + running.length + liveAgents} icon={<Code2 className="size-4" />}>
              {main ? (
                <div className="mb-2 flex items-start gap-2 rounded-lg bg-night-2/60 p-2">
                  <StateDot status={main.status} />
                  <div className="min-w-0 flex-1">
                    <p className="text-caption font-semibold text-sand-0">Main thread</p>
                    <p className="line-clamp-2 text-micro text-sand-1" title={main.title}>{main.title}</p>
                    <p className="text-micro text-sand-2">
                      {stateOf(main.status).label}
                      {main.step != null && ` · step ${main.step}`}
                      {main.tool_calls != null && ` · ${main.tool_calls} tool calls`}
                      {main.elapsed_s != null && ` · ${formatDuration(main.elapsed_s)}`}
                    </p>
                    {isRunning(main.status) && main.next && <p className="mt-0.5 text-micro text-sand-1">Next: {main.next}</p>}
                  </div>
                </div>
              ) : (
                <p className="text-micro text-sand-2">No question asked yet.</p>
              )}
              {(running.length > 0 || finished.length > 0) && (
                <>
                  <p className="mt-2 mb-0.5 text-micro font-semibold text-sand-2">Code</p>
                  <ul>
                    {running.map((r) => <CodeRunRow key={`r${r.step}`} run={r} />)}
                    {shownFinished.map((r) => <CodeRunRow key={`f${r.step}`} run={r} />)}
                  </ul>
                  {finished.length > 5 && (
                    <button type="button" onClick={() => setAllRuns((v) => !v)} className="mt-1 text-micro text-sand-2 underline-offset-2 hover:text-sand-0 hover:underline">
                      {allRuns ? "Show fewer" : `Show all ${finished.length}`}
                    </button>
                  )}
                </>
              )}
              {agents.length > 0 && (
                <>
                  <p className="mt-2 mb-0.5 flex items-center gap-1.5 text-micro font-semibold text-sand-2">
                    <Bot aria-hidden="true" className="size-3.5" /> Sub-agents ({agents.length})
                  </p>
                  <ul className="grid gap-0.5">
                    {agents.map((a) => <AgentRowButton key={a.agent_id} agent={a} onOpen={() => setOpenAgent(a.agent_id)} />)}
                  </ul>
                </>
              )}
            </Section>
            <Section id="outputs" title="Outputs" count={fileCount} icon={<File className="size-4" />}>
              <FileGroups groups={data.files} />
            </Section>
            <Section id="loaded" title="Tools and skills this round" count={data.tools.length + data.skills.length} icon={<Wrench className="size-4" />}>
              {data.skills.length > 0 && (
                <>
                  <p className="mb-1 flex items-center gap-1.5 text-micro font-semibold text-sand-2"><Sparkles aria-hidden="true" className="size-3.5" /> Skills</p>
                  <ul className="mb-2 grid gap-0.5">
                    {data.skills.map((s) => (
                      <li key={s.name} className="text-caption text-sand-0">
                        {s.title} <span className="text-micro text-sand-2">· {s.source}</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {data.tools.length > 0 ? (
                <>
                  <p className="mb-1 flex items-center gap-1.5 text-micro font-semibold text-sand-2"><Wrench aria-hidden="true" className="size-3.5" /> Tools</p>
                  <ul className="flex flex-wrap gap-1">
                    {data.tools.map((t) => (
                      <li key={t.name} className="rounded-md border border-sand-2/25 px-1.5 py-0.5 text-micro text-sand-1" title={`${t.calls} ${t.calls === 1 ? "call" : "calls"}`}>
                        {t.title} <span className="text-sand-2 tabular-nums">×{t.calls}</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : (
                data.skills.length === 0 && <p className="text-micro text-sand-2">Nothing used yet this round.</p>
              )}
            </Section>
          </>
        )}
      </div>
    </div>
  );
}
