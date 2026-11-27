import { ArrowLeft, CheckCircle2, CircleAlert, Lightbulb, MessagesSquare, PanelRightClose, Wrench } from "lucide-react";
import { useState } from "react";
import { useAgentSession, type SessionEntry } from "../lib/activity";
import { humanise } from "../lib/format";
import { cn } from "../lib/utils";
import { agentName, stateOf, StateDot } from "./ActivityPanel";
import { IconButton, Spinner } from "./ui";

function Entry({ e }: { e: SessionEntry }) {
  if (e.role === "plan") {
    return (
      <li className="flex gap-2">
        <Lightbulb aria-hidden="true" className="mt-1 size-4 shrink-0 text-ember" />
        <div className="min-w-0 flex-1">
          <p className="text-micro font-semibold text-sand-2">
            Step {e.step} · {e.action === "call_tool" && e.tool ? `calls ${humanise(e.tool)}` : humanise(e.action ?? "plan")}
          </p>
          {e.rationale && <p className="text-caption whitespace-pre-wrap text-sand-0 [overflow-wrap:anywhere]">{e.rationale}</p>}
          {e.args && (
            <details className="mt-0.5">
              <summary className="cursor-pointer text-micro text-sand-2 hover:text-sand-0">Arguments</summary>
              <pre className="mt-1 max-h-60 overflow-auto rounded-md bg-night-0 p-2 font-mono text-micro whitespace-pre-wrap text-sand-1 [overflow-wrap:anywhere]">{e.args}</pre>
            </details>
          )}
        </div>
      </li>
    );
  }
  if (e.role === "result") {
    return (
      <li className="flex gap-2">
        {e.ok ? <Wrench aria-hidden="true" className="mt-1 size-4 shrink-0 text-sand-2" /> : <CircleAlert aria-hidden="true" className="mt-1 size-4 shrink-0 text-bad" />}
        <details className="min-w-0 flex-1">
          <summary className="cursor-pointer text-micro font-semibold text-sand-2 hover:text-sand-0">
            {e.ok ? "Result" : "Error"} from {humanise(e.tool ?? "tool")}
          </summary>
          <pre className="mt-1 max-h-80 overflow-auto rounded-md bg-night-0 p-2 font-mono text-micro whitespace-pre-wrap text-sand-1 [overflow-wrap:anywhere]">{e.text}</pre>
        </details>
      </li>
    );
  }
  if (e.role === "evaluation") {
    const s = stateOf(e.verdict ?? "");
    return (
      <li className="flex gap-2">
        <CheckCircle2 aria-hidden="true" className={cn("mt-1 size-4 shrink-0", s.tone === "ok" ? "text-ok" : "text-sand-2")} />
        <div className="min-w-0 flex-1">
          <p className="text-micro font-semibold text-sand-2">Check · {s.label}</p>
          <p className="text-caption whitespace-pre-wrap text-sand-1 [overflow-wrap:anywhere]">{e.text}</p>
        </div>
      </li>
    );
  }
  return (
    <li className="flex gap-2">
      <MessagesSquare aria-hidden="true" className="mt-1 size-4 shrink-0 text-sand-2" />
      <details className="min-w-0 flex-1">
        <summary className="cursor-pointer text-micro font-semibold text-sand-2 hover:text-sand-0">Messages from the other lanes</summary>
        <pre className="mt-1 max-h-60 overflow-auto rounded-md bg-night-0 p-2 font-mono text-micro whitespace-pre-wrap text-sand-1 [overflow-wrap:anywhere]">{e.text}</pre>
      </details>
    </li>
  );
}

// One sub-agent in full: its brief, the report it returned, and its session in order (each plan, tool result, message read
// from the board, and check).
export function AgentSessionView({ missionId, agentId, onBack, onClose }: { missionId: string; agentId: string; onBack: () => void; onClose: () => void }) {
  const { session, error } = useAgentSession(missionId, agentId);
  const [fullTask, setFullTask] = useState(false);
  const title = session ? agentName(session) : `Sub-agent ${agentId.replace(/^a/, "")}`;
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-1 border-b border-sand-2/20 px-2 py-2">
        <IconButton label="Back to activity" icon={<ArrowLeft aria-hidden="true" className="size-5" />} onClick={onBack} />
        <h2 className="min-w-0 flex-1 truncate text-caption font-semibold text-sand-0">{title}</h2>
        <IconButton label="Hide activity" icon={<PanelRightClose aria-hidden="true" className="size-5" />} onClick={onClose} />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {error && <p role="alert" className="text-caption text-bad">{error}</p>}
        {!session && !error && <p className="flex items-center gap-2 text-caption text-sand-2"><Spinner className="size-4" /> Loading</p>}
        {session && (
          <div className="grid gap-4">
            <div className="flex items-start gap-2">
              <StateDot status={session.status} />
              <p className="text-micro text-sand-2">
                {stateOf(session.status).label} · {humanise(session.agent_type)}
                {session.lane && " · research lane"}
              </p>
            </div>
            <div>
              <p className="mb-1 text-micro font-semibold text-sand-2">Task</p>
              <p className={cn("text-caption whitespace-pre-wrap text-sand-0 [overflow-wrap:anywhere]", !fullTask && "line-clamp-6")}>
                {session.task || "No brief was recorded."}
              </p>
              {session.task.length > 360 && (
                <button type="button" onClick={() => setFullTask((v) => !v)} className="mt-1 text-micro text-sand-2 underline-offset-2 hover:text-sand-0 hover:underline">
                  {fullTask ? "Show less" : "Show the whole brief"}
                </button>
              )}
              {session.deliverable && (
                <>
                  <p className="mt-2 mb-1 text-micro font-semibold text-sand-2">Asked to return</p>
                  <p className="text-caption whitespace-pre-wrap text-sand-1 [overflow-wrap:anywhere]">{session.deliverable}</p>
                </>
              )}
            </div>
            {(session.answer || session.findings.length > 0) && (
              <div className="rounded-lg border border-sand-2/25 p-2">
                <p className="mb-1 text-micro font-semibold text-sand-2">Report</p>
                {session.answer && <p className="text-caption whitespace-pre-wrap text-sand-0 [overflow-wrap:anywhere]">{session.answer}</p>}
                {session.findings.length > 0 && (
                  <ul className="mt-1 list-disc pl-5 text-caption text-sand-1">
                    {session.findings.map((f, k) => <li key={k}>{f}</li>)}
                  </ul>
                )}
              </div>
            )}
            <div>
              <p className="mb-2 text-micro font-semibold text-sand-2">Session ({session.entries.length} entries)</p>
              {session.entries.length === 0 ? (
                <p className="text-micro text-sand-2">No steps yet.</p>
              ) : (
                <ol className="grid gap-3">
                  {session.entries.map((e, k) => <Entry key={`${e.step}-${k}`} e={e} />)}
                </ol>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
