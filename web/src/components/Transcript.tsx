import type { ReactNode } from "react";
import type { EventRow, Mission, StepRow } from "../lib/api";
import { formatTime, truncate } from "../lib/format";
import { cn } from "../lib/utils";
import { QuestionCard } from "./QuestionCard";
import { Card, Label, StatusPill } from "./ui";

const STEP_LABEL: Record<string, string> = { plan: "Plan", act: "Tool step", ask: "Question", eval: "Check progress" };

const EVENT_LABEL: Record<string, string> = {
  "tool.started": "Tool started",
  "tool.done": "Tool finished",
  "tool.failed": "Tool failed",
  "question.opened": "Waiting for your answer",
  "mission.outcome_unknown": "Stopped: outcome unknown, a person must decide",
  "mission.budget_exhausted": "Stopped: budget reached",
  "mission.failed": "Mission failed",
  "service.error": "Service error",
};

function stepTitle(step: StepRow): string {
  const prefix = step.name.split(":")[0];
  const base = STEP_LABEL[prefix] ?? step.name;
  const tool = step.output && typeof step.output === "object" ? (step.output as { tool?: unknown }).tool : undefined;
  return prefix === "act" && typeof tool === "string" ? `Tool: ${tool}` : base;
}

// Turns a stored step output into one plain sentence. The full output stays one click away.
function stepSummary(step: StepRow): string {
  const o = step.output;
  if (o == null) return step.status === "done" ? "Done" : "Working…";
  if (typeof o === "string") return truncate(o);
  const rec = o as Record<string, unknown>;
  for (const key of ["rationale", "answer", "reason", "summary"]) {
    if (typeof rec[key] === "string" && rec[key]) return truncate(rec[key] as string);
  }
  if (typeof rec.error === "string") return `Error: ${truncate(rec.error)}`;
  return "Recorded";
}

function FinalAnswer({ answer }: { answer: Record<string, unknown> }) {
  const status = String(answer.status ?? "");
  const iterations = typeof answer.iterations === "number" ? answer.iterations : null;
  const text =
    status === "met"
      ? String(answer.answer ?? "")
      : status === "impossible"
        ? String(answer.reason ?? "The planner judged the goal impossible with the available tools.")
        : status === "stalled"
          ? "The planner stopped making progress: the same tool results kept repeating."
          : status === "iteration_budget"
            ? "The planner used its iteration budget before reaching a verified answer."
            : "";
  return (
    <Card className="p-4 sm:p-5">
      <div className="grid gap-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Label>Final answer</Label>
          {iterations !== null && (
            <span className="text-xs text-sand-2 tabular">
              {iterations} {iterations === 1 ? "iteration" : "iterations"}
            </span>
          )}
        </div>
        <p className="whitespace-pre-wrap text-balance text-sand-0">{text || "No answer text was recorded."}</p>
      </div>
    </Card>
  );
}

function Bubble({ who, children }: { who: string; children: ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <Label>{who}</Label>
      {children}
    </div>
  );
}

export function Transcript({ mission, onAnswered }: { mission: Mission; onAnswered: () => void }) {
  const visibleEvents: EventRow[] = mission.events.filter((e) => e.kind in EVENT_LABEL);
  const working = mission.status === "created" || mission.status === "running";

  return (
    <div className="grid gap-5">
      <Bubble who="You">
        <p className="whitespace-pre-wrap text-balance rounded-lg border border-sand-2/30 bg-night-1/80 p-4 text-sand-0 [overflow-wrap:anywhere]">
          {mission.goal}
        </p>
      </Bubble>

      <div className="flex flex-wrap items-center gap-3">
        <StatusPill status={mission.status} />
        <span role="status" aria-live="polite" className={cn("text-sm text-sand-1", !working && "sr-only")}>
          {working ? "Prophet is planning and running tools…" : ""}
        </span>
      </div>

      {mission.steps.length > 0 && (
        <ol className="grid gap-3" aria-label="Planner steps">
          {mission.steps.map((step) => (
            <li key={step.idx}>
              <Card className="p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="font-medium text-sand-0">{stepTitle(step)}</span>
                  <span className="text-xs text-sand-2">
                    {step.status === "done" ? "Done" : step.status}
                  </span>
                </div>
                <p className="mt-1.5 text-sm text-sand-1 [overflow-wrap:anywhere]">{stepSummary(step)}</p>
                {step.output != null && (
                  <details className="mt-3">
                    <summary className="cursor-pointer text-xs text-sand-2 hover:text-sand-1">Full output</summary>
                    <pre className="mt-2 max-h-64 overflow-auto rounded-md bg-night-0 p-3 text-xs leading-relaxed text-sand-1 whitespace-pre-wrap [overflow-wrap:anywhere]">
                      {JSON.stringify(step.output, null, 2)}
                    </pre>
                  </details>
                )}
              </Card>
            </li>
          ))}
        </ol>
      )}

      {mission.questions.map((q) => (
        <QuestionCard key={q.id} question={q} onAnswered={onAnswered} />
      ))}

      {mission.answer && typeof mission.answer === "object" && (
        <FinalAnswer answer={mission.answer as Record<string, unknown>} />
      )}

      {visibleEvents.length > 0 && (
        <details className="group">
          <summary className="cursor-pointer text-xs uppercase tracking-[0.08em] text-sand-1 hover:text-sand-0">
            Run log ({visibleEvents.length})
          </summary>
          <ul className="mt-3 grid gap-1.5 border-l border-sand-2/40 pl-4 text-sm">
            {visibleEvents.map((e) => (
              <li key={e.seq} className="grid gap-0.5">
                <span className="text-sand-0">{EVENT_LABEL[e.kind]}</span>
                <span className="text-xs text-sand-2 tabular">
                  {formatTime(e.ts)}
                  {typeof e.payload.tool === "string" ? ` · ${e.payload.tool}` : ""}
                  {typeof e.payload.error === "string" ? ` · ${truncate(e.payload.error, 160)}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
