import { AlertTriangle, Check, CheckCheck, ChevronRight, Copy, FileText, Play, RotateCcw, SkipForward } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type AnsweredQuestion, type ArtifactRun, type DatasetRef, type Mission, type ToolInfo } from "../lib/api";
import { errorMessage, formatDuration, truncate } from "../lib/format";
import { buildSteps, currentActivity, type StepView } from "../lib/steps";
import { cn, formatBytes, isCheckpoint, isWorking } from "../lib/utils";
import { Markdown } from "./Markdown";
import { MentionText } from "./MentionChip";
import { QuestionCard, SKIPPED } from "./QuestionCard";
import { Results } from "./Results";
import { Button, Card, Spinner } from "./ui";

function StepIcon({ state }: { state: StepView["state"] }) {
  if (state === "running") return <Spinner className="text-ember" />;
  if (state === "failed") return <AlertTriangle aria-label="Needed a retry" className="size-4 text-bad" />;
  return <Check aria-label="Done" className="size-4 text-ok" />;
}

function StepRow({ step }: { step: StepView }) {
  const hasDetails = step.facts.length > 0 || step.code || step.output || step.error;
  return (
    <li className="grid gap-1 py-2">
      <div className="flex items-start gap-3">
        <span className="mt-1 shrink-0">
          <StepIcon state={step.state} />
        </span>
        <div className="grid min-w-0 flex-1 gap-0.5">
          <span className="text-base font-semibold text-sand-0">{step.title}</span>
          {step.note && <span className="text-caption text-sand-1 [overflow-wrap:anywhere]">{step.note}</span>}
          {step.error && step.state === "failed" && (
            <span className="text-caption text-bad [overflow-wrap:anywhere]">{truncate(step.error, 220)}</span>
          )}
          {step.followUp && <span className="text-caption text-sand-2">Not there yet: {truncate(step.followUp, 200)}</span>}
        </div>
      </div>
      {hasDetails && (
        <details className="ml-7">
          <summary className="inline-flex min-h-10 cursor-pointer items-center gap-1 text-caption text-sand-2 hover:text-sand-0 sm:min-h-0">
            <ChevronRight aria-hidden="true" className="size-3.5 transition-transform [details[open]>summary_&]:rotate-90" />
            Details
          </summary>
          <div className="mt-2 grid gap-3">
            {step.facts.length > 0 && (
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-caption">
                {step.facts.map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-sand-2">{k}</dt>
                    <dd className="text-sand-0 [overflow-wrap:anywhere]">{v}</dd>
                  </div>
                ))}
              </dl>
            )}
            {step.code && (
              <div className="grid gap-1">
                <span className="text-caption text-sand-2">Code that ran</span>
                <pre className="max-h-72 overflow-auto rounded-lg bg-night-0 p-3 text-caption leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-0">
                  {step.code}
                </pre>
              </div>
            )}
            {step.output && (
              <div className="grid gap-1">
                <span className="text-caption text-sand-2">What it printed</span>
                <pre className="max-h-60 overflow-auto rounded-lg bg-night-0 p-3 text-caption leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-1">
                  {truncate(step.output, 4000)}
                </pre>
              </div>
            )}
            <details>
              <summary className="inline-flex min-h-10 cursor-pointer items-center text-caption text-sand-2 hover:text-sand-0 sm:min-h-0">Full record</summary>
              <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-night-0 p-3 font-mono text-meta leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-1">
                {JSON.stringify(step.raw, null, 2)}
              </pre>
            </details>
          </div>
        </details>
      )}
    </li>
  );
}

function StepsPanel({ steps, working }: { steps: StepView[]; working: boolean }) {
  const [open, setOpen] = useState(working);
  const [touched, setTouched] = useState(false);

  // Open while the agent works so progress is visible; fold away once done so the answer leads.
  useEffect(() => {
    if (!touched) setOpen(working);
  }, [working, touched]);

  if (steps.length === 0) return null;
  const retried = steps.filter((s) => s.state === "failed").length;
  const summary = working
    ? `${steps.length} ${steps.length === 1 ? "step" : "steps"} so far`
    : `Worked in ${steps.length} ${steps.length === 1 ? "step" : "steps"}${retried ? ` · ${retried} needed a retry` : ""}`;

  return (
    <details
      open={open}
      onToggle={(e) => {
        const now = (e.currentTarget as HTMLDetailsElement).open;
        if (now !== open) {
          setTouched(true);
          setOpen(now);
        }
      }}
      className="group"
    >
      <summary className="inline-flex min-h-10 cursor-pointer items-center gap-1.5 rounded-lg pr-3 text-caption text-sand-2 transition-colors hover:text-sand-0">
        <ChevronRight aria-hidden="true" className="size-4 transition-transform [details[open]>summary_&]:rotate-90" />
        {summary}
      </summary>
      <ol className="mt-1 ml-1.5 grid gap-0.5 border-l border-sand-2/25 pl-4" aria-label="What Prophet did">
        {steps.map((s) => (
          <StepRow key={s.key} step={s} />
        ))}
      </ol>
    </details>
  );
}

// What went wrong, why, and what to do next, in that order. Never a bare status code.
function AnswerActions({ text, seconds }: { text: string; seconds: number }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="-mt-1 flex items-center gap-1 text-caption text-sand-2">
      <button
        type="button"
        onClick={() => void copy()}
        aria-label={copied ? "Copied" : "Copy answer"}
        title={copied ? "Copied" : "Copy answer"}
        className="grid size-9 place-items-center rounded-lg transition-colors hover:bg-night-2 hover:text-sand-0"
      >
        {copied ? <Check aria-hidden="true" className="size-4 text-ok" /> : <Copy aria-hidden="true" className="size-4" />}
      </button>
      {seconds >= 1 && <span className="tabular-nums">Took {seconds < 60 ? `${Math.round(seconds)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`}</span>}
    </div>
  );
}

// The recovery action follows the state. "retry" resubmits the same goal at once; "edit" puts the goal back in the
// composer so the person can change it first; "none" means the person must check the work first (outcome_unknown),
// because repeating a step that may have half-run could duplicate it.
type Problem = { what: string; why: string; next: string; action: "retry" | "edit" | "none" };

function problemFor(mission: Mission): Problem | null {
  // A checkpoint is not a problem: it has its own card with Continue and Answer now.
  if (isCheckpoint(mission.status)) return null;
  const answer = mission.answer && typeof mission.answer === "object" ? (mission.answer as Record<string, any>) : {};
  const failedEvent = [...mission.events].reverse().find((e) => e.kind === "mission.failed");
  switch (mission.status) {
    case "failed":
      return {
        what: "Prophet couldn't finish this forecast.",
        why: typeof failedEvent?.payload.error === "string" ? truncate(failedEvent.payload.error, 240) : "Something went wrong while it was working.",
        next: "Try again. If it keeps happening, rephrase the request or share your data in a different layout.",
        action: "retry",
      };
    case "stalled":
      return {
        what: "Prophet stopped because it kept getting the same result.",
        why: "More attempts on the same approach would not change the outcome.",
        next: "Add detail to the request (the column to forecast, the time range, or what the numbers mean) and try again.",
        action: "retry",
      };
    case "impossible":
      return {
        what: "This can't be done with the data available.",
        why: typeof answer.reason === "string" ? answer.reason : "The data or the request does not support a reliable forecast.",
        next: "Add more history or more detail, or change what you are asking for.",
        action: "retry",
      };
    case "cancelled":
      return {
        what: "You stopped this question.",
        why: "Nothing more was done after you stopped it. The steps above show what had finished.",
        next: "Ask it again when you are ready, or change the question.",
        action: "edit",
      };
    case "outcome_unknown":
      return {
        what: "Prophet paused because a step may not have finished.",
        why: "It cannot tell whether the last action completed, and repeating it could duplicate work.",
        next: "Check the result of that step yourself, then start a new request.",
        action: "none",
      };
    default:
      return null;
  }
}

// A part of the work ended on purpose. Every step is kept; the person decides whether Prophet keeps going with a fresh
// allowance, or writes the answer now from the evidence it already has.
// The file type shown under an attachment name: the extension in capitals, or "File" when there is none.
function fileKind(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 && dot < name.length - 1 ? name.slice(dot + 1).toUpperCase() : "File";
}

// The decisions the person has made, kept in the transcript after their question card has gone: the question, then
// the answer or the fact that it was skipped. Like the record Claude keeps once a question is answered.
function Decisions({ items }: { items: AnsweredQuestion[] }) {
  const rows = items.flatMap((item) =>
    (item.payload.questions ?? []).map((spec) => {
      const raw = item.answers[spec.header];
      const values = (Array.isArray(raw) ? raw : typeof raw === "string" ? [raw] : []).filter(Boolean);
      return { key: `${item.id}-${spec.header}`, question: spec.question, values };
    }),
  );
  if (rows.length === 0) return null;
  return (
    <ul aria-label="Your answers" className="grid gap-2">
      {rows.map((r) => {
        const skipped = r.values.length === 0 || r.values.includes(SKIPPED);
        return (
          <li key={r.key} className="grid gap-1.5 rounded-xl border border-sand-2/20 bg-night-1/60 px-4 py-3">
            <p className="text-caption text-sand-1 text-balance">{r.question}</p>
            <p className="flex items-start gap-2 text-base text-sand-0 [overflow-wrap:anywhere]">
              {skipped ? (
                <SkipForward aria-hidden="true" className="mt-1 size-4 shrink-0 text-sand-2" />
              ) : (
                <Check aria-hidden="true" className="mt-1 size-4 shrink-0 text-ok" />
              )}
              <span>{skipped ? "Skipped. Prophet will use its best judgement." : r.values.join(", ")}</span>
            </p>
          </li>
        );
      })}
    </ul>
  );
}

function CheckpointCard({ mission, onChanged }: { mission: Mission; onChanged: () => void }) {
  const [busy, setBusy] = useState<"continue" | "answer" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const which = [...mission.events].reverse().find((e) => e.kind === "mission.checkpoint")?.payload.which;
  const cause =
    which === "max_seconds" ? "This part of the work used its time allowance." : "This part of the work used its tool-call allowance.";

  async function run(kind: "continue" | "answer") {
    setBusy(kind);
    setError(null);
    try {
      if (kind === "continue") await api.continueMission(mission.id);
      else await api.answerNow(mission.id);
      onChanged();
    } catch (err) {
      setError(errorMessage(err, "Couldn't continue. Try again."));
    } finally {
      setBusy(null);
    }
  }

  return (
    <Card role="status" className="grid gap-3 border-ember/40 p-4 sm:p-5">
      <p className="text-base font-bold text-sand-0">Paused at a checkpoint</p>
      <p className="text-caption text-sand-1">
        {cause} Every step so far is kept. Continue to let Prophet keep going, or answer now with what it has found.
      </p>
      {error && (
        <p role="alert" className="text-caption text-bad">
          {error}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <Button onClick={() => void run("continue")} disabled={busy !== null}>
          {busy === "continue" ? <Spinner /> : <Play aria-hidden="true" className="size-4" />}
          Continue
        </Button>
        <Button variant="secondary" onClick={() => void run("answer")} disabled={busy !== null}>
          {busy === "answer" ? <Spinner /> : <CheckCheck aria-hidden="true" className="size-4" />}
          Answer now
        </Button>
      </div>
    </Card>
  );
}

export function Turn({
  mission,
  tools,
  runs,
  onAnswered,
  onRetry,
  onEdit,
}: {
  mission: Mission;
  tools: Record<string, ToolInfo>;
  runs: ArtifactRun[];
  onAnswered: () => void;
  // A retry or an edit keeps the files the forecast was made from, so they are not silently dropped.
  onRetry: (goal: string, datasetIds: string[]) => void;
  onEdit: (goal: string, datasets: DatasetRef[]) => void;
}) {
  const working = isWorking(mission.status);
  const steps = buildSteps(mission, tools);
  const datasets = mission.datasets ?? [];
  const startedAt = mission.events[0]?.ts ?? Date.now() / 1000;
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    if (!working) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [working]);

  const elapsed = now - startedAt;
  const answer = mission.answer && typeof mission.answer === "object" ? (mission.answer as Record<string, any>) : null;
  const text = (answer?.status === "met" || answer?.status === "partial") && typeof answer.answer === "string" ? answer.answer : "";
  // The named reasons come from the API (outcome). The planner's own text is never shown as a reason.
  const reasons = mission.outcome?.reasons ?? [];
  const problem = problemFor(mission);

  // One quiet announcer for the two moments a screen reader user needs to hear: a question for them, or a finished answer.
  // The first render is skipped, so opening an old question does not announce it again.
  const [announce, setAnnounce] = useState("");
  const seen = useRef<{ status: string; answered: boolean } | null>(null);
  useEffect(() => {
    const prev = seen.current;
    seen.current = { status: mission.status, answered: text !== "" };
    if (!prev) return;
    if (mission.status === "waiting_user" && prev.status !== "waiting_user") setAnnounce("Prophet needs your answer.");
    else if (isCheckpoint(mission.status) && !isCheckpoint(prev.status)) setAnnounce("Paused at a checkpoint. Continue or answer now.");
    else if (text !== "" && !prev.answered) setAnnounce("Answer ready.");
  }, [mission.status, text]);

  return (
    <article className="grid gap-5" aria-label="Question and answer">
      <div className="grid justify-items-end gap-2">
        {datasets.length > 0 && (
          <ul aria-label="Attached files" className="flex max-w-[88%] flex-wrap justify-end gap-2 sm:max-w-[80%]">
            {datasets.map((d) => (
              <li
                key={d.id}
                className="flex min-w-0 max-w-full items-center gap-2.5 rounded-2xl border border-sand-2/30 bg-night-1 py-2 pr-3.5 pl-2.5"
              >
                <span aria-hidden="true" className="grid size-9 shrink-0 place-items-center rounded-lg bg-night-2 text-sand-1">
                  <FileText className="size-4" />
                </span>
                <span className="grid min-w-0 gap-0.5">
                  <span className="truncate text-caption text-sand-0" title={d.name}>
                    {d.name}
                  </span>
                  <span className="text-micro text-sand-2 tabular-nums">
                    {fileKind(d.name)} · {formatBytes(d.bytes)}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        )}
        <div className="max-w-[88%] rounded-3xl bg-night-2 px-5 py-3 text-sand-0 [overflow-wrap:anywhere] whitespace-pre-wrap sm:max-w-[80%]">
          <MentionText text={mission.goal} />
        </div>
      </div>

      <div className="grid gap-4">
        <StepsPanel steps={steps} working={working} />

        {isCheckpoint(mission.status) && <CheckpointCard mission={mission} onChanged={onAnswered} />}

        {working && (
          <div className="flex items-center gap-2.5 text-base text-sand-0">
            <Spinner className="text-ember" />
            <span role="status">{currentActivity(mission, tools)}</span>
            <span className="text-sand-2 tabular-nums">{formatDuration(elapsed)}</span>
          </div>
        )}

        {working && (
          <p className="text-caption text-sand-2 tabular-nums">
            {mission.progress && mission.progress.checklist_total > 0 && (
              <>
                {mission.progress.checklist_done} of {mission.progress.checklist_total} checks done
                {mission.progress.next ? `. Next: ${mission.progress.next}` : ""}.{" "}
              </>
            )}
            Measured runs took a median of about 9 minutes, and up to about 29 minutes.
          </p>
        )}

        <p className="sr-only" aria-live="polite">
          {announce}
        </p>

        <Decisions items={mission.answered_questions ?? []} />

        {mission.questions.map((q) => (
          <QuestionCard key={q.id} question={q} onAnswered={onAnswered} />
        ))}

        {answer?.status === "partial" && (
          <Card role="status" className="grid gap-2 border-ember/40 p-4">
            <p className="text-base font-bold text-sand-0">This is a partial answer.</p>
            <ul aria-label="Why this answer is partial" className="grid gap-2.5">
              {reasons.map((r) => (
                <li key={r.code} className="grid gap-1">
                  <p className="text-caption text-sand-1">{r.user_reason}</p>
                  {r.checks && r.checks.length > 0 && (
                    <ul className="grid gap-0.5 pl-4 text-caption text-sand-1 list-disc">
                      {r.checks.map((c) => (
                        <li key={c.name}>{c.plain}</li>
                      ))}
                    </ul>
                  )}
                  {r.figures && r.figures.length > 0 && (
                    <p className="text-caption text-sand-1 [overflow-wrap:anywhere]">
                      Could not be traced to the results: {r.figures.join(", ")}
                    </p>
                  )}
                  <p className="text-caption text-sand-0">Next: {r.next_step}</p>
                </li>
              ))}
            </ul>
          </Card>
        )}

        {text && <Markdown text={text} className="text-lead leading-[1.7]" />}

        <Results runs={runs} />


        {text && <AnswerActions text={text} seconds={Math.max(0, (mission.events[mission.events.length - 1]?.ts ?? startedAt) - startedAt)} />}

        {problem && (
          <Card role="alert" className={cn("grid gap-2 border-bad/40 p-4")}>
            <p className="flex items-start gap-2 text-base font-bold text-sand-0">
              <AlertTriangle aria-hidden="true" className="mt-1 size-4 shrink-0 text-bad" />
              {problem.what}
            </p>
            <p className="text-caption text-sand-1">{problem.why}</p>
            <p className="text-caption text-sand-0">{problem.next}</p>
            {problem.action !== "none" && (
              <div className="pt-1">
                {problem.action === "retry" ? (
                  <Button onClick={() => onRetry(mission.goal, datasets.map((d) => d.id))}>
                    <RotateCcw aria-hidden="true" className="size-4" />
                    Try again
                  </Button>
                ) : (
                  <Button onClick={() => onEdit(mission.goal, datasets)}>
                    <RotateCcw aria-hidden="true" className="size-4" />
                    Edit and ask again
                  </Button>
                )}
              </div>
            )}
          </Card>
        )}
      </div>
    </article>
  );
}
