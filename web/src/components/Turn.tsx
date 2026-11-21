import { AlertTriangle, Check, ChevronRight, Copy, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ArtifactRun, Mission, ToolInfo } from "../lib/api";
import { formatDuration, truncate } from "../lib/format";
import { buildSteps, currentActivity, type StepView } from "../lib/steps";
import { cn, isWorking } from "../lib/utils";
import { Markdown } from "./Markdown";
import { QuestionCard } from "./QuestionCard";
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
          {step.note && <span className="text-[0.9375rem] text-sand-1 [overflow-wrap:anywhere]">{step.note}</span>}
          {step.error && step.state === "failed" && (
            <span className="text-[0.9375rem] text-bad [overflow-wrap:anywhere]">{truncate(step.error, 220)}</span>
          )}
          {step.followUp && <span className="text-[0.9375rem] text-sand-2">Not there yet: {truncate(step.followUp, 200)}</span>}
        </div>
      </div>
      {hasDetails && (
        <details className="ml-7">
          <summary className="inline-flex min-h-10 cursor-pointer items-center gap-1 text-[0.9375rem] text-sand-2 hover:text-sand-0 sm:min-h-0">
            <ChevronRight aria-hidden="true" className="size-3.5 transition-transform [details[open]>summary_&]:rotate-90" />
            Details
          </summary>
          <div className="mt-2 grid gap-3">
            {step.facts.length > 0 && (
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[0.9375rem]">
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
                <span className="text-[0.9375rem] text-sand-2">Code that ran</span>
                <pre className="max-h-72 overflow-auto rounded-lg bg-night-0 p-3 text-[0.9375rem] leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-0">
                  {step.code}
                </pre>
              </div>
            )}
            {step.output && (
              <div className="grid gap-1">
                <span className="text-[0.9375rem] text-sand-2">What it printed</span>
                <pre className="max-h-60 overflow-auto rounded-lg bg-night-0 p-3 text-[0.9375rem] leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-1">
                  {truncate(step.output, 4000)}
                </pre>
              </div>
            )}
            <details>
              <summary className="inline-flex min-h-10 cursor-pointer items-center text-[0.9375rem] text-sand-2 hover:text-sand-0 sm:min-h-0">Full record</summary>
              <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-night-0 p-3 font-mono text-[0.875rem] leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-1">
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
      <summary className="inline-flex min-h-10 cursor-pointer items-center gap-1.5 rounded-lg pr-3 text-[0.9375rem] text-sand-2 transition-colors hover:text-sand-0">
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
    <div className="-mt-1 flex items-center gap-1 text-[0.9375rem] text-sand-2">
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
    case "iteration_budget":
    case "budget_exhausted":
      return {
        what: "Prophet reached its working limit before it was sure of the answer.",
        why: "Each request has a fixed amount of work so it cannot run forever.",
        next: "Ask again with a narrower question, or break the work into two requests.",
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
  onRetry: (goal: string) => void;
  onEdit: (goal: string) => void;
}) {
  const working = isWorking(mission.status);
  const steps = buildSteps(mission, tools);
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
  const unverified: string[] = Array.isArray(answer?.unverified) ? answer.unverified.map(String) : [];
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
    else if (text !== "" && !prev.answered) setAnnounce("Answer ready.");
  }, [mission.status, text]);

  return (
    <article className="grid gap-5" aria-label="Question and answer">
      <div className="justify-self-end max-w-[88%] rounded-3xl bg-night-2 px-5 py-3 text-sand-0 [overflow-wrap:anywhere] whitespace-pre-wrap sm:max-w-[80%]">
        {mission.goal}
      </div>

      <div className="grid gap-4">
        <StepsPanel steps={steps} working={working} />

        {working && (
          <div className="flex items-center gap-2.5 text-base text-sand-0">
            <Spinner className="text-ember" />
            <span role="status">{currentActivity(mission, tools)}</span>
            <span className="text-sand-2 tabular-nums">{formatDuration(elapsed)}</span>
            {elapsed > 60 && <span className="hidden text-sand-2 sm:inline">Longer requests can take a few minutes.</span>}
          </div>
        )}

        <p className="sr-only" aria-live="polite">
          {announce}
        </p>

        {mission.questions.map((q) => (
          <QuestionCard key={q.id} question={q} onAnswered={onAnswered} />
        ))}

        {answer?.status === "partial" && (
          <Card role="status" className="grid gap-1.5 border-ember/40 p-4">
            <p className="text-base font-bold text-sand-0">This is a partial answer.</p>
            {typeof answer.reason === "string" && answer.reason && <p className="text-[0.9375rem] text-sand-1">{answer.reason}</p>}
            {unverified.length > 0 && (
              <p className="text-[0.9375rem] text-sand-1">Not found in the results: {unverified.join(", ")}</p>
            )}
          </Card>
        )}

        {text && <Markdown text={text} className="text-[1.125rem] leading-[1.7]" />}

        <Results runs={runs} />

        {text && <AnswerActions text={text} seconds={Math.max(0, (mission.events[mission.events.length - 1]?.ts ?? startedAt) - startedAt)} />}

        {problem && (
          <Card role="alert" className={cn("grid gap-2 border-bad/40 p-4")}>
            <p className="flex items-start gap-2 text-base font-bold text-sand-0">
              <AlertTriangle aria-hidden="true" className="mt-1 size-4 shrink-0 text-bad" />
              {problem.what}
            </p>
            <p className="text-[0.9375rem] text-sand-1">{problem.why}</p>
            <p className="text-[0.9375rem] text-sand-0">{problem.next}</p>
            {problem.action !== "none" && (
              <div className="pt-1">
                {problem.action === "retry" ? (
                  <Button onClick={() => onRetry(mission.goal)}>
                    <RotateCcw aria-hidden="true" className="size-4" />
                    Try again
                  </Button>
                ) : (
                  <Button onClick={() => onEdit(mission.goal)}>
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
