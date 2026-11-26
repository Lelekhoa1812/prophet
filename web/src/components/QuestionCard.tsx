import { MessageCircleQuestion, SkipForward } from "lucide-react";
import { useState, type FormEvent } from "react";
import { api, type OpenQuestion } from "../lib/api";
import { errorMessage } from "../lib/format";
import { cn } from "../lib/utils";
import { Button, Card, IconButton, Spinner } from "./ui";

type Answers = Record<string, string | string[]>;

const OTHER = "\u0000other";
export const SKIPPED = "Skipped: use your best judgement";

// Questions from the agent, as one card above the work. Single choice uses native radios (arrow keys, one tab stop),
// multiple choice uses native checkboxes. Every question can be skipped; the agent is told to use its own judgement.
export function QuestionCard({ question, onAnswered }: { question: OpenQuestion; onAnswered: () => void }) {
  const specs = question.payload.questions ?? [];
  const [answers, setAnswers] = useState<Answers>({});
  const [custom, setCustom] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const resolved = (header: string, multi: boolean): string | string[] | null => {
    const a = answers[header];
    if (multi) {
      if (!Array.isArray(a) || a.length === 0) return null;
      return a.map((x) => (x === OTHER ? (custom[header] ?? "").trim() : x)).filter(Boolean);
    }
    if (typeof a !== "string" || !a) return null;
    if (a === OTHER) return (custom[header] ?? "").trim() || null;
    return a;
  };

  const answered = specs.filter((s) => {
    const r = resolved(s.header, s.multiSelect);
    return Array.isArray(r) ? r.length > 0 : !!r;
  }).length;
  const complete = answered === specs.length && specs.length > 0;

  const setSingle = (header: string, label: string) => setAnswers((p) => ({ ...p, [header]: label }));
  // A real option replaces a skip, so the skip marker never travels with an answer.
  const toggleMulti = (header: string, label: string) =>
    setAnswers((p) => {
      const cur = Array.isArray(p[header]) ? (p[header] as string[]).filter((x) => x !== SKIPPED) : [];
      return { ...p, [header]: cur.includes(label) ? cur.filter((x) => x !== label) : [...cur, label] };
    });

  const submit = async (e?: FormEvent) => {
    e?.preventDefault();
    if (!complete || busy) return;
    setBusy(true);
    setError(null);
    try {
      const out: Answers = {};
      for (const s of specs) out[s.header] = resolved(s.header, s.multiSelect) as string | string[];
      await api.answer(question.id, out);
      onAnswered();
    } catch (err) {
      setError(errorMessage(err, "Couldn't send your answer. Try again."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="border-ember/40 p-4 sm:p-5" role="group" aria-label="Questions from Prophet">
      <form onSubmit={submit} className="grid gap-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="flex items-center gap-2 text-lg font-bold text-sand-0">
            <MessageCircleQuestion aria-hidden="true" className="size-5 text-ember" />
            {specs.length === 1 ? "A quick question" : `${specs.length} quick questions`}
          </h3>
          {specs.length > 1 && (
            <span className="text-caption text-sand-1 tabular-nums">
              {answered} of {specs.length} answered
            </span>
          )}
        </div>

        {specs.map((spec, si) => {
          const sel = answers[spec.header];
          const isOther = spec.multiSelect ? Array.isArray(sel) && sel.includes(OTHER) : sel === OTHER;
          return (
            <fieldset key={`${spec.header}-${si}`} className="grid min-w-0 gap-2.5">
              <legend className="mb-1 flex w-full flex-wrap items-baseline justify-between gap-2">
                <span className="text-base font-semibold text-sand-0 text-balance">{spec.question}</span>
                <IconButton
                  label={`Skip this question: ${spec.question}`}
                  icon={<SkipForward aria-hidden="true" className="size-4" />}
                  onClick={() => setAnswers((p) => ({ ...p, [spec.header]: spec.multiSelect ? [SKIPPED] : SKIPPED }))}
                  className="size-9"
                />
              </legend>
              <div className="grid gap-2">
                {[...spec.options.map((o) => ({ label: o.label, description: o.description })), { label: OTHER, description: "" }].map(
                  (opt, oi) => {
                    const isOtherOpt = opt.label === OTHER;
                    const checked = spec.multiSelect
                      ? Array.isArray(sel) && sel.includes(opt.label)
                      : sel === opt.label;
                    return (
                      <label
                        key={`${si}-${oi}`}
                        className={cn(
                          // relative: the radio is sr-only (absolute). Without a positioned label it sits at the corner of the
                          // page layout, and focusing it on click scrolls the whole app away from the card.
                          "group relative flex min-h-11 cursor-pointer items-start gap-3 rounded-lg border px-3 py-2.5 transition-colors",
                          "border-sand-2/30 bg-night-0/40 hover:border-sand-1",
                          "has-checked:border-ember has-checked:bg-ember-soft",
                          "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-ember",
                        )}
                      >
                        <input
                          type={spec.multiSelect ? "checkbox" : "radio"}
                          name={`${question.id}-${si}`}
                          checked={checked}
                          onChange={() =>
                            spec.multiSelect ? toggleMulti(spec.header, opt.label) : setSingle(spec.header, opt.label)
                          }
                          className="sr-only"
                        />
                        <span
                          aria-hidden="true"
                          className={cn(
                            "mt-1 grid size-4 shrink-0 place-items-center border border-sand-2",
                            spec.multiSelect ? "rounded-sm" : "rounded-full",
                          )}
                        >
                          <span
                            className={cn(
                              "size-2 bg-ember opacity-0 transition-opacity group-has-checked:opacity-100",
                              spec.multiSelect ? "rounded-[1px]" : "rounded-full",
                            )}
                          />
                        </span>
                        <span className="grid min-w-0 flex-1 gap-0.5">
                          <span className="text-base text-sand-0">{isOtherOpt ? "Something else" : opt.label}</span>
                          {opt.description && <span className="text-caption text-sand-1">{opt.description}</span>}
                          {isOtherOpt && isOther && (
                            <input
                              type="text"
                              autoFocus
                              value={custom[spec.header] ?? ""}
                              onChange={(e) => setCustom((p) => ({ ...p, [spec.header]: e.target.value }))}
                              aria-label={`Your answer to: ${spec.question}`}
                              placeholder="Type your answer"
                              className="mt-1.5 w-full rounded-md border border-sand-2/50 bg-night-0 px-2.5 py-1.5 text-base text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
                            />
                          )}
                        </span>
                      </label>
                    );
                  },
                )}
              </div>
              {(sel === SKIPPED || (Array.isArray(sel) && sel[0] === SKIPPED)) && (
                <p className="text-caption text-sand-1">Skipped. Prophet will use its best judgement.</p>
              )}
            </fieldset>
          );
        })}

        {error && (
          <p role="alert" className="text-base text-bad">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-caption text-sand-1">{complete ? "Ready to send" : "Answer or skip each question to continue"}</p>
          <Button type="submit" disabled={!complete || busy}>
            {busy && <Spinner />}
            {busy ? "Sending" : "Send answers"}
          </Button>
        </div>
      </form>
    </Card>
  );
}
