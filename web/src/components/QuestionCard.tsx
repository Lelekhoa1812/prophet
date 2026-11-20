import { useState, type FormEvent } from "react";
import { api, type OpenQuestion } from "../lib/api";
import { errorMessage } from "../lib/format";
import { cn } from "../lib/utils";
import { Button, Card, Label } from "./ui";

type Answers = Record<string, string | string[]>;

// One structured question from the planner. Single-select uses native radios (arrow keys work, one tab stop per
// group); multi-select uses native checkboxes. Answers are keyed by each question's header, as the API expects.
export function QuestionCard({ question, onAnswered }: { question: OpenQuestion; onAnswered: () => void }) {
  const specs = question.payload.questions ?? [];
  const [answers, setAnswers] = useState<Answers>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const complete = specs.every((s) => {
    const a = answers[s.header];
    return s.multiSelect ? Array.isArray(a) && a.length > 0 : typeof a === "string" && a.length > 0;
  });

  const pickSingle = (header: string, label: string) => setAnswers((p) => ({ ...p, [header]: label }));

  const toggleMulti = (header: string, label: string) =>
    setAnswers((p) => {
      const cur = Array.isArray(p[header]) ? (p[header] as string[]) : [];
      return { ...p, [header]: cur.includes(label) ? cur.filter((x) => x !== label) : [...cur, label] };
    });

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!complete || busy) return;
    setBusy(true);
    setError(null);
    try {
      await api.answer(question.id, answers);
      onAnswered();
    } catch (err) {
      setError(errorMessage(err, "Could not send your answer. Try again."));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="border-ember/50 p-4 sm:p-5">
      <form onSubmit={submit} className="grid gap-6">
        <div className="grid gap-1">
          <Label>Prophet needs your input</Label>
          <p className="font-display text-xl text-sand-0">Answer to continue the run</p>
        </div>

        {specs.map((spec, si) => (
          <fieldset key={`${spec.header}-${si}`} className="grid min-w-0 gap-3">
            <legend className="mb-1 flex flex-wrap items-center gap-2 text-sand-0">
              <span className="rounded-sm bg-ember-soft px-1.5 py-0.5 text-xs text-ember">{spec.header}</span>
              <span className="text-balance">{spec.question}</span>
            </legend>
            <div className="grid gap-2">
              {spec.options.map((opt, oi) => {
                const checked = spec.multiSelect
                  ? Array.isArray(answers[spec.header]) && (answers[spec.header] as string[]).includes(opt.label)
                  : answers[spec.header] === opt.label;
                return (
                  <label
                    key={`${si}-${oi}`}
                    className={cn(
                      "group flex min-h-11 cursor-pointer gap-3 rounded-md border px-3 py-2.5 transition-colors",
                      "border-sand-2/50 bg-night-0/60 hover:border-sand-1",
                      "has-checked:border-ember has-checked:bg-ember-soft",
                      "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-ember",
                    )}
                  >
                    <input
                      type={spec.multiSelect ? "checkbox" : "radio"}
                      name={`${question.id}-${si}`}
                      value={opt.label}
                      checked={checked}
                      onChange={() =>
                        spec.multiSelect ? toggleMulti(spec.header, opt.label) : pickSingle(spec.header, opt.label)
                      }
                      className="sr-only"
                    />
                    <span
                      aria-hidden="true"
                      className={cn(
                        "mt-0.5 grid size-4 shrink-0 place-items-center border border-sand-2",
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
                    <span className="grid gap-0.5">
                      <span className="text-sand-0">{opt.label}</span>
                      {opt.description && <span className="text-sm text-sand-1">{opt.description}</span>}
                    </span>
                  </label>
                );
              })}
            </div>
          </fieldset>
        ))}

        {error && (
          <p role="alert" className="text-sm text-bad">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-sand-2">{complete ? "Ready to send" : "Choose an option for each question"}</p>
          <Button type="submit" disabled={!complete || busy}>
            {busy ? "Sending…" : "Send answer"}
          </Button>
        </div>
      </form>
    </Card>
  );
}
