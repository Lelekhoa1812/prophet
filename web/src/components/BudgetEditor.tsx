import { useId, useState, type FormEvent } from "react";
import type { ModelWindow, WindowValues } from "../lib/api";
import { errorMessage } from "../lib/format";
import { WINDOWS } from "../lib/usage";
import { Button, Spinner } from "./ui";

type Budgets = WindowValues<number | null>;

// "1200000" and "1,200,000" both work. Empty means no limit. Zero, decimals and words are invalid (undefined).
function parseTokens(text: string): number | null | undefined {
  const digits = text.replace(/[,\s]/g, "");
  if (digits === "") return null;
  const n = Number(digits);
  return Number.isInteger(n) && n > 0 ? n : undefined;
}

function toInput(n: number | null): string {
  return n === null ? "" : n.toLocaleString("en-AU");
}

// The token budget editor for one model, for the Settings screen. The caller supplies the save, so this component
// makes no request of its own. Give it a key per model so a switch starts from that model's saved values.
export function BudgetEditor({
  modelLabel,
  budget,
  onSave,
}: {
  modelLabel: string;
  budget: Budgets;
  onSave: (budget: Budgets) => Promise<void>;
}) {
  const idBase = useId();
  const [text, setText] = useState<Record<ModelWindow, string>>(() => ({
    "1d": toInput(budget["1d"]),
    "7d": toInput(budget["7d"]),
    "30d": toInput(budget["30d"]),
  }));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const parsed = {
    "1d": parseTokens(text["1d"]),
    "7d": parseTokens(text["7d"]),
    "30d": parseTokens(text["30d"]),
  };
  const invalid = WINDOWS.some((w) => parsed[w.key] === undefined);
  const dirty = WINDOWS.some((w) => parsed[w.key] !== budget[w.key]);

  function edit(key: ModelWindow, value: string) {
    setText((t) => ({ ...t, [key]: value }));
    setSaved(false);
    setError(null);
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (invalid || !dirty || busy) return;
    const next: Budgets = { "1d": parsed["1d"] ?? null, "7d": parsed["7d"] ?? null, "30d": parsed["30d"] ?? null };
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await onSave(next);
      setSaved(true);
    } catch (err) {
      setError(errorMessage(err, "Couldn't save these budgets. Try again."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-4" aria-label={`Token budgets for ${modelLabel}`}>
      <div className="grid gap-1">
        <h3 className="text-base font-semibold text-sand-0">Token budgets</h3>
        <p className="text-caption text-sand-2">Limits for {modelLabel}. Leave a box empty for no limit.</p>
      </div>

      <ul className="grid gap-3">
        {WINDOWS.map((w) => {
          const inputId = `${idBase}-${w.key}`;
          return (
            <li key={w.key} className="grid gap-1.5 sm:grid-cols-[10rem_minmax(0,1fr)] sm:items-center sm:gap-4">
              <label htmlFor={inputId} className="text-caption text-sand-0">
                {w.title}
              </label>
              <div className="flex min-w-0 items-center gap-2">
                <input
                  id={inputId}
                  type="text"
                  inputMode="numeric"
                  autoComplete="off"
                  placeholder="No limit"
                  value={text[w.key]}
                  onChange={(e) => edit(w.key, e.target.value)}
                  aria-invalid={parsed[w.key] === undefined}
                  className="h-11 min-w-0 flex-1 rounded-lg border border-sand-2/40 bg-night-2 px-3 text-base tabular-nums text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
                />
                <span className="text-caption text-sand-2">tokens</span>
              </div>
            </li>
          );
        })}
      </ul>

      {invalid && !error && (
        <p className="text-caption text-bad">Enter a whole number above 0, or leave the box empty for no limit.</p>
      )}
      {error && (
        <p role="alert" className="text-caption text-bad">
          {error}
        </p>
      )}
      {saved && !error && (
        <p role="status" className="text-caption text-ok">
          Budgets saved.
        </p>
      )}

      <div>
        <Button type="submit" disabled={invalid || !dirty || busy}>
          {busy && <Spinner />}
          Save budgets
        </Button>
      </div>
    </form>
  );
}
