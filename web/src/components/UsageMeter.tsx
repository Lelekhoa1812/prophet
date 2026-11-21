import { AlertCircle, X } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import type { UsageModel } from "../lib/api";
import { useDismiss } from "../lib/useDismiss";
import { useUsage } from "../lib/useUsage";
import { WINDOWS, formatReset, formatTokens, percentUsed, ringReading } from "../lib/usage";
import { cn } from "../lib/utils";
import { Button } from "./ui";

// The ring. An empty track means no budget; the arc is the share of the budget used, and turns red at or over 100 percent.
function Ring({ pct, loading }: { pct: number | null; loading: boolean }) {
  const over = pct !== null && pct >= 100;
  return (
    <svg viewBox="0 0 36 36" aria-hidden="true" className={cn("size-3.5 shrink-0 -rotate-90", loading && "animate-pulse")}>
      <circle cx="18" cy="18" r="14" pathLength={100} fill="none" strokeWidth="3.5" className="stroke-sand-2/30" />
      {pct !== null && pct > 0 && (
        <circle
          cx="18"
          cy="18"
          r="14"
          pathLength={100}
          fill="none"
          strokeWidth="3.5"
          strokeLinecap="round"
          strokeDasharray={`${Math.min(pct, 100)} 100`}
          className={over ? "stroke-bad" : "stroke-ember"}
        />
      )}
    </svg>
  );
}

function UsageRow({
  title,
  used,
  budget,
  resets,
}: {
  title: string;
  used: number;
  budget: number | null;
  resets: string;
}) {
  const pct = budget === null ? null : percentUsed(used, budget);
  const reset = formatReset(resets);
  return (
    <li className="grid gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-caption font-semibold text-sand-0">{title}</span>
        <span className="text-caption tabular-nums text-sand-1">{pct === null ? "No limit" : `${Math.round(pct)}%`}</span>
      </div>
      {budget !== null && pct !== null && (
        <div
          role="progressbar"
          aria-label={`${title} token budget`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.min(Math.round(pct), 100)}
          aria-valuetext={`${Math.round(pct)} percent used`}
          className="h-2 overflow-hidden rounded-full bg-night-3"
        >
          <div
            className={cn("h-full rounded-full transition-[width] duration-300 motion-reduce:transition-none", pct >= 100 ? "bg-bad" : "bg-ember")}
            style={{ width: `${Math.min(pct, 100)}%` }}
          />
        </div>
      )}
      <p className="flex flex-wrap justify-between gap-x-3 gap-y-0.5 text-caption text-sand-2">
        <span className="tabular-nums text-sand-1">
          {budget === null ? `${formatTokens(used)} tokens used` : `${formatTokens(used)} of ${formatTokens(budget)} tokens`}
        </span>
        {reset && <span>Resets {reset}</span>}
      </p>
    </li>
  );
}

// A usage ring for the selected model, beside the model switcher. Opens a "Token budgets" popover with each window.
export function UsageMeter({
  modelId,
  modelLabel,
  onManageBudgets,
}: {
  modelId: string | null;
  modelLabel: string | null;
  onManageBudgets: () => void;
}) {
  const [open, setOpen] = useState(false);
  const usage = useUsage(modelId !== null);
  const { refresh } = usage;
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const titleId = useId();

  const closePopover = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus();
  };
  useDismiss(open, wrap, (reason) => closePopover(reason === "escape"));

  // Numbers in the popover are read fresh each time it opens. Focus moves into it so keyboard users land on the content.
  useEffect(() => {
    if (!open) return;
    refresh();
    dialog.current?.focus();
  }, [open, refresh]);

  if (modelId === null) return null;

  const model: UsageModel | null = usage.data?.models.find((m) => m.id === modelId) ?? null;
  const reading = model ? ringReading(model) : null;
  const failed = usage.data === null && usage.status === "error";
  const loading = usage.data === null && !failed;

  const label = failed
    ? "Usage unavailable. Open to try again."
    : reading
      ? `Token budget, ${reading.adjective} window ${Math.round(reading.pct)} percent used`
      : loading
        ? "Token budget, loading"
        : "Token budget, no budget set";
  // The pill is icon-only. Its state is in the aria-label and the popover; the words stay out of the composer.
  return (
    <div ref={wrap} className="relative min-w-0">
      <button
        ref={trigger}
        type="button"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={label}
        title="Token budgets"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "inline-flex min-h-10 min-w-10 items-center justify-center rounded-lg px-2 transition-colors hover:bg-night-2 hover:text-sand-0 sm:min-h-8 sm:min-w-8",
          open ? "bg-night-2 text-sand-0" : "text-sand-1",
        )}
      >
        {failed ? (
          <AlertCircle aria-hidden="true" className="size-3.5 shrink-0 text-bad" />
        ) : (
          <Ring pct={reading ? reading.pct : null} loading={loading} />
        )}
      </button>

      {open && (
        <div
          ref={dialog}
          role="dialog"
          aria-labelledby={titleId}
          tabIndex={-1}
          className="absolute right-0 bottom-full z-30 mb-2 grid w-80 gap-4 rounded-xl border border-sand-2/30 bg-night-2 p-4 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)] focus:outline-none max-sm:fixed max-sm:inset-x-0 max-sm:bottom-0 max-sm:mb-0 max-sm:w-auto max-sm:max-h-[80dvh] max-sm:overflow-y-auto max-sm:rounded-b-none max-sm:pb-[max(1rem,env(safe-area-inset-bottom))]"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="grid min-w-0 gap-0.5">
              <h2 id={titleId} className="text-base font-semibold text-sand-0">
                Token budgets
              </h2>
              {modelLabel && <p className="truncate text-caption text-sand-2">{modelLabel}</p>}
            </div>
            <button
              type="button"
              aria-label="Close token budgets"
              onClick={() => closePopover(true)}
              className="grid size-11 shrink-0 place-items-center rounded-lg text-sand-2 transition-colors hover:bg-night-3 hover:text-sand-0 sm:size-9"
            >
              <X aria-hidden="true" className="size-4" />
            </button>
          </div>

          {usage.data === null ? (
            failed ? (
              <div className="grid gap-2">
                <p className="text-caption text-sand-1">Usage unavailable. Check the connection, then try again.</p>
                <div>
                  <Button variant="secondary" size="sm" onClick={() => refresh()}>
                    Try again
                  </Button>
                </div>
              </div>
            ) : (
              <p role="status" className="text-caption text-sand-2">
                Loading usage
              </p>
            )
          ) : model === null ? (
            <p className="text-caption text-sand-1">No usage recorded for this model yet.</p>
          ) : (
            <ul className="grid gap-4" aria-label="Usage by window">
              {WINDOWS.map((w) => (
                <UsageRow
                  key={w.key}
                  title={w.title}
                  used={model.used[w.key]}
                  budget={model.budget[w.key]}
                  resets={model.resets[w.key]}
                />
              ))}
            </ul>
          )}

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-sand-2/20 pt-3">
            <p className="text-micro text-sand-2">Prompt and completion tokens, summed.</p>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                closePopover(false);
                onManageBudgets();
              }}
            >
              Manage budgets
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
