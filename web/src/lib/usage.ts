import type { ModelWindow, UsageModel } from "./api";

// The three budget windows, in the order the ring prefers them: the shortest window with a budget is the one shown.
export const WINDOWS: { key: ModelWindow; title: string; adjective: string }[] = [
  { key: "1d", title: "Last 24 hours", adjective: "24-hour" },
  { key: "7d", title: "Last 7 days", adjective: "7-day" },
  { key: "30d", title: "Last 30 days", adjective: "30-day" },
];

export type RingReading = { title: string; adjective: string; used: number; budget: number; pct: number };

// The ring reads the 24-hour window when it has a budget, else 7 days, else 30 days. Null when no window has a budget.
export function ringReading(model: UsageModel): RingReading | null {
  for (const w of WINDOWS) {
    const budget = model.budget[w.key];
    if (typeof budget === "number") {
      const used = model.used[w.key];
      return { title: w.title, adjective: w.adjective, used, budget, pct: percentUsed(used, budget) };
    }
  }
  return null;
}

// Not clamped: a model over budget reads above 100 so the label stays truthful. A zero budget is exhausted once any token is used.
export function percentUsed(used: number, budget: number): number {
  if (budget <= 0) return used > 0 ? 100 : 0;
  return (used / budget) * 100;
}

const COMPACT = new Intl.NumberFormat("en-AU", { notation: "compact", maximumFractionDigits: 1 });
const FULL = new Intl.NumberFormat("en-AU");

// Small counts keep their thousands separators; large ones go compact (1.2M) so the row stays on one line on a phone.
export function formatTokens(n: number): string {
  return n >= 100_000 ? COMPACT.format(n) : FULL.format(n);
}

const RESET = new Intl.DateTimeFormat("en-AU", { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });

export function formatReset(iso: string): string | null {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? null : RESET.format(t);
}
