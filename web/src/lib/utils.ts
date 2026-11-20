import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export type Tone = "ok" | "bad" | "ember" | "sand";

// What the person sees for each state. Plain words, no internal status names.
const STATUS: Record<string, { label: string; tone: Tone }> = {
  created: { label: "Starting", tone: "ember" },
  running: { label: "Working", tone: "ember" },
  waiting_user: { label: "Needs your answer", tone: "ember" },
  completed: { label: "Done", tone: "ok" },
  met: { label: "Done", tone: "ok" },
  failed: { label: "Couldn't finish", tone: "bad" },
  outcome_unknown: { label: "Paused for review", tone: "bad" },
  budget_exhausted: { label: "Reached its limit", tone: "bad" },
  iteration_budget: { label: "Reached its limit", tone: "bad" },
  stalled: { label: "Stuck", tone: "bad" },
  impossible: { label: "Not possible with this data", tone: "bad" },
  cancelled: { label: "Stopped", tone: "sand" },
};

export function statusInfo(status: string) {
  return STATUS[status] ?? { label: "Unknown", tone: "sand" as Tone };
}

export function isWorking(status: string): boolean {
  return status === "created" || status === "running";
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}
