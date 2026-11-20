import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

const STATUS: Record<string, { label: string; tone: "ok" | "bad" | "ember" | "sand" }> = {
  created: { label: "Queued", tone: "sand" },
  running: { label: "Working", tone: "ember" },
  waiting_user: { label: "Needs your answer", tone: "ember" },
  completed: { label: "Done", tone: "ok" },
  failed: { label: "Failed", tone: "bad" },
  outcome_unknown: { label: "Outcome unknown", tone: "bad" },
  budget_exhausted: { label: "Budget reached", tone: "bad" },
  met: { label: "Done", tone: "ok" },
  stalled: { label: "Stalled: no progress", tone: "bad" },
  impossible: { label: "Not achievable with these tools", tone: "bad" },
  iteration_budget: { label: "Iteration budget reached", tone: "bad" },
};

export function statusInfo(status: string) {
  return STATUS[status] ?? { label: status, tone: "sand" as const };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}
