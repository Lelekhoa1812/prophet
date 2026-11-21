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
  // A checkpoint is a planned pause: the work so far is kept and the person decides whether to carry on.
  checkpoint: { label: "Paused at a checkpoint", tone: "ember" },
  // Runs from before checkpoints ended at a limit. They are still shown, with the same wording as the checkpoint.
  budget_exhausted: { label: "Paused at a checkpoint", tone: "ember" },
  iteration_budget: { label: "Paused at a checkpoint", tone: "ember" },
  partial: { label: "Partly answered", tone: "ember" },
  stalled: { label: "Stuck", tone: "bad" },
  impossible: { label: "Not possible with this data", tone: "bad" },
  cancelled: { label: "Stopped", tone: "sand" },
  // The app was closed or restarted mid-run. Finished steps are kept; the mission can be resumed from where it stopped.
  interrupted: { label: "Interrupted", tone: "sand" },
  lease_lost: { label: "Moved to another run", tone: "sand" },
};

export function statusInfo(status: string) {
  return STATUS[status] ?? { label: "Unknown", tone: "sand" as Tone };
}

export function isWorking(status: string): boolean {
  return status === "created" || status === "running";
}

// Mirrors RESUMABLE_AT_CHECKPOINT in the API service: these can be continued or answered now.
export const RESUMABLE_STATUSES = ["checkpoint", "budget_exhausted", "iteration_budget"] as const;

export function isCheckpoint(status: string): boolean {
  return (RESUMABLE_STATUSES as readonly string[]).includes(status);
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}
