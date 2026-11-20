export type EventRow = { seq: number; kind: string; payload: Record<string, unknown>; ts: number };

export type StepRow = { idx: number; name: string; status: string; output: any };

export type QuestionSpec = {
  question: string;
  header: string;
  multiSelect: boolean;
  options: { label: string; description: string }[];
};

export type OpenQuestion = { id: string; tool_key: string; payload: { questions: QuestionSpec[] } };

export type Mission = {
  id: string;
  goal: string;
  status: string;
  budget: Record<string, unknown>;
  answer: any;
  events: EventRow[];
  steps: StepRow[];
  questions: OpenQuestion[];
};

export type MissionRow = { id: string; goal: string; status: string; created: number; updated: number };

export type ArtifactFile = { path: string; sha256: string; bytes: number };

export type ArtifactRun = {
  run_id: string;
  files: ArtifactFile[];
  forecast: {
    config: Record<string, unknown> | null;
    point: number[] | null;
    q10: number[] | null;
    q50: number[] | null;
    q90: number[] | null;
  } | null;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  // no-store: mission state must never come from a stale HTTP cache entry (a dev port reused by another app caused this).
  const res = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* not JSON: keep the status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => req<{ ok: boolean; nim_keys_configured: boolean }>("/api/health"),
  missions: () => req<MissionRow[]>("/api/missions"),
  mission: (id: string, after = 0) => req<Mission>(`/api/missions/${id}?after=${after}`),
  createMission: (goal: string) => req<{ id: string }>("/api/missions", { method: "POST", body: JSON.stringify({ goal }) }),
  answer: (questionId: string, answers: Record<string, string | string[]>) =>
    req<{ mission_id: string }>(`/api/questions/${questionId}/answer`, {
      method: "POST",
      body: JSON.stringify({ answers }),
    }),
  artifacts: (id: string) => req<ArtifactRun[]>(`/api/missions/${id}/artifacts`),
};

export function artifactUrl(runId: string, path: string): string {
  return `/api/artifacts/${runId}/${path.replace(/^out\//, "")}`;
}
