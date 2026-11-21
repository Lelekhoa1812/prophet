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

export type MissionRow = { id: string; goal: string; title: string; status: string; created: number; updated: number; pinned: boolean };

export type ToolInfo = { name: string; title: string; activity: string; description: string };

export type ArtifactFile = { path: string; sha256: string; bytes: number };

export type ArtifactRun = {
  run_id: string;
  files: ArtifactFile[];
  history?: number[];
  forecast: {
    config: Record<string, unknown> | null;
    point: number[] | null;
    q10: number[] | null;
    q50: number[] | null;
    q90: number[] | null;
  } | null;
};

export type SourceStatus = "profiling" | "ready" | "needs_attention" | "failed";

export type SheetCard = {
  sheet: string | null;
  rows: number;
  header_row?: number | null;
  columns: string[];
  columns_omitted: number;
  issues: string[];
  layout_hints: string[];
};

export type SourceCard = {
  sheets: SheetCard[];
  hidden_sheets?: string[];
  sheets_not_profiled?: string[];
} | null;

export type Source = {
  id: string;
  notebook_id: string;
  dataset_id: string;
  name: string;
  sha256: string;
  bytes: number;
  status: SourceStatus;
  card: SourceCard;
  error: string | null;
  included: boolean;
  removed: boolean;
  created: number;
  duplicate?: boolean;
};

export type KeptTable = {
  id: string;
  name: string;
  description: string;
  version: number;
  rows: number;
  columns: string[];
  sha256: string;
  created: number;
  turn_id: string | null;
};

export type TurnRow = {
  id: string;
  seq: number;
  goal: string;
  status: string;
  created: number;
  updated: number;
  answer: any;
};

export type Notebook = {
  id: string;
  title: string;
  instructions: string;
  archived: boolean;
  created: number;
  updated: number;
  sources: Source[];
  tables: KeptTable[];
  turns: TurnRow[];
};

export type NotebookRow = {
  id: string;
  title: string;
  pinned: boolean;
  created: number;
  updated: number;
  turns: number;
  sources: number;
  status: string | null;
};

async function send<T>(path: string, init: RequestInit): Promise<T> {
  // no-store: mission state must never come from a stale HTTP cache entry (a dev port reused by another app caused this).
  const res = await fetch(path, { ...init, cache: "no-store" });
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

function req<T>(path: string, init?: RequestInit): Promise<T> {
  return send<T>(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
}

export const api = {
  health: () => req<{ ok: boolean; nim_keys_configured: boolean }>("/api/health"),
  tools: () => req<ToolInfo[]>("/api/tools"),
  missions: () => req<MissionRow[]>("/api/missions"),
  archiveMission: (id: string) => req<{ id: string }>(`/api/missions/${id}`, { method: "PATCH", body: JSON.stringify({ archived: true }) }),
  unarchiveMission: (id: string) => req<{ id: string }>(`/api/missions/${id}`, { method: "PATCH", body: JSON.stringify({ archived: false }) }),
  mission: (id: string, after = 0) => req<Mission>(`/api/missions/${id}?after=${after}`),
  createMission: (goal: string, datasetIds: string[] = []) =>
    req<{ id: string }>("/api/missions", { method: "POST", body: JSON.stringify({ goal, dataset_ids: datasetIds }) }),
  renameMission: (id: string, title: string) => req<{ id: string }>(`/api/missions/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  pinMission: (id: string, pinned: boolean) => req<{ id: string }>(`/api/missions/${id}`, { method: "PATCH", body: JSON.stringify({ pinned }) }),
  deleteMission: (id: string) => req<{ id: string }>(`/api/missions/${id}`, { method: "DELETE" }),
  // Uploads a quick-chat file as a dataset. The name travels in a header, so it never appears in a URL.
  uploadDataset: (file: File) =>
    send<{ id: string; name: string; bytes: number }>("/api/datasets", {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
    }),
  answer: (questionId: string, answers: Record<string, string | string[]>) =>
    req<{ mission_id: string }>(`/api/questions/${questionId}/answer`, {
      method: "POST",
      body: JSON.stringify({ answers }),
    }),
  artifacts: (id: string) => req<ArtifactRun[]>(`/api/missions/${id}/artifacts`),
  cancel: (missionId: string) => req<{ status: string }>(`/api/missions/${missionId}/cancel`, { method: "POST" }),
  // A checkpoint is a pause, not a failure: continue gives the work another part, answer-now writes what is established.
  continueMission: (missionId: string) => req<{ id: string; status: string }>(`/api/missions/${missionId}/continue`, { method: "POST" }),
  answerNow: (missionId: string) => req<{ id: string; status: string }>(`/api/missions/${missionId}/answer-now`, { method: "POST" }),

  settings: () => req<SettingsView>("/api/settings"),
  saveLimits: (patch: Partial<Record<SettingsSection, Record<string, number>>>) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),
  saveKeys: (keys: string[]) => req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify({ nim: { keys } }) }),
  testKeys: () => req<{ keys: KeyProbe[] }>("/api/settings/test-keys", { method: "POST" }),
  clearData: (confirm: string) => req<{ removed: Record<string, number> }>("/api/settings/clear-data", { method: "POST", body: JSON.stringify({ confirm }) }),

  notebooks: () => req<NotebookRow[]>("/api/notebooks"),
  createNotebook: (title: string) =>
    req<Notebook>("/api/notebooks", { method: "POST", body: JSON.stringify({ title }) }),
  notebook: (id: string) => req<Notebook>(`/api/notebooks/${id}`),
  // The raw file goes in the body and its name in a header, so a file name never appears in a URL.
  uploadSource: (notebookId: string, file: File) =>
    send<Source>(`/api/notebooks/${notebookId}/sources`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream", "X-File-Name": encodeURIComponent(file.name) },
    }),
  patchNotebook: (nid: string, body: { title?: string; pinned?: boolean; archived?: boolean }) =>
    req<NotebookRow>(`/api/notebooks/${nid}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteNotebook: (nid: string) => req<{ id: string }>(`/api/notebooks/${nid}`, { method: "DELETE" }),
  patchSource: (notebookId: string, sourceId: string, body: { included?: boolean; removed?: boolean }) =>
    req<Source>(`/api/notebooks/${notebookId}/sources/${sourceId}`, { method: "PATCH", body: JSON.stringify(body) }),
  createTurn: (notebookId: string, goal: string) =>
    req<{ id: string; seq: number }>(`/api/notebooks/${notebookId}/turns`, {
      method: "POST",
      body: JSON.stringify({ goal }),
    }),
};

export type SettingsSection = "agent" | "context";

export type KeyProbe = {
  slot: number;
  model: string;
  status: number | null;
  error: string | null;
  elapsed_s: number;
  masked: string;
  ok: boolean;
};

export type PoolKey = {
  slot: number;
  in_flight: number;
  sent_last_minute: number;
  per_minute: number;
  retired: boolean;
  retired_for_s: number;
  cooling: { model: string; for_s: number }[];
};

export type SettingsView = {
  nim: {
    configured: boolean;
    keys: { slot: number; masked: string }[];
    models: string[];
    pool: { keys: PoolKey[]; demoted_models: { model: string; for_s: number }[]; in_flight: number } | null;
  };
  agent: Record<string, number>;
  context: Record<string, number>;
  defaults: Record<SettingsSection, Record<string, number>>;
  bounds: Record<SettingsSection, Record<string, [number, number]>>;
  timesfm: { checkpoint: string; engine_installed: boolean; licence: string };
  sandbox: {
    timeout_s: number;
    mem_mb: number;
    file_mb: number;
    memory_limit_enforced: boolean;
    isolation: string;
  };
  data: { missions: number; notebooks: number; datasets: number; bytes: number };
  about: { name: string; licence: string; clear_phrase: string };
};

export function artifactUrl(runId: string, path: string): string {
  return `/api/artifacts/${runId}/${path.replace(/^out\//, "")}`;
}

export function tableDownloadUrl(notebookId: string, name: string, version?: number): string {
  const q = version ? `?version=${version}` : "";
  return `/api/notebooks/${notebookId}/tables/${encodeURIComponent(name)}/download${q}`;
}
