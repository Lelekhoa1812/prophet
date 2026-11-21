export type EventRow = { seq: number; kind: string; payload: Record<string, unknown>; ts: number };

export type StepRow = { idx: number; name: string; status: string; output: any };

export type QuestionSpec = {
  question: string;
  header: string;
  multiSelect: boolean;
  options: { label: string; description: string }[];
};

export type OpenQuestion = { id: string; tool_key: string; payload: { questions: QuestionSpec[] } };

// A question the person has answered (or skipped). answers is keyed by each question's header.
export type AnsweredQuestion = {
  id: string;
  payload: { questions: QuestionSpec[] };
  answers: Record<string, string | string[]>;
  answered: number | null;
};

// A file the person attached to a quick forecast.
export type DatasetRef = { id: string; name: string; bytes: number };

export type Mission = {
  id: string;
  goal: string;
  status: string;
  budget: Record<string, unknown>;
  answer: any;
  events: EventRow[];
  steps: StepRow[];
  questions: OpenQuestion[];
  answered_questions: AnsweredQuestion[];
  datasets: DatasetRef[];
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
  // The model id is sent only when one is selected, so the server default applies otherwise.
  createMission: (goal: string, datasetIds: string[] = [], model?: string) =>
    req<{ id: string }>("/api/missions", {
      method: "POST",
      body: JSON.stringify({ goal, dataset_ids: datasetIds, ...(model ? { model } : {}) }),
    }),
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
  // Token usage per model for the composer's meter. Counts are prompt plus completion tokens summed.
  usage: () => req<UsageView>("/api/usage"),
  // The models a person may choose from. Read from /api/settings and normalised, so a malformed entry is dropped.
  modelCatalog: async () => readCatalog(await req<unknown>("/api/settings")),
  saveLimits: (patch: Partial<Record<SettingsSection, Record<string, number>>>) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),
  // Keys: masked values echoed back keep the stored key. Only newly typed keys are sent in full.
  saveProviderKeys: (provider: string, keys: string[]) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify({ providers: { [provider]: { keys } } }) }),
  setProviderEnabled: (provider: string, enabled: boolean) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify({ llm: { providers: { [provider]: { enabled } } } }) }),
  setModelEnabled: (model: string, enabled: boolean) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify({ llm: { models: { [model]: { enabled } } } }) }),
  // Token budgets for one model: prompt plus completion tokens summed per window. Null means no limit.
  setModelBudget: (modelId: string, budget: LlmBudget) =>
    req<SettingsView>("/api/settings", { method: "PUT", body: JSON.stringify({ llm: { budgets: { [modelId]: budget } } }) }),
  testKeys: (provider: string) =>
    req<{ provider: string; keys: KeyProbe[] }>(`/api/settings/test-keys?provider=${encodeURIComponent(provider)}`, { method: "POST" }),
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
  // The model id is sent only when one is selected, so the server default applies otherwise (as for quick forecasts).
  createTurn: (notebookId: string, goal: string, model?: string) =>
    req<{ id: string; seq: number }>(`/api/notebooks/${notebookId}/turns`, {
      method: "POST",
      body: JSON.stringify({ goal, ...(model ? { model } : {}) }),
    }),
};

export type SettingsSection = "agent" | "context";

export type KeyStatus = "valid" | "invalid" | "untested";
export type ModelStatus = "verified" | "failed" | "untested";

// Result of POST /api/settings/test-keys. The status is also persisted server-side, so refetch settings afterwards.
export type KeyProbe = {
  slot: number;
  masked: string;
  status: KeyStatus;
  http_status: number | null;
  ok: boolean;
  error: string | null;
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

export type PoolView = { keys: PoolKey[]; demoted_models: { model: string; for_s: number }[]; in_flight: number };

export type ProviderKeyView = {
  slot: number;
  masked: string;
  status: KeyStatus;
  http_status: number | null;
  checked_at: string | null;
};

export type ProviderModelView = {
  id: string;
  label: string;
  enabled: boolean;
  status: ModelStatus;
  checked_at: string | null;
};

export type LlmProviderView = {
  name: string;
  label: string;
  enabled: boolean;
  ready: boolean;
  endpoint: string | null;
  multi_key: boolean;
  key_hint: string;
  keys: ProviderKeyView[];
  models: ProviderModelView[];
};

export type LlmBudget = { "1d": number | null; "7d": number | null; "30d": number | null };

export type LlmView = {
  providers: LlmProviderView[];
  // Typed for the contract; the settings screen does not display budgets yet.
  budgets?: Record<string, LlmBudget>;
  // Present only when the backend reports it. Shown as one line while providers are running requests.
  pool?: PoolView | null;
};

export type SettingsView = {
  llm: LlmView;
  agent: Record<string, number>;
  context: Record<string, number>;
  defaults: Record<SettingsSection, Record<string, number>>;
  bounds: Record<SettingsSection, Record<string, [number, number]>>;
  timesfm: { checkpoint: string; engine_installed: boolean };
  sandbox: {
    timeout_s: number;
    mem_mb: number;
    file_mb: number;
    memory_limit_enforced: boolean;
    isolation: string;
  };
  data: { missions: number; notebooks: number; datasets: number; bytes: number };
  about: { name: string; clear_phrase: string };
};

// -- Model choice and token budgets (composer) --------------------------------------------------------------------------

export type ModelWindow = "1d" | "7d" | "30d";
export type WindowValues<T> = Record<ModelWindow, T>;

export type UsageModel = {
  id: string;
  provider: string;
  label: string;
  used: WindowValues<number>;
  // Null means no budget is set for that window.
  budget: WindowValues<number | null>;
  // ISO timestamps for when each window next resets.
  resets: WindowValues<string>;
};

export type UsageView = { models: UsageModel[] };

export type CatalogModel = { id: string; label: string; enabled: boolean; status: string; checked_at: string | null };
export type CatalogProvider = { name: string; label: string; enabled: boolean; ready: boolean; models: CatalogModel[] };
export type ModelCatalog = { providers: CatalogProvider[] };

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// Only object entries count. A bare string model (the older settings shape) has no enabled flag, so it is never selectable.
function readCatalog(raw: unknown): ModelCatalog {
  const providers = isRecord(raw) && isRecord(raw.llm) && Array.isArray(raw.llm.providers) ? raw.llm.providers : [];
  return {
    providers: providers.filter(isRecord).map((p) => {
      const models = Array.isArray(p.models) ? p.models : [];
      return {
        name: String(p.name ?? ""),
        label: String(p.label ?? p.name ?? ""),
        enabled: p.enabled === true,
        ready: p.ready === true,
        models: models
          .filter(isRecord)
          .map((m) => ({
            id: String(m.id ?? ""),
            label: String(m.label ?? m.id ?? ""),
            enabled: m.enabled === true,
            status: String(m.status ?? ""),
            checked_at: typeof m.checked_at === "string" ? m.checked_at : null,
          }))
          .filter((m) => m.id !== ""),
      };
    }),
  };
}

export function artifactUrl(runId: string, path: string): string {
  return `/api/artifacts/${runId}/${path.replace(/^out\//, "")}`;
}

export function tableDownloadUrl(notebookId: string, name: string, version?: number): string {
  const q = version ? `?version=${version}` : "";
  return `/api/notebooks/${notebookId}/tables/${encodeURIComponent(name)}/download${q}`;
}
