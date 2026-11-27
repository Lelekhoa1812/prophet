import { useEffect, useRef, useState } from "react";
import { errorMessage } from "./format";

// The right-hand panel's data (API: /api/missions/{id}/activity, /api/notebooks/{id}/activity). Everything is read from the
// mission's stored record; nothing here asks the model.
export type MainThread = {
  mission_id: string;
  title: string;
  status: string;
  step: number | null;
  tool_calls: number | null;
  elapsed_s: number | null;
  next: string | null;
  doing: string | null;
};
export type CodeRun = { step: number; tool: string; title: string; agent: string; status: string; run_id: string | null; updated: number };
export type AgentRow = {
  agent_id: string;
  agent_type: string;
  task: string;
  lane: boolean;
  status: string;
  steps: number;
  tool_calls: number;
  last_tool: string | null;
  started: number | null;
};
export type OutputFile = { name: string; path: string; run_id: string; sha256: string; bytes: number | null; kind: string; url: string; step: number };
export type FileGroup = { mission_id: string; topic: string; files: OutputFile[] };
export type Activity = {
  main: MainThread | null;
  code: CodeRun[];
  agents: AgentRow[];
  files: FileGroup[];
  tools: { name: string; title: string; calls: number }[];
  skills: { name: string; title: string; source: string }[];
};
export type SessionEntry =
  | { role: "plan"; step: number; action: string | null; tool: string | null; rationale: string; args: string | null }
  | { role: "result"; step: number; tool: string | null; ok: boolean; text: string }
  | { role: "evaluation"; step: number; verdict: string | null; text: string }
  | { role: "board"; step: number; text: string };
export type AgentSession = {
  mission_id: string;
  agent_id: string;
  agent_type: string;
  task: string;
  deliverable: string;
  lane: boolean;
  status: string;
  answer: string;
  findings: string[];
  entries: SessionEntry[];
};

// What the panel shows: a notebook (this round is its newest question) or one quick forecast.
export type ActivityTarget = { kind: "notebook"; id: string } | { kind: "mission"; id: string };

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
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

export const activityApi = {
  activity: (t: ActivityTarget) =>
    get<Activity>(t.kind === "notebook" ? `/api/notebooks/${encodeURIComponent(t.id)}/activity` : `/api/missions/${encodeURIComponent(t.id)}/activity`),
  agent: (missionId: string, agentId: string) =>
    get<AgentSession>(`/api/missions/${encodeURIComponent(missionId)}/agents/${encodeURIComponent(agentId)}`),
};

export function isRunning(status: string | undefined | null): boolean {
  return status === "running" || status === "created" || status === "started";
}

// Polls the panel: every 3 s while the round is working, every 20 s otherwise (a new question may start at any time).
export function useActivity(target: ActivityTarget | null, enabled = true) {
  const [data, setData] = useState<Activity | null>(null);
  const [error, setError] = useState<string | null>(null);
  const key = target ? `${target.kind}:${target.id}` : "";
  const keyRef = useRef(key);

  useEffect(() => {
    keyRef.current = key;
    setData(null);
    setError(null);
    if (!target || !enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      let next = 20000;
      try {
        const got = await activityApi.activity(target);
        if (cancelled || keyRef.current !== key) return;
        setData(got);
        setError(null);
        if (isRunning(got.main?.status)) next = 3000;
      } catch (err) {
        if (cancelled) return;
        setError(errorMessage(err, "Could not load the activity"));
        next = 8000;
      }
      if (!cancelled) timer = setTimeout(tick, document.hidden ? Math.max(next, 15000) : next);
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
    // target is identified by key
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  return { data, error };
}

// One sub-agent's session, polled while it runs.
export function useAgentSession(missionId: string | null, agentId: string | null) {
  const [session, setSession] = useState<AgentSession | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setSession(null);
    setError(null);
    if (!missionId || !agentId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const got = await activityApi.agent(missionId, agentId);
        if (cancelled) return;
        setSession(got);
        setError(null);
        if (isRunning(got.status)) timer = setTimeout(tick, 3000);
      } catch (err) {
        if (!cancelled) setError(errorMessage(err, "Could not load this sub-agent"));
      }
    };
    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [missionId, agentId]);

  return { session, error };
}
