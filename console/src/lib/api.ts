// Typed client for the lab API. Every page reads through here so loading and error states stay consistent.
import { useCallback, useEffect, useState } from "react";

export type Severity = "high" | "medium" | "low";
export interface Fact { label: string; value: string | number; unit?: string }
export interface SeriesEvidence {
  kind: "series"; label: string; unit: string;
  history: { x: string[]; y: number[] };
  forecast: { x: string[]; q10: number[]; q50: number[]; q90: number[] };
}
export interface Proposal {
  proposal_id: string; agent: string; as_of: string; entity_type: string; entity_id: string; title: string;
  useful_rate?: number | null; severity: Severity; dollars_at_stake: number; confidence: number; autonomy_level: string; owner: string; status: string;
  evidence: { series?: SeriesEvidence | null; facts?: Fact[]; why?: string; method?: string; targets?: any[]; lots?: any[] };
  action: Record<string, any>; narration_templated?: string; narration_model?: string | null;
  decisions?: { at: string; by: string; decision: string; reason?: string; note?: string }[];
}
export interface AgentInfo {
  key: string; title: string; owner: string; cadence: string; summary: string;
  config: { enabled: boolean; owner: string; quiet_start: string; quiet_end: string; cooldown_days: number; params: Record<string, any>; autonomy_level: string };
  stats: Record<string, any>; operator_decisions: Record<string, number>; defaults: Record<string, any>;
  ladder: { current: string; label: string; next: string | null; next_label: string | null; eligible: boolean; demote: boolean;
    criteria: { label: string; value: number | null; target: number | null; met: boolean; direction: string }[] };
}
export interface Meta {
  model: { name: string; checkpoint: string; backend: string; loaded: boolean; licence: string; licence_status: string };
  datasets: Record<string, { key: string; source: string; first: string; last: string; counts: Record<string, number>; revenue: number; has_truth: boolean; extracted_at?: string }>;
  agents: Record<string, { title: string; owner: string; cadence: string; summary: string }>;
  bedrock: { model: string; spent_usd: number; cap_usd: number };
  ladder: { key: string; label: string }[];
  ship_bar: { improvement_vs_best_baseline: number; band_coverage: number[]; useful_share: number; invented_numbers: number };
}

export async function api<T = any>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r.json();
}
export const post = <T = any>(path: string, body?: unknown) => api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
export const put = <T = any>(path: string, body: unknown) => api<T>(path, { method: "PUT", body: JSON.stringify(body) });

/** Fetch with loading and error state. Re-runs when `path` changes. Pass null to skip. */
export function useApi<T = any>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!path);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!path) { setData(null); setLoading(false); return; }
    let live = true;
    setLoading(true); setError(null);
    api<T>(path).then((d) => live && setData(d)).catch((e) => live && setError(String(e.message ?? e))).finally(() => live && setLoading(false));
    return () => { live = false; };
  }, [path, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}
