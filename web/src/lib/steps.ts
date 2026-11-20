import type { Mission, ToolInfo } from "./api";
import { formatNumber, humanise, truncate } from "./format";
import { isWorking } from "./utils";

// Turns the raw plan / act / ask / eval records into rows a person can read. Code names and JSON stay behind "Details".
export type StepState = "running" | "done" | "failed";

export type StepView = {
  key: string;
  title: string;
  note: string;
  state: StepState;
  facts: [string, string][];
  code?: string;
  output?: string;
  error?: string;
  followUp?: string; // what the planner said was still missing after this step
  raw: unknown;
};

type Obj = Record<string, any>;
const HIDDEN_FACT_KEYS = new Set(["code", "stdout", "stderr", "files", "run_id", "manifest", "kind", "tool", "args", "result"]);

function factValue(v: unknown): string | null {
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return formatNumber(v);
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return truncate(v, 120);
  if (Array.isArray(v)) return `${v.length} ${v.length === 1 ? "item" : "items"}`;
  return null;
}

export function factsFrom(...sources: unknown[]): [string, string][] {
  const out: [string, string][] = [];
  for (const src of sources) {
    if (!src || typeof src !== "object") continue;
    for (const [k, v] of Object.entries(src as Obj)) {
      if (HIDDEN_FACT_KEYS.has(k) || out.length >= 8) continue;
      const val = factValue(v);
      if (val !== null) out.push([humanise(k), val]);
    }
  }
  return out;
}

function toolTitle(name: string, tools: Record<string, ToolInfo>): string {
  return tools[name]?.title || humanise(name);
}

export function buildSteps(mission: Mission, tools: Record<string, ToolInfo>): StepView[] {
  const byName = new Map(mission.steps.map((s) => [s.name, s]));
  const working = isWorking(mission.status);
  const views: StepView[] = [];

  for (let i = 0; ; i++) {
    const planRow = byName.get(`plan:${i}`);
    if (!planRow) break;
    const plan: Obj = planRow.output ?? {};
    const evalRow = byName.get(`eval:${i}`);
    const verdict: Obj | undefined = evalRow?.output;
    const followUp = verdict && verdict.verdict === "not_met" && typeof verdict.reason === "string" ? verdict.reason : undefined;
    const note = typeof plan.rationale === "string" ? plan.rationale : "";

    if (plan.action === "ask_user") {
      const ask = byName.get(`ask:${i}`);
      views.push({
        key: `q${i}`,
        title: "Asked you a question",
        note,
        state: ask ? "done" : "running",
        facts: ask?.output && typeof ask.output === "object" ? factsFrom(ask.output) : [],
        followUp,
        raw: { plan, answers: ask?.output },
      });
    } else if (plan.action === "finish") {
      views.push({ key: `f${i}`, title: "Wrapped up", note, state: "done", facts: [], raw: plan });
    } else {
      const name = String(plan.tool ?? "");
      const act = byName.get(`act:${i}`);
      const out: Obj | undefined = act?.output;
      const error = typeof out?.error === "string" ? out.error : !act && !working ? "This step did not run." : undefined;
      const args: Obj = (out?.args as Obj) ?? (plan.args as Obj) ?? {};
      const result: Obj = out?.result && typeof out.result === "object" ? (out.result as Obj) : {};
      views.push({
        key: `a${i}`,
        title: toolTitle(name, tools),
        note,
        state: error ? "failed" : act ? "done" : "running",
        facts: factsFrom(args, result),
        code: typeof args.code === "string" ? args.code : undefined,
        output: typeof result.stdout === "string" && result.stdout.trim() ? result.stdout : undefined,
        error,
        followUp,
        raw: { plan, result: out },
      });
    }
  }
  return views;
}

// One plain sentence about what is happening right now. Never a tool's code name.
export function currentActivity(mission: Mission, tools: Record<string, ToolInfo>): string {
  const names = new Set(mission.steps.map((s) => s.name));
  let last = -1;
  while (names.has(`plan:${last + 1}`)) last++;
  if (last < 0) return "Reading your request";
  const plan: Obj = mission.steps.find((s) => s.name === `plan:${last}`)?.output ?? {};
  const hasAct = names.has(`act:${last}`);
  const hasEval = names.has(`eval:${last}`);
  if (hasEval) return "Planning the next step";
  if (hasAct) return "Checking the result";
  if (plan.action === "call_tool") {
    const name = String(plan.tool ?? "");
    return tools[name]?.activity || `${toolTitle(name, tools)}…`;
  }
  if (plan.action === "ask_user") return "Waiting for your answer";
  return "Wrapping up";
}
