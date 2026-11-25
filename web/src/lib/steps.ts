import type { Mission, StepRow, ToolInfo } from "./api";
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

// Ledger names are plan:d0:3 (depth, then the loop index). Older ledgers used plan:3. A harness decision is auto:d0:3.
const STEP_NAME = /^(composestall|evalstall|evalask|eval2|plan|act|eval|ask|auto):(?:d(\d+):)?(\d+)$/;

type ParsedStep = { kind: string; depth: number; index: number };

function parseStepName(name: string): ParsedStep | null {
  const match = STEP_NAME.exec(name);
  if (!match) return null;
  return { kind: match[1], depth: match[2] === undefined ? 0 : Number(match[2]), index: Number(match[3]) };
}

// One planner pass: the decision, the tool call, the question and the judgement. Ordered by when they were written.
type Pass = {
  key: string;
  order: number;
  plan?: StepRow;
  act?: StepRow;
  eval?: StepRow;
  ask?: StepRow;
};

function passes(steps: StepRow[]): Pass[] {
  const grouped = new Map<string, Pass>();
  for (const row of steps) {
    const parsed = parseStepName(row.name);
    if (!parsed || parsed.kind === "composestall") continue;
    const key = `${parsed.depth}:${parsed.index}`;
    const pass = grouped.get(key) ?? { key, order: row.idx };
    pass.order = Math.min(pass.order, row.idx);
    if (parsed.kind === "plan" || parsed.kind === "auto") {
      if (!pass.plan) pass.plan = row;
    } else if (parsed.kind === "act") pass.act = row;
    else if (parsed.kind === "ask" || parsed.kind === "evalask") pass.ask = row;
    else pass.eval = row;
    grouped.set(key, pass);
  }
  return [...grouped.values()].filter((pass) => pass.plan).sort((a, b) => a.order - b.order);
}

export function buildSteps(mission: Mission, tools: Record<string, ToolInfo>): StepView[] {
  const working = isWorking(mission.status);
  const groups = passes(mission.steps);
  const views: StepView[] = [];

  groups.forEach((group, n) => {
    const plan: Obj = group.plan?.output ?? {};
    const verdict: Obj | undefined = group.eval?.output;
    const followUp = verdict && verdict.verdict === "not_met" && typeof verdict.reason === "string" ? verdict.reason : undefined;
    const note = typeof plan.rationale === "string" ? plan.rationale : "";
    const later = groups.slice(n + 1).some((item) => item.plan);

    if (plan.action === "ask_user") {
      views.push({
        key: `q${group.key}`,
        title: "Asked you a question",
        note,
        state: group.ask ? "done" : "running",
        facts: group.ask?.output && typeof group.ask.output === "object" ? factsFrom(group.ask.output) : [],
        followUp,
        raw: { plan, answers: group.ask?.output },
      });
    } else if (plan.action === "finish") {
      views.push({ key: `f${group.key}`, title: "Wrapped up", note, state: "done", facts: [], raw: plan });
    } else if (plan.action === "delegate") {
      const task = typeof plan.task === "string" ? plan.task : "";
      views.push({
        key: `d${group.key}`,
        title: "Asked a specialist",
        note: note || task,
        state: "done",
        facts: [],
        followUp,
        raw: plan,
      });
    } else {
      const name = String(plan.tool ?? "");
      const out: Obj | undefined = group.act?.output;
      // A repeated call is not stored again. A later pass means this one was skipped on purpose.
      const skippedRepeat = !group.act && later;
      const error = typeof out?.error === "string" ? out.error : !group.act && !working && !skippedRepeat ? "This step did not run." : undefined;
      const args: Obj = (out?.args as Obj) ?? (plan.args as Obj) ?? {};
      const result: Obj = out?.result && typeof out.result === "object" ? (out.result as Obj) : {};
      views.push({
        key: `a${group.key}`,
        title: toolTitle(name, tools),
        note,
        state: error ? "failed" : group.act || skippedRepeat ? "done" : "running",
        facts: factsFrom(args, result),
        code: typeof args.code === "string" ? args.code : undefined,
        output: typeof result.stdout === "string" && result.stdout.trim() ? result.stdout : undefined,
        error,
        followUp,
        raw: { plan, result: out },
      });
    }
  });
  return views;
}

// One plain sentence about what is happening right now. Never a tool's code name.
export function currentActivity(mission: Mission, tools: Record<string, ToolInfo>): string {
  const last = passes(mission.steps).at(-1);
  if (!last) return "Reading your request";
  const plan: Obj = last.plan?.output ?? {};
  if (last.eval) return "Planning the next step";
  if (last.act) return "Checking the result";
  if (plan.action === "call_tool") {
    const name = String(plan.tool ?? "");
    return tools[name]?.activity || `${toolTitle(name, tools)}…`;
  }
  if (plan.action === "ask_user") return "Waiting for your answer";
  if (plan.action === "delegate") return "A specialist is working on one part";
  return "Wrapping up";
}
