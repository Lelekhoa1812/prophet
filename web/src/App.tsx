import { useCallback, useEffect, useState, type FormEvent, type KeyboardEvent } from "react";
import { Artifacts } from "./components/Artifacts";
import { Starfield } from "./components/Starfield";
import { Transcript } from "./components/Transcript";
import { Button, Card, Label, StatusPill } from "./components/ui";
import { api, type ArtifactRun, type MissionRow } from "./lib/api";
import { errorMessage, formatTime, truncate } from "./lib/format";
import { useMission } from "./lib/useMission";
import { cn } from "./lib/utils";

const GOAL_MIN = 3;
const GOAL_MAX = 2000;

export default function App() {
  const [missions, setMissions] = useState<MissionRow[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; nim_keys_configured: boolean } | null>(null);
  const [navOpen, setNavOpen] = useState(false);
  const [runs, setRuns] = useState<ArtifactRun[]>([]);
  const [goal, setGoal] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const { mission, error: pollError, refresh } = useMission(selectedId);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const refreshList = useCallback(() => {
    api
      .missions()
      .then(setMissions)
      .catch(() => {});
  }, []);

  // The list changes when a mission is created or changes status.
  useEffect(() => {
    refreshList();
  }, [refreshList, selectedId, mission?.status]);

  // Artefacts appear as runs finish their tools, so reload when the status or the event count moves.
  const eventCount = mission?.events.length ?? 0;
  useEffect(() => {
    if (!selectedId) {
      setRuns([]);
      return;
    }
    api
      .artifacts(selectedId)
      .then(setRuns)
      .catch(() => setRuns([]));
  }, [selectedId, mission?.status, eventCount]);

  const goalLength = goal.trim().length;
  const canStart = goalLength >= GOAL_MIN && goalLength <= GOAL_MAX && !creating;

  async function startMission(e?: FormEvent) {
    e?.preventDefault();
    if (!canStart) return;
    setCreating(true);
    setCreateError(null);
    try {
      const { id } = await api.createMission(goal.trim());
      setGoal("");
      setSelectedId(id);
      setNavOpen(false);
      refreshList();
    } catch (err) {
      setCreateError(errorMessage(err, "Could not start the mission."));
    } finally {
      setCreating(false);
    }
  }

  function onGoalKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      void startMission();
    }
  }

  const noKeys = health !== null && !health.nim_keys_configured;
  const apiDown = health === null && pollError !== null;

  return (
    <div className="relative min-h-dvh">
      <Starfield />

      <header className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-4 px-4 pt-6 sm:px-8">
        <div className="grid gap-0.5">
          <h1 className="font-display text-4xl leading-none font-semibold text-sand-0">Prophet</h1>
          <p className="text-sm text-sand-1">Forecasting agent for R&amp;D · research use only</p>
        </div>
        <div className="flex items-center gap-3">
          {health && (
            <span
              className={cn(
                "inline-flex items-center gap-2 rounded-full border px-2.5 py-0.5 text-xs",
                health.nim_keys_configured ? "border-ok/50 text-ok" : "border-bad/50 text-bad",
              )}
            >
              <span aria-hidden="true" className="size-1.5 rounded-full bg-current" />
              {health.nim_keys_configured ? "Planner ready" : "Planner not configured"}
            </span>
          )}
          <button
            type="button"
            onClick={() => setNavOpen((v) => !v)}
            aria-expanded={navOpen}
            aria-controls="missions-nav"
            className="min-h-10 rounded-md border border-sand-2/60 px-3 text-sm text-sand-0 hover:border-sand-1 lg:hidden"
          >
            Missions ({missions.length})
          </button>
        </div>
      </header>

      {(noKeys || apiDown) && (
        <div className="mx-auto mt-5 max-w-[1400px] px-4 sm:px-8">
          <Card role="alert" className="border-bad/50 p-4 text-sm text-sand-0">
            {apiDown
              ? "The Prophet API is not reachable. Start it, then reload this page."
              : "NVIDIA NIM keys are not configured on this machine, so missions cannot start. Add them to the git-ignored .env file and restart the API."}
          </Card>
        </div>
      )}

      <main className="mx-auto mt-6 grid max-w-[1400px] gap-6 px-4 pb-16 sm:px-8 lg:grid-cols-[240px_minmax(0,1fr)_340px]">
        <aside
          id="missions-nav"
          aria-label="Missions"
          className={cn("min-w-0 gap-3", navOpen ? "grid" : "hidden", "lg:grid lg:content-start")}
        >
          <Button
            variant="ghost"
            className="w-full justify-start"
            onClick={() => {
              setSelectedId(null);
              setNavOpen(false);
            }}
          >
            New mission
          </Button>
          <div className="grid gap-2">
            <Label>Recent</Label>
            {missions.length === 0 && <p className="text-sm text-sand-2">No missions yet.</p>}
            <ul className="grid gap-2">
              {missions.map((m) => (
                <li key={m.id}>
                  <button
                    type="button"
                    onClick={() => {
                      setSelectedId(m.id);
                      setNavOpen(false);
                    }}
                    aria-current={m.id === selectedId ? "true" : undefined}
                    className={cn(
                      "grid w-full gap-1.5 rounded-md border p-3 text-left transition-colors",
                      m.id === selectedId
                        ? "border-ember/60 bg-ember-soft"
                        : "border-sand-2/30 bg-night-1/70 hover:border-sand-1",
                    )}
                  >
                    <span className="line-clamp-2 text-sm text-sand-0 [overflow-wrap:anywhere]">
                      {truncate(m.goal, 90)}
                    </span>
                    <span className="flex flex-wrap items-center justify-between gap-2 text-xs text-sand-2">
                      <span className="tabular-nums">{formatTime(m.created)}</span>
                      <StatusPill status={m.status} />
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </aside>

        <section aria-label={selectedId ? "Mission transcript" : "Start a mission"} className="grid min-w-0 content-start gap-6">
          {selectedId === null ? (
            <Card className="grid gap-5 p-5 sm:p-7">
              <div className="grid gap-2">
                <Label>New mission</Label>
                <h2 className="font-display text-3xl leading-tight text-sand-0 text-balance">
                  What should Prophet forecast?
                </h2>
                <p className="text-sm text-sand-1">
                  Describe the series, the horizon and the decision it informs. Prophet plans, runs TimesFM 3 and
                  sandboxed analytics, asks when it needs you, and writes a forecast pack you can download.
                </p>
              </div>
              <form onSubmit={startMission} className="grid gap-3">
                <label htmlFor="goal" className="text-sm text-sand-0">
                  Goal
                </label>
                <textarea
                  id="goal"
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  onKeyDown={onGoalKey}
                  rows={4}
                  maxLength={GOAL_MAX}
                  placeholder="e.g. Forecast weekly demand for the next 12 weeks and show the P10 to P90 range."
                  className="min-h-28 w-full resize-y rounded-md border border-sand-2/60 bg-night-0 p-3 text-base text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
                />
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <span className="text-xs text-sand-2 tabular-nums">
                    {goalLength}/{GOAL_MAX} · Ctrl or ⌘ + Enter to start
                  </span>
                  <Button type="submit" disabled={!canStart || noKeys}>
                    {creating ? "Starting…" : "Start mission"}
                  </Button>
                </div>
                {createError && (
                  <p role="alert" className="text-sm text-bad">
                    {createError}
                  </p>
                )}
              </form>
            </Card>
          ) : mission && mission.id === selectedId ? (
            <Transcript mission={mission} onAnswered={refresh} />
          ) : (
            <Card className="p-5">
              <p className="text-sm text-sand-1" role="status">
                {pollError ?? "Loading mission…"}
              </p>
            </Card>
          )}
        </section>

        <aside aria-label="Artefacts" className="grid min-w-0 content-start gap-4">
          <Artifacts runs={runs} />
        </aside>
      </main>

      <footer className="mx-auto max-w-[1400px] px-4 pb-10 text-xs text-sand-2 sm:px-8">
        TimesFM 3 weights are non-commercial: outputs are for research use only, not for commercial decisions, client
        deliverables or paid services. Numbers from synthetic generators are labelled as synthetic.
      </footer>
    </div>
  );
}
