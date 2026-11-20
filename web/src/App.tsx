import { Menu, Plus, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { NotebookView } from "./components/NotebookView";
import { Sidebar } from "./components/Sidebar";
import { Turn } from "./components/Turn";
import { Button, Card, Spinner } from "./components/ui";
import { api, type ArtifactRun, type MissionRow, type NotebookRow, type ToolInfo } from "./lib/api";
import { errorMessage } from "./lib/format";
import { useMission } from "./lib/useMission";
import { cn, isWorking } from "./lib/utils";

// Starting points that carry their own data, so a first-time visitor can see a full run without preparing a file.
const EXAMPLES: { label: string; text: string }[] = [
  {
    label: "Weekly sales, next 8 weeks",
    text: "Forecast the next 8 weeks from these weekly sales: 120, 132, 128, 141, 150, 147, 160, 171, 168, 180, 176, 190, 185, 198, 204, 199.",
  },
  {
    label: "Monthly revenue, next 6 months",
    text: "Here is monthly revenue for two years: 310, 295, 330, 352, 371, 365, 380, 402, 388, 410, 455, 520, 335, 322, 349, 375, 396, 391, 405, 428, 415, 439, 486, 551. Forecast the next 6 months and show the likely range.",
  },
  {
    label: "Spare parts with many zeros",
    text: "Weekly demand for a spare part: 0, 0, 3, 0, 0, 0, 5, 0, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 6, 0, 0, 3. Forecast the next 8 weeks and tell me how much stock to hold.",
  },
];

export default function App() {
  const [missions, setMissions] = useState<MissionRow[]>([]);
  const [notebooks, setNotebooks] = useState<NotebookRow[]>([]);
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; nim_keys_configured: boolean } | null>(null);
  const [tools, setTools] = useState<Record<string, ToolInfo>>({});
  const [navOpen, setNavOpen] = useState(false);
  const [runs, setRuns] = useState<ArtifactRun[]>([]);
  const [draft, setDraft] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const { mission, error: pollError, refresh } = useMission(selectedId);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .tools()
      .then((list) => setTools(Object.fromEntries(list.map((t) => [t.name, t]))))
      .catch(() => {});
  }, []);

  const refreshList = useCallback(() => {
    api.missions().then(setMissions).catch(() => {});
    api.notebooks().then(setNotebooks).catch(() => {});
  }, []);

  useEffect(() => {
    refreshList();
  }, [refreshList, selectedId, selectedNotebookId, mission?.status]);

  const eventCount = mission?.events.length ?? 0;
  useEffect(() => {
    if (!selectedId) {
      setRuns([]);
      return;
    }
    api.artifacts(selectedId).then(setRuns).catch(() => setRuns([]));
  }, [selectedId, mission?.status, eventCount]);

  // Follow the work down the page, unless the person has scrolled up to read. (Notebooks follow their own thread.)
  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTo({ top: el.scrollHeight });
  }, [eventCount, mission?.status, runs.length, selectedId]);

  useEffect(() => {
    if (!navOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setNavOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navOpen]);

  const openNotebook = (id: string) => {
    setSelectedNotebookId(id);
    setSelectedId(null);
    setNavOpen(false);
  };

  const openMission = (id: string) => {
    stick.current = true;
    setSelectedNotebookId(null);
    setSelectedId(id);
    setNavOpen(false);
  };

  const newNotebook = async () => {
    try {
      const nb = await api.createNotebook("New notebook");
      openNotebook(nb.id);
      refreshList();
    } catch (err) {
      setDraft(undefined);
      throw new Error(errorMessage(err, "Couldn't create a notebook. Try again."));
    }
  };

  const start = async (goal: string) => {
    setSending(true);
    try {
      const { id } = await api.createMission(goal);
      stick.current = true;
      setSelectedNotebookId(null);
      setSelectedId(id);
      setDraft(undefined);
      setNavOpen(false);
      refreshList();
    } catch (err) {
      throw new Error(errorMessage(err, "Couldn't start the forecast. Try again."));
    } finally {
      setSending(false);
    }
  };

  const newForecast = () => {
    setSelectedNotebookId(null);
    setSelectedId(null);
    setDraft(undefined);
    setNavOpen(false);
  };

  const stopForecast = async () => {
    if (!selectedId) return;
    setStopping(true);
    try {
      await api.cancel(selectedId);
    } finally {
      setStopping(false);
      refresh();
      refreshList();
    }
  };

  const noKeys = health !== null && !health.nim_keys_configured;
  const apiDown = health === null && pollError !== null;
  const current = mission && mission.id === selectedId ? mission : null;
  const inNotebook = selectedNotebookId !== null;
  const waitingOnYou = current?.status === "waiting_user";
  const blockedReason = noKeys
    ? "Prophet needs a model key before it can start."
    : waitingOnYou
      ? "Answer the question above to continue."
      : current && isWorking(current.status)
        ? "Prophet is working. Press Stop to interrupt it."
        : null;

  const sidebar = (
    <Sidebar
      notebooks={notebooks}
      missions={missions}
      selectedNotebookId={selectedNotebookId}
      selectedMissionId={selectedId}
      onSelectNotebook={openNotebook}
      onSelectMission={openMission}
      onNewNotebook={() => void newNotebook().catch(() => {})}
      onNewForecast={newForecast}
      onHideMission={(id) => void api.archiveMission(id).then(refreshList).catch(() => {})}
    />
  );

  return (
    <div className="flex h-dvh bg-night-0 text-sand-0">
      <aside className="hidden w-72 shrink-0 flex-col gap-5 border-r border-sand-2/20 bg-night-1/50 p-4 lg:flex">
        <h1 className="px-2 text-2xl font-bold">Prophet</h1>
        <div className="min-h-0 flex-1">{sidebar}</div>
      </aside>

      {navOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button type="button" aria-label="Close menu" onClick={() => setNavOpen(false)} className="absolute inset-0 bg-black/60" />
          <div className="absolute inset-y-0 left-0 flex w-[86%] max-w-80 flex-col gap-4 bg-night-1 p-4">
            <div className="flex items-center justify-between">
              <span className="px-2 text-2xl font-bold">Prophet</span>
              <Button variant="ghost" size="icon" aria-label="Close menu" onClick={() => setNavOpen(false)}>
                <X aria-hidden="true" className="size-5" />
              </Button>
            </div>
            <div className="min-h-0 flex-1">{sidebar}</div>
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-2 border-b border-sand-2/20 px-3 py-2 lg:hidden">
          <Button variant="ghost" size="icon" aria-label="Open menu" aria-expanded={navOpen} onClick={() => setNavOpen(true)}>
            <Menu aria-hidden="true" className="size-5" />
          </Button>
          <span className="text-xl font-bold">Prophet</span>
          <Button variant="ghost" size="icon" aria-label="New notebook" onClick={() => void newNotebook().catch(() => {})}>
            <Plus aria-hidden="true" className="size-5" />
          </Button>
        </header>

        {(noKeys || apiDown) && (
          <div className="px-4 pt-4 sm:px-6">
            <Card role="alert" className="mx-auto grid w-full max-w-3xl gap-1 border-bad/40 p-4">
              <p className="text-base font-bold">{apiDown ? "Prophet isn't running" : "Prophet needs a model key"}</p>
              <p className="text-[0.9375rem] text-sand-1">
                {apiDown
                  ? "Start the Prophet server, then reload this page."
                  : "Add your NVIDIA key to the .env file in the project folder, then restart the server."}
              </p>
            </Card>
          </div>
        )}

        {inNotebook ? (
          <main className="min-h-0 flex-1 overflow-hidden">
            <NotebookView key={selectedNotebookId} notebookId={selectedNotebookId} tools={tools} onChanged={refreshList} />
          </main>
        ) : (
          <main
            ref={scroller}
            onScroll={(e) => {
              const el = e.currentTarget;
              stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
            }}
            className="min-h-0 flex-1 overflow-y-auto"
          >
            <div className="mx-auto grid w-full max-w-3xl gap-6 px-4 py-6 sm:px-6">
              {selectedId === null ? (
                <section aria-label="Start" className="grid gap-8 pt-6 sm:pt-14">
                  <div className="grid gap-3">
                    <h2 className="text-3xl leading-tight font-bold text-balance sm:text-4xl">What would you like to work on?</h2>
                    <p className="text-lg text-sand-1">
                      Start a notebook to add spreadsheets and ask questions about them, or ask a quick forecast below.
                    </p>
                    <div>
                      <Button onClick={() => void newNotebook().catch(() => {})}>
                        <Plus aria-hidden="true" className="size-4" />
                        New notebook
                      </Button>
                    </div>
                  </div>
                  <div className="grid gap-2">
                    <p className="text-[0.9375rem] font-semibold text-sand-2">Or try a quick forecast</p>
                    <ul className="flex flex-wrap gap-2">
                      {EXAMPLES.map((ex) => (
                        <li key={ex.label}>
                          <button
                            type="button"
                            onClick={() => setDraft(ex.text)}
                            className="min-h-10 rounded-full border border-sand-2/40 px-4 text-base text-sand-0 transition-colors hover:border-sand-1 hover:bg-night-2"
                          >
                            {ex.label}
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                </section>
              ) : current ? (
                <Turn mission={current} tools={tools} runs={runs} onAnswered={refresh} onRetry={(g) => void start(g).catch(() => {})} />
              ) : (
                <p role="status" className="flex items-center gap-2 text-base text-sand-1">
                  {pollError ? (
                    <span className="text-bad">{pollError}</span>
                  ) : (
                    <>
                      <Spinner /> Loading
                    </>
                  )}
                </p>
              )}
            </div>
          </main>
        )}

        {!inNotebook && (
          <div className={cn("border-t border-sand-2/15 bg-night-0 px-4 pt-3 pb-3 sm:px-6")}>
            <div className="mx-auto w-full max-w-3xl">
              <Composer
                onSend={start}
                busy={sending}
                blockedReason={blockedReason}
                working={current !== null && isWorking(current.status)}
                onStop={() => void stopForecast()}
                stopping={stopping}
                initial={draft}
                placeholder={selectedId === null ? "Describe a quick forecast" : "Describe another forecast"}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
