import { Menu, Plus, Settings as SettingsIcon, Undo2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Composer } from "./components/Composer";
import { NotebookView } from "./components/NotebookView";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { Turn } from "./components/Turn";
import { Button, Card, IconButton, Spinner } from "./components/ui";
import { api, type ArtifactRun, type MissionRow, type NotebookRow, type ToolInfo } from "./lib/api";
import { errorMessage } from "./lib/format";
import { useMission } from "./lib/useMission";
import { cn, isWorking } from "./lib/utils";

// One notice slot above the main column. Errors are never swallowed: they land here until dismissed.
type Notice = { id: number; message: string; tone: "error" | "info"; action?: { label: string; run: () => void } };

export default function App() {
  const [missions, setMissions] = useState<MissionRow[]>([]);
  const [notebooks, setNotebooks] = useState<NotebookRow[]>([]);
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [health, setHealth] = useState<{ ok: boolean; nim_keys_configured: boolean } | null>(null);
  const [tools, setTools] = useState<Record<string, ToolInfo>>({});
  const [navOpen, setNavOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [runs, setRuns] = useState<ArtifactRun[]>([]);
  const [draft, setDraft] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const { mission, error: pollError, refresh } = useMission(selectedId);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const menuButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const wasNavOpen = useRef(false);

  const showError = (message: string) => setNotice({ id: Date.now(), message, tone: "error" });

  // Keys can change in Settings, so the header's "model key" state is read again when the page closes.
  const closeSettings = () => {
    setSettingsOpen(false);
    api.health().then(setHealth).catch(() => undefined);
  };

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .tools()
      .then((list) => setTools(Object.fromEntries(list.map((t) => [t.name, t]))))
      .catch(() => showError("Couldn't load the tool list. Refresh the page to try again."));
  }, []);

  const refreshList = useCallback(() => {
    const failed = () =>
      setNotice({ id: Date.now(), message: "Couldn't refresh your list. Check the connection and try again.", tone: "error" });
    api.missions().then(setMissions).catch(failed);
    api.notebooks().then(setNotebooks).catch(failed);
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

  // Focus moves into the drawer when it opens and returns to the menu button when it closes.
  useEffect(() => {
    if (navOpen) closeButton.current?.focus();
    else if (wasNavOpen.current) menuButton.current?.focus();
    wasNavOpen.current = navOpen;
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
      showError(errorMessage(err, "Couldn't create a notebook. Try again."));
    }
  };

  // Hiding is reversible: the notice offers Undo, which restores the forecast to the list.
  const hideMission = async (id: string) => {
    try {
      await api.archiveMission(id);
      refreshList();
      setNotice({
        id: Date.now(),
        message: "Forecast hidden.",
        tone: "info",
        action: {
          label: "Undo",
          run: () => {
            api
              .unarchiveMission(id)
              .then(() => {
                refreshList();
                setNotice(null);
              })
              .catch((err) => showError(errorMessage(err, "Couldn't restore this forecast. Try again.")));
          },
        },
      });
    } catch (err) {
      showError(errorMessage(err, "Couldn't hide this forecast. Try again."));
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
    } catch (err) {
      showError(errorMessage(err, "Couldn't stop Prophet. Try again."));
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
      onNewNotebook={() => void newNotebook()}
      onNewForecast={newForecast}
      onHideMission={(id) => void hideMission(id)}
    />
  );

  return (
    <div className="flex h-dvh bg-night-0 text-sand-0">
      <aside className="hidden w-72 shrink-0 flex-col gap-5 border-r border-sand-2/20 bg-night-1/50 p-4 lg:flex" inert={navOpen}>
        <div className="flex items-center justify-between gap-2">
          <h1 className="px-2 font-display text-2xl font-semibold">Prophet</h1>
          <IconButton label="Settings" icon={<SettingsIcon aria-hidden="true" className="size-5" />} onClick={() => setSettingsOpen(true)} />
        </div>
        <div className="min-h-0 flex-1">{sidebar}</div>
      </aside>

      {navOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button type="button" tabIndex={-1} aria-hidden="true" onClick={() => setNavOpen(false)} className="absolute inset-0 bg-black/60" />
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Workspace"
            className="absolute inset-y-0 left-0 flex w-[86%] max-w-80 flex-col gap-4 bg-night-1 p-4"
          >
            <div className="flex items-center justify-between">
              <span className="px-2 font-display text-2xl font-semibold">Prophet</span>
              <div className="flex items-center">
                <IconButton label="Settings" icon={<SettingsIcon aria-hidden="true" className="size-5" />} onClick={() => setSettingsOpen(true)} />
                <IconButton ref={closeButton} label="Close menu" icon={<X aria-hidden="true" className="size-5" />} onClick={() => setNavOpen(false)} />
              </div>
            </div>
            <div className="min-h-0 flex-1">{sidebar}</div>
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col" inert={navOpen}>
        <header className="flex items-center justify-between gap-2 border-b border-sand-2/20 px-3 py-2 lg:hidden">
          <IconButton ref={menuButton} label="Open menu" aria-expanded={navOpen} icon={<Menu aria-hidden="true" className="size-5" />} onClick={() => setNavOpen(true)} />
          <span className="font-display text-xl font-semibold">Prophet</span>
          <div className="flex items-center">
            <IconButton label="New notebook" icon={<Plus aria-hidden="true" className="size-5" />} onClick={() => void newNotebook()} />
            <IconButton label="Settings" icon={<SettingsIcon aria-hidden="true" className="size-5" />} onClick={() => setSettingsOpen(true)} />
          </div>
        </header>

        {notice && (
          <div className="px-4 pt-4 sm:px-6">
            <div
              role={notice.tone === "error" ? "alert" : "status"}
              className={cn(
                "mx-auto flex w-full max-w-3xl flex-wrap items-center justify-between gap-3 rounded-xl border bg-night-1 p-3 text-caption",
                notice.tone === "error" ? "border-bad/40" : "border-sand-2/30",
              )}
            >
              <p className={cn("min-w-0 [overflow-wrap:anywhere]", notice.tone === "error" ? "text-bad" : "text-sand-0")}>{notice.message}</p>
              <div className="flex items-center gap-1">
                {notice.action && (
                  <IconButton label={notice.action.label} icon={<Undo2 aria-hidden="true" className="size-5" />} onClick={notice.action.run} />
                )}
                <IconButton label="Dismiss message" icon={<X aria-hidden="true" className="size-5" />} onClick={() => setNotice(null)} />
              </div>
            </div>
          </div>
        )}

        {(noKeys || apiDown) && (
          <div className="px-4 pt-4 sm:px-6">
            <Card role="alert" className="mx-auto grid w-full max-w-3xl gap-2 border-bad/40 p-4">
              <p className="text-base font-bold">{apiDown ? "Prophet isn't running" : "Prophet needs a model key"}</p>
              <p className="text-caption text-sand-1">
                {apiDown
                  ? "Start the Prophet server, then reload this page."
                  : "Prophet cannot run yet: no model key is set. Ask the person who set up this app to add one, then reload."}
              </p>
              {noKeys && (
                <details className="text-caption text-sand-1">
                  <summary className="inline-flex min-h-10 cursor-pointer items-center text-sand-2 hover:text-sand-0 sm:min-h-0">Details</summary>
                  <p className="pt-1">Technical detail: add your NVIDIA key to the .env file in the project folder, then restart the server.</p>
                </details>
              )}
            </Card>
          </div>
        )}

        {pollError && (
          <div className="px-4 pt-4 sm:px-6">
            <Card role="alert" className="mx-auto grid w-full max-w-3xl gap-1 border-bad/40 p-4">
              <p className="text-caption text-sand-0">
                Lost contact with Prophet. Retrying every few seconds. Progress will update when the connection is back.
              </p>
            </Card>
          </div>
        )}

        {inNotebook ? (
          <main className="min-h-0 flex-1 overflow-hidden">
            <NotebookView
              key={selectedNotebookId}
              notebookId={selectedNotebookId}
              tools={tools}
              onChanged={refreshList}
              onError={showError}
            />
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
                    <h2 className="font-display text-3xl leading-tight font-semibold text-balance sm:text-4xl">What would you like to work on?</h2>
                    <p className="text-lg text-sand-1">
                      Start a notebook to add spreadsheets and ask questions about them, or ask a quick forecast below.
                    </p>
                    <div>
                      <Button onClick={() => void newNotebook()}>
                        <Plus aria-hidden="true" className="size-4" />
                        New notebook
                      </Button>
                    </div>
                  </div>
                </section>
              ) : current ? (
                <Turn
                  mission={current}
                  tools={tools}
                  runs={runs}
                  onAnswered={refresh}
                  onRetry={(g) => void start(g).catch((err) => showError(errorMessage(err, "Couldn't start the forecast. Try again.")))}
                  onEdit={(g) => setDraft(g)}
                />
              ) : (
                <p role="status" className="flex items-center gap-2 text-base text-sand-1">
                  <Spinner /> Loading
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

      <SettingsModal
        open={settingsOpen}
        onClose={closeSettings}
        onDataChanged={() => {
          setSelectedId(null);
          setSelectedNotebookId(null);
          refreshList();
          api.health().then(setHealth).catch(() => undefined);
        }}
      />
    </div>
  );
}
