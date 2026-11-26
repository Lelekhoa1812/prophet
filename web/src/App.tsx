import { BarChart3, BookOpen, Menu, PanelLeft, PanelLeftClose, Plus, ScanSearch, Settings as SettingsIcon, Undo2, Upload, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import logo from "./public/logo.png";
import { Composer, type Attachment } from "./components/Composer";
import { ComposerControls } from "./components/ComposerControls";
import { NotebookView } from "./components/NotebookView";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar, type ItemKind } from "./components/Sidebar";
import { Turn } from "./components/Turn";
import { Button, Card, IconButton, Spinner } from "./components/ui";
import { api, type ArtifactRun, type DatasetRef, type Mission, type MissionRow, type NotebookRow, type ToolInfo } from "./lib/api";
import { offerDraft } from "./lib/checkpoint";
import { errorMessage } from "./lib/format";
import { parseRoute, pathFor, routeFor } from "./lib/route";
import { SKILLS } from "./lib/skills";
import { useMission } from "./lib/useMission";
import { useModelCatalog, useSelectedModel } from "./lib/useModels";
import { cn, isWorking } from "./lib/utils";

// One notice slot above the main column. Errors are never swallowed: they land here until dismissed.
type Notice = { id: number; message: string; tone: "error" | "info"; action?: { label: string; run: () => void }; ttlMs?: number };

export default function App() {
  const [missions, setMissions] = useState<MissionRow[]>([]);
  const [notebooks, setNotebooks] = useState<NotebookRow[]>([]);
  // The address carries the place: /n/<id> opens a notebook, /c/<id> a quick forecast.
  const [selectedNotebookId, setSelectedNotebookId] = useState<string | null>(() => {
    const r = parseRoute(window.location.pathname);
    return r?.kind === "n" ? r.id : null;
  });
  const [selectedId, setSelectedId] = useState<string | null>(() => {
    const r = parseRoute(window.location.pathname);
    return r?.kind === "c" ? r.id : null;
  });
  const [health, setHealth] = useState<{ ok: boolean; nim_keys_configured: boolean } | null>(null);
  const [tools, setTools] = useState<Record<string, ToolInfo>>({});
  const [navOpen, setNavOpen] = useState(false);
  // Desktop sidebar can be collapsed out of the way; the choice is remembered per browser (best effort).
  const [sidebarHidden, setSidebarHidden] = useState<boolean>(() => {
    try {
      return localStorage.getItem("prophet.sidebarHidden") === "1";
    } catch {
      return false;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("prophet.sidebarHidden", sidebarHidden ? "1" : "0");
    } catch {
      /* storage unavailable: the sidebar simply starts open next time */
    }
  }, [sidebarHidden]);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [runs, setRuns] = useState<ArtifactRun[]>([]);
  const [draft, setDraft] = useState<string | undefined>(undefined);
  const [sending, setSending] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const attachSeq = useRef(0);
  const [stopping, setStopping] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  // A notice with ttlMs clears itself. Only the notice it was set for is cleared, so a newer one is never removed early.
  const noticeId = notice?.id;
  const noticeTtl = notice?.ttlMs;
  useEffect(() => {
    if (noticeId === undefined || !noticeTtl) return;
    const timer = window.setTimeout(() => setNotice((n) => (n?.id === noticeId ? null : n)), noticeTtl);
    return () => window.clearTimeout(timer);
  }, [noticeId, noticeTtl]);
  const { mission, error: pollError, refresh } = useMission(selectedId);
  // The model for the next forecast is chosen under the composer and remembered per browser.
  const catalog = useModelCatalog();
  const { model, choose: chooseModel } = useSelectedModel(catalog.selectable);
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
    catalog.reload();
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

  // Keep the address in step with what is open, so the link can be copied, bookmarked or reloaded.
  useEffect(() => {
    const want = pathFor(routeFor(selectedNotebookId, selectedId));
    if (window.location.pathname !== want) window.history.pushState(null, "", want);
  }, [selectedNotebookId, selectedId]);

  // Back and forward move between what was open.
  useEffect(() => {
    const onPop = () => {
      const r = parseRoute(window.location.pathname);
      setSelectedNotebookId(r?.kind === "n" ? r.id : null);
      setSelectedId(r?.kind === "c" ? r.id : null);
      setNavOpen(false);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // A link to something that is gone (deleted, or from another machine) lands on the start page with a notice, and a
  // forecast that belongs to a notebook opens in its notebook.
  useEffect(() => {
    const r = parseRoute(window.location.pathname);
    if (!r) return;
    const home = (message: string) => {
      window.history.replaceState(null, "", "/");
      setSelectedNotebookId(null);
      setSelectedId(null);
      setNotice({ id: Date.now(), message, tone: "info", ttlMs: 6000 });
    };
    if (r.kind === "n") {
      api.notebook(r.id).catch(() => home("That notebook isn't here. It may have been deleted."));
    } else {
      api
        .mission(r.id)
        .then((m) => {
          if (!m.notebook_id) return;
          window.history.replaceState(null, "", pathFor({ kind: "n", id: m.notebook_id }));
          setSelectedId(null);
          setSelectedNotebookId(m.notebook_id);
        })
        .catch(() => home("That chat isn't here. It may have been deleted."));
    }
  }, []);

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

  // Staged files belong to the view they were added in: switching to another item clears them, so nothing carries over.
  const openNotebook = (id: string) => {
    if (id !== selectedNotebookId) setAttachments([]);
    setSelectedNotebookId(id);
    setSelectedId(null);
    setNavOpen(false);
  };

  const openMission = (id: string) => {
    stick.current = true;
    if (id !== selectedId) setAttachments([]);
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

  // Files are added to the quick forecast as data sets. Each one uploads in the background; the question waits for them.
  const addAttachments = (files: File[]) => {
    const usable = files.filter((f) => f.name && !f.name.startsWith("."));
    if (usable.length === 0) return;
    const items = usable.map((file) => ({ key: `att-${++attachSeq.current}`, file }));
    setAttachments((prev) => [...prev, ...items.map(({ key, file }) => ({ key, name: file.name, status: "uploading" as const }))]);
    let next = 0;
    const worker = async () => {
      while (next < items.length) {
        const { key, file } = items[next++];
        try {
          const meta = await api.uploadDataset(file);
          setAttachments((prev) => prev.map((a) => (a.key === key ? { ...a, status: "ready" as const, datasetId: meta.id } : a)));
        } catch (err) {
          const message = errorMessage(err, "Couldn't add this file.");
          setAttachments((prev) => prev.map((a) => (a.key === key ? { ...a, status: "failed" as const, message } : a)));
        }
      }
    };
    // Three at a time: quick enough for folders with many files, and gentle on the local server.
    void Promise.all([worker(), worker(), worker()]);
  };

  const removeAttachment = (key: string) => setAttachments((prev) => prev.filter((a) => a.key !== key));
  const readyDatasetIds = () => attachments.flatMap((a) => (a.status === "ready" && a.datasetId ? [a.datasetId] : []));

  const renameItem = async (kind: ItemKind, id: string, title: string) => {
    try {
      if (kind === "forecast") await api.renameMission(id, title);
      else await api.patchNotebook(id, { title });
      refreshList();
    } catch (err) {
      showError(errorMessage(err, "Couldn't rename this. Try again."));
    }
  };

  const pinItem = async (kind: ItemKind, id: string, pinned: boolean) => {
    try {
      if (kind === "forecast") await api.pinMission(id, pinned);
      else await api.patchNotebook(id, { pinned });
      refreshList();
    } catch (err) {
      showError(errorMessage(err, "Couldn't change the pin. Try again."));
    }
  };

  const archiveItem = (kind: ItemKind, id: string) => {
    if (kind === "forecast") {
      void hideMission(id);
      return;
    }
    api
      .patchNotebook(id, { archived: true })
      .then(() => {
        if (selectedNotebookId === id) setSelectedNotebookId(null);
        refreshList();
        setNotice({ id: Date.now(), message: "Notebook archived.", tone: "info" });
      })
      .catch((err) => showError(errorMessage(err, "Couldn't archive this notebook. Try again.")));
  };

  const deleteItem = async (kind: ItemKind, id: string) => {
    try {
      if (kind === "forecast") {
        await api.deleteMission(id);
        if (selectedId === id) setSelectedId(null);
      } else {
        await api.deleteNotebook(id);
        if (selectedNotebookId === id) setSelectedNotebookId(null);
      }
      setAttachments([]);
      refreshList();
      setNotice({ id: Date.now(), message: kind === "forecast" ? "Forecast deleted." : "Notebook deleted.", tone: "info", ttlMs: 2000 });
    } catch (err) {
      showError(errorMessage(err, "Couldn't delete this. Try again."));
    }
  };

  const start = async (goal: string, datasetIds: string[] = []) => {
    setSending(true);
    try {
      const { id } = await api.createMission(goal, datasetIds, model?.id);
      stick.current = true;
      setSelectedNotebookId(null);
      setSelectedId(id);
      setDraft(undefined);
      setNavOpen(false);
      if (datasetIds.length > 0) setAttachments([]);
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

  // Fork and Rewind on a quick forecast, which has one message and nothing before it. Both give the message and its files back
  // in a fresh box. A fork leaves the forecast as it is; a rewind hides it (as Hide does) and offers Undo.
  const filesBack = (ds: DatasetRef[]) =>
    setAttachments(ds.map((d) => ({ key: `att-${++attachSeq.current}`, name: d.name, status: "ready" as const, datasetId: d.id })));

  const forkForecast = (m: Mission) => {
    newForecast();
    setDraft(m.goal);
    filesBack(m.datasets ?? []);
    setNotice({ id: Date.now(), message: "Forked. The original forecast is unchanged.", tone: "info", ttlMs: 4000 });
  };

  const rewindForecast = async (m: Mission) => {
    try {
      const out = await api.rewindMission(m.id);
      newForecast();
      setDraft(out.prefill);
      filesBack(m.datasets ?? []);
      refreshList();
      setNotice({
        id: Date.now(),
        message: "Rewound. The forecast is hidden.",
        tone: "info",
        action: {
          label: "Undo",
          run: () => {
            api
              .undoRewind(out.rewind_id)
              .then(() => {
                refreshList();
                setNotice(null);
                openMission(m.id);
              })
              .catch((err) => showError(errorMessage(err, "Couldn't undo that rewind.")));
          },
        },
      });
    } catch (err) {
      showError(errorMessage(err, "Couldn't rewind. Try again."));
    }
  };

  // Fork from a notebook: the new notebook opens with the forked message in its box.
  const openForked = (id: string, prefill: string) => {
    offerDraft(id, prefill);
    openNotebook(id);
    refreshList();
    setNotice({ id: Date.now(), message: "Forked. The original notebook is unchanged.", tone: "info", ttlMs: 4000 });
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
  const uploading = attachments.some((a) => a.status === "uploading");
  const blockedReason = noKeys
    ? "Prophet needs a model key before it can start."
    : waitingOnYou
      ? "Answer the question above to continue."
      : current && isWorking(current.status)
        ? "Prophet is working. Press Stop to interrupt it."
        : uploading
          ? "Adding your files. You can ask once they are done."
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
      onRename={(kind, id, title) => void renameItem(kind, id, title)}
      onPin={(kind, id, pinned) => void pinItem(kind, id, pinned)}
      onArchive={archiveItem}
      onDelete={(kind, id) => void deleteItem(kind, id)}
    />
  );

  return (
    <div className="flex h-dvh bg-night-0 text-sand-0">
      <aside className={cn("w-72 shrink-0 flex-col gap-5 border-r border-sand-2/20 bg-night-1/50 p-4", sidebarHidden ? "hidden" : "hidden lg:flex")} inert={navOpen}>
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2.5 px-2">
            <img src={logo} alt="" aria-hidden="true" className="size-8 shrink-0 rounded-md object-cover" />
            <h1 className="font-display text-2xl font-semibold">Prophet</h1>
          </div>
          <div className="flex items-center">
            <IconButton label="Settings" icon={<SettingsIcon aria-hidden="true" className="size-5" />} onClick={() => setSettingsOpen(true)} />
            <IconButton label="Hide" aria-expanded={true} icon={<PanelLeftClose aria-hidden="true" className="size-5" />} onClick={() => setSidebarHidden(true)} />
          </div>
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
              <span className="flex items-center gap-2.5 px-2">
                <img src={logo} alt="" aria-hidden="true" className="size-8 shrink-0 rounded-md object-cover" />
                <span className="font-display text-2xl font-semibold">Prophet</span>
              </span>
              <div className="flex items-center">
                <IconButton label="Settings" icon={<SettingsIcon aria-hidden="true" className="size-5" />} onClick={() => setSettingsOpen(true)} />
                <IconButton ref={closeButton} label="Close" icon={<X aria-hidden="true" className="size-5" />} onClick={() => setNavOpen(false)} />
              </div>
            </div>
            <div className="min-h-0 flex-1">{sidebar}</div>
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col" inert={navOpen}>
        {sidebarHidden && (
          <div className="hidden items-center border-b border-sand-2/20 px-3 py-2 lg:flex">
            <IconButton label="Show" icon={<PanelLeft aria-hidden="true" className="size-5" />} onClick={() => setSidebarHidden(false)} />
          </div>
        )}
        <header className="flex items-center justify-between gap-2 border-b border-sand-2/20 px-3 py-2 lg:hidden">
          <IconButton ref={menuButton} label="Open" aria-expanded={navOpen} icon={<Menu aria-hidden="true" className="size-5" />} onClick={() => setNavOpen(true)} />
          <span className="flex items-center gap-2"><img src={logo} alt="" aria-hidden="true" className="size-7 rounded-md object-cover" /><span className="font-display text-xl font-semibold">Prophet</span></span>
          <div className="flex items-center">
            <IconButton label="Notebook" icon={<Plus aria-hidden="true" className="size-5" />} onClick={() => void newNotebook()} />
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
                <IconButton label="Dismiss" icon={<X aria-hidden="true" className="size-5" />} onClick={() => setNotice(null)} />
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
              models={catalog.selectable}
              catalogStatus={catalog.status}
              model={model}
              onSelectModel={chooseModel}
              onOpenSettings={() => setSettingsOpen(true)}
              onOpenNotebook={openForked}
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
                    <p className="text-base text-sand-1">
                      Start a notebook to add spreadsheets and ask questions about them, or ask a quick forecast below.
                    </p>
                    <div>
                      <Button onClick={() => void newNotebook()}>
                        <Plus aria-hidden="true" className="size-4" />
                        Notebook
                      </Button>
                    </div>
                  </div>
                  <ul className="grid gap-3 sm:grid-cols-2" aria-label="Quick actions">
                    <QuickCard
                      icon={<Upload aria-hidden="true" className="size-4" />}
                      title="Forecast from files"
                      body="Add a spreadsheet or a whole folder, then ask for the next period."
                      onClick={() => fileInput.current?.click()}
                    />
                    <QuickCard
                      icon={<BookOpen aria-hidden="true" className="size-4" />}
                      title="Work in a notebook"
                      body="Keep files together and ask follow-up questions over time."
                      onClick={() => void newNotebook()}
                    />
                    <QuickCard
                      icon={<BarChart3 aria-hidden="true" className="size-4" />}
                      title={SKILLS[1].title}
                      body={SKILLS[1].description}
                      onClick={() => setDraft(SKILLS[1].prompt)}
                    />
                    <QuickCard
                      icon={<ScanSearch aria-hidden="true" className="size-4" />}
                      title={SKILLS[2].title}
                      body={SKILLS[2].description}
                      onClick={() => setDraft(SKILLS[2].prompt)}
                    />
                  </ul>
                </section>
              ) : current ? (
                <Turn
                  mission={current}
                  tools={tools}
                  runs={runs}
                  onAnswered={refresh}
                  onRetry={(g, ids) => void start(g, ids).catch((err) => showError(errorMessage(err, "Couldn't start the forecast. Try again.")))}
                  checkpoint={{
                    onFork: () => forkForecast(current),
                    onRewind: () => rewindForecast(current),
                    rewindBlocked: isWorking(current.status) || current.status === "waiting_user" ? "Rewind is off while Prophet is working. Press Stop first." : null,
                  }}
                  onEdit={(g, ds) => {
                    setDraft(g);
                    // The files come back as attachments, so the person can see them and send them again.
                    setAttachments((prev) => [
                      ...prev,
                      ...ds.map((d) => ({ key: `att-${++attachSeq.current}`, name: d.name, status: "ready" as const, datasetId: d.id })),
                    ]);
                  }}
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
                onSend={(text) => start(text, readyDatasetIds())}
                quick={{
                  attachments,
                  tools: Object.values(tools),
                  skills: SKILLS,
                  onRemoveAttachment: removeAttachment,
                  onAddFiles: addAttachments,
                  onOpenFiles: () => fileInput.current?.click(),
                  onOpenFolder: () => folderInput.current?.click(),
                }}
                busy={sending}
                blockedReason={blockedReason}
                working={current !== null && isWorking(current.status)}
                onStop={() => void stopForecast()}
                stopping={stopping}
                footer={
                  <ComposerControls
                    models={catalog.selectable}
                    catalogStatus={catalog.status}
                    selected={model}
                    onSelect={chooseModel}
                    onOpenSettings={() => setSettingsOpen(true)}
                  />
                }
                initial={draft}
                placeholder={selectedId === null ? "Describe a quick forecast" : "Describe another forecast"}
              />
            </div>
          </div>
        )}
      </div>

      {/* Hidden pickers behind the composer's + menu and the start-screen cards. Folders use the browser's folder picker. */}
      <input
        ref={fileInput}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          addAttachments(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      <input
        ref={folderInput}
        type="file"
        multiple
        className="hidden"
        {...({ webkitdirectory: "" } as Record<string, string>)}
        onChange={(e) => {
          addAttachments(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />

      <SettingsModal
        open={settingsOpen}
        onClose={closeSettings}
        onDataChanged={() => {
          setSelectedId(null);
          setSelectedNotebookId(null);
          refreshList();
          api.health().then(setHealth).catch(() => undefined);
          catalog.reload();
        }}
      />
    </div>
  );
}

// Start-screen action card: an icon, a short title and one line of help. The whole card is the button.
function QuickCard({ icon, title, body, onClick }: { icon: ReactNode; title: string; body: string; onClick: () => void }) {
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className="group grid h-full min-h-28 w-full content-start gap-2 rounded-xl border border-sand-2/25 bg-night-1 p-4 text-left transition-colors hover:border-ember/60 hover:bg-night-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ember"
      >
        <span className="grid size-8 place-items-center rounded-lg bg-ember-soft text-ember">{icon}</span>
        <span className="text-base font-semibold text-sand-0">{title}</span>
        <span className="text-caption text-sand-1">{body}</span>
      </button>
    </li>
  );
}
