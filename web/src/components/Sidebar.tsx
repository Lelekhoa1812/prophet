import { BookOpen, Plus, Search, Sparkles, X } from "lucide-react";
import { useId, useMemo, useState } from "react";
import type { MissionRow, NotebookRow } from "../lib/api";
import { relativeTime, truncate } from "../lib/format";
import { cn, isWorking, statusInfo } from "../lib/utils";
import { Button, StatusBadge } from "./ui";

// Notebooks first (the main way to work), then quick forecasts. Search narrows the quick forecasts by the words in them.
export function Sidebar({
  notebooks,
  missions,
  selectedNotebookId,
  selectedMissionId,
  onSelectNotebook,
  onSelectMission,
  onNewNotebook,
  onNewForecast,
  onHideMission,
}: {
  notebooks: NotebookRow[];
  missions: MissionRow[];
  selectedNotebookId: string | null;
  selectedMissionId: string | null;
  onSelectNotebook: (id: string) => void;
  onSelectMission: (id: string) => void;
  onNewNotebook: () => void;
  onNewForecast: () => void;
  onHideMission: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  // The sidebar renders twice (desktop column and mobile drawer), so each copy gets its own id prefix.
  const uid = useId();
  const notebooksId = `${uid}-notebooks`;
  const forecastsId = `${uid}-forecasts`;
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? missions.filter((m) => m.goal.toLowerCase().includes(q)) : missions;
  }, [missions, query]);

  return (
    <nav aria-label="Workspace" className="flex h-full min-h-0 flex-col gap-4">
      <Button onClick={onNewNotebook} className="w-full justify-start">
        <Plus aria-hidden="true" className="size-4" />
        New notebook
      </Button>

      <div className="grid min-h-0 flex-1 content-start gap-5 overflow-y-auto">
        <section className="grid gap-1" aria-labelledby={notebooksId}>
          <h2 id={notebooksId} className="px-2 pb-1 text-[0.9375rem] font-semibold text-sand-2">
            Notebooks
          </h2>
          {notebooks.length === 0 && (
            <p className="px-2 text-[0.9375rem] text-sand-2">Add spreadsheets and ask questions about them. Notebooks keep your files.</p>
          )}
          <ul className="grid gap-1">
            {notebooks.map((n) => {
              const active = n.id === selectedNotebookId;
              return (
                <li key={n.id}>
                  <button
                    type="button"
                    onClick={() => onSelectNotebook(n.id)}
                    aria-current={active ? "true" : undefined}
                    className={cn("grid min-h-11 w-full gap-0.5 rounded-lg px-3 py-2 text-left transition-colors hover:bg-night-2", active && "bg-night-2")}
                  >
                    <span className="flex items-center gap-2 text-base text-sand-0">
                      <BookOpen aria-hidden="true" className="size-4 shrink-0 text-sand-2" />
                      <span className="line-clamp-1 [overflow-wrap:anywhere]">{truncate(n.title, 60)}</span>
                    </span>
                    <span className="flex flex-wrap items-center gap-x-2 text-[0.9375rem] text-sand-2">
                      <span className="tabular-nums">
                        {n.sources} {n.sources === 1 ? "file" : "files"} · {n.turns} {n.turns === 1 ? "question" : "questions"}
                      </span>
                      {n.status && isWorking(n.status) && <StatusBadge status={n.status} />}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="grid gap-1" aria-labelledby={forecastsId}>
          <div className="flex items-center justify-between gap-2 px-2 pb-1">
            <h2 id={forecastsId} className="text-[0.9375rem] font-semibold text-sand-2">
              Quick forecasts
            </h2>
            <button
              type="button"
              onClick={onNewForecast}
              className="inline-flex min-h-9 items-center gap-1 rounded-md px-2 text-[0.9375rem] text-sand-1 hover:bg-night-2 hover:text-sand-0"
            >
              <Sparkles aria-hidden="true" className="size-3.5" />
              New
            </button>
          </div>

          {missions.length > 4 && (
            <label className="relative mb-1 block">
              <span className="sr-only">Search your quick forecasts</span>
              <Search aria-hidden="true" className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-sand-2" />
              <input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search"
                className="min-h-10 w-full rounded-lg border border-sand-2/30 bg-night-0/50 pr-3 pl-9 text-base text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
              />
            </label>
          )}

          {missions.length === 0 && <p className="px-2 text-[0.9375rem] text-sand-2">Nothing yet.</p>}
          {missions.length > 0 && shown.length === 0 && (
            <p className="px-2 text-[0.9375rem] text-sand-2">No forecasts match “{query}”.</p>
          )}
          <ul className="grid gap-1">
            {shown.map((m) => {
              const active = m.id === selectedMissionId;
              const tone = statusInfo(m.status).tone;
              return (
                <li key={m.id} className="group relative">
                  {!isWorking(m.status) && (
                    <button
                      type="button"
                      aria-label={`Hide forecast: ${truncate(m.goal, 40)}`}
                      title="Hide this forecast"
                      onClick={() => onHideMission(m.id)}
                      className="absolute top-1.5 right-1.5 z-10 grid size-10 place-items-center rounded-md sm:size-8 text-sand-2 opacity-0 transition-opacity hover:bg-night-3 hover:text-sand-0 focus-visible:opacity-100 group-hover:opacity-100 max-lg:opacity-100"
                    >
                      <X aria-hidden="true" className="size-4" />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => onSelectMission(m.id)}
                    aria-current={active ? "true" : undefined}
                    className={cn("grid min-h-11 w-full gap-0.5 rounded-lg py-2 pr-10 pl-3 text-left transition-colors hover:bg-night-2", active && "bg-night-2")}
                  >
                    <span className="line-clamp-2 text-base text-sand-0 [overflow-wrap:anywhere]">{truncate(m.goal, 90)}</span>
                    <span className="flex items-center gap-2 text-[0.9375rem] text-sand-2">
                      <span className="tabular-nums">{relativeTime(m.updated || m.created)}</span>
                      {(tone !== "ok" || isWorking(m.status)) && <StatusBadge status={m.status} />}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      </div>
    </nav>
  );
}
