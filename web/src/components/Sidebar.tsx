import { Archive, BookOpen, Pencil, Pin, PinOff, Plus, Search, Sparkles, Trash2, X } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import type { MissionRow, NotebookRow } from "../lib/api";
import { relativeTime, truncate } from "../lib/format";
import { cn, isWorking, statusInfo } from "../lib/utils";
import { ActionMenu, type MenuItem } from "./ActionMenu";
import { Button, IconButton, StatusBadge } from "./ui";

export type ItemKind = "notebook" | "forecast";

// Notebooks and quick forecasts share one row pattern: the name, a quiet line of facts, and an ellipsis menu.
// Pinned items rise into their own section. Rename is inline; delete asks first because it cannot be undone.
export function Sidebar({
  notebooks,
  missions,
  selectedNotebookId,
  selectedMissionId,
  onSelectNotebook,
  onSelectMission,
  onNewNotebook,
  onNewForecast,
  onRename,
  onPin,
  onArchive,
  onDelete,
}: {
  notebooks: NotebookRow[];
  missions: MissionRow[];
  selectedNotebookId: string | null;
  selectedMissionId: string | null;
  onSelectNotebook: (id: string) => void;
  onSelectMission: (id: string) => void;
  onNewNotebook: () => void;
  onNewForecast: () => void;
  onRename: (kind: ItemKind, id: string, title: string) => void;
  onPin: (kind: ItemKind, id: string, pinned: boolean) => void;
  onArchive: (kind: ItemKind, id: string) => void;
  onDelete: (kind: ItemKind, id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [renaming, setRenaming] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<{ kind: ItemKind; id: string; title: string } | null>(null);
  // The sidebar renders twice (desktop column and mobile drawer), so each copy gets its own id prefix.
  const uid = useId();
  const notebooksId = `${uid}-notebooks`;
  const forecastsId = `${uid}-forecasts`;
  const pinnedId = `${uid}-pinned`;

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? missions.filter((m) => (m.title || m.goal).toLowerCase().includes(q)) : missions;
  }, [missions, query]);

  const pinnedNotebooks = notebooks.filter((n) => n.pinned);
  const pinnedForecasts = shown.filter((m) => m.pinned);
  const hasPinned = pinnedNotebooks.length + pinnedForecasts.length > 0;
  const plainNotebooks = notebooks.filter((n) => !n.pinned);
  const plainForecasts = shown.filter((m) => !m.pinned);

  const menuFor = (kind: ItemKind, id: string, pinned: boolean, title: string): MenuItem[] => [
    { key: "pin", label: pinned ? "Unpin" : "Pin", icon: pinned ? <PinOff className="size-4" /> : <Pin className="size-4" />, onSelect: () => onPin(kind, id, !pinned) },
    { key: "rename", label: "Rename", icon: <Pencil className="size-4" />, onSelect: () => setRenaming(`${kind}:${id}`) },
    { key: "archive", label: "Archive", icon: <Archive className="size-4" />, onSelect: () => onArchive(kind, id) },
    { key: "delete", label: "Delete", tone: "danger", icon: <Trash2 className="size-4" />, onSelect: () => setConfirming({ kind, id, title }) },
  ];

  const renderNotebook = (n: NotebookRow) => {
    const active = n.id === selectedNotebookId;
    return (
      <Row
        key={`nb-${n.id}`}
        active={active}
        renaming={renaming === `notebook:${n.id}`}
        title={n.title}
        pinned={n.pinned}
        icon={<BookOpen aria-hidden="true" className="size-4 shrink-0 text-sand-2" />}
        meta={
          <>
            <span className="tabular-nums">
              {n.sources} {n.sources === 1 ? "file" : "files"} · {n.turns} {n.turns === 1 ? "question" : "questions"}
            </span>
            {n.status && isWorking(n.status) && <StatusBadge status={n.status} />}
          </>
        }
        onSelect={() => onSelectNotebook(n.id)}
        onRename={(t) => {
          setRenaming(null);
          if (t && t !== n.title) onRename("notebook", n.id, t);
        }}
        onCancelRename={() => setRenaming(null)}
        menu={menuFor("notebook", n.id, n.pinned, n.title)}
        menuLabel={`Actions for notebook ${n.title}`}
      />
    );
  };

  const renderForecast = (m: MissionRow) => {
    const active = m.id === selectedMissionId;
    const title = m.title || m.goal;
    const tone = statusInfo(m.status).tone;
    return (
      <Row
        key={`fc-${m.id}`}
        active={active}
        renaming={renaming === `forecast:${m.id}`}
        title={title}
        pinned={m.pinned}
        twoLine
        meta={
          <>
            <span className="tabular-nums">{relativeTime(m.updated || m.created)}</span>
            {(tone !== "ok" || isWorking(m.status)) && <StatusBadge status={m.status} />}
          </>
        }
        onSelect={() => onSelectMission(m.id)}
        onRename={(t) => {
          setRenaming(null);
          if (t && t !== title) onRename("forecast", m.id, t);
        }}
        onCancelRename={() => setRenaming(null)}
        menu={menuFor("forecast", m.id, m.pinned, title)}
        menuLabel={`Actions for ${truncate(title, 40)}`}
      />
    );
  };

  return (
    <nav aria-label="Workspace" className="flex h-full min-h-0 flex-col gap-4">
      <Button onClick={onNewNotebook} className="w-full justify-start">
        <Plus aria-hidden="true" className="size-4" />
        Notebook
      </Button>

      <div className="grid min-h-0 flex-1 content-start gap-5 overflow-y-auto">
        {hasPinned && (
          <section className="grid gap-1" aria-labelledby={pinnedId}>
            <h2 id={pinnedId} className="flex items-center gap-2 px-2 pb-1 text-caption font-semibold text-sand-2">
              <Pin aria-hidden="true" className="size-3.5" />
              Pinned
            </h2>
            <ul className="grid gap-1">
              {pinnedNotebooks.map((n) => (
                <li key={`pn-${n.id}`}>{renderNotebook(n)}</li>
              ))}
              {pinnedForecasts.map((m) => (
                <li key={`pf-${m.id}`}>{renderForecast(m)}</li>
              ))}
            </ul>
          </section>
        )}

        <section className="grid gap-1" aria-labelledby={notebooksId}>
          <h2 id={notebooksId} className="px-2 pb-1 text-caption font-semibold text-sand-2">
            Notebooks
          </h2>
          {notebooks.length === 0 && (
            <p className="px-2 text-caption text-sand-2">Add spreadsheets and ask questions about them. Notebooks keep your files.</p>
          )}
          <ul className="grid gap-1">{plainNotebooks.map((n) => <li key={n.id}>{renderNotebook(n)}</li>)}</ul>
        </section>

        <section className="grid gap-1" aria-labelledby={forecastsId}>
          <div className="flex items-center justify-between gap-2 px-2 pb-1">
            <h2 id={forecastsId} className="text-caption font-semibold text-sand-2">
              Quick forecasts
            </h2>
            <IconButton label="Quick" icon={<Sparkles aria-hidden="true" className="size-4" />} onClick={onNewForecast} className="size-9" />
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

          {missions.length === 0 && <p className="px-2 text-caption text-sand-2">Nothing yet.</p>}
          {missions.length > 0 && shown.length === 0 && <p className="px-2 text-caption text-sand-2">No forecasts match “{query}”.</p>}
          <ul className="grid gap-1">{plainForecasts.map((m) => <li key={m.id}>{renderForecast(m)}</li>)}</ul>
        </section>
      </div>

      {confirming && (
        <ConfirmDelete
          title={truncate(confirming.title, 60)}
          kind={confirming.kind}
          onCancel={() => setConfirming(null)}
          onConfirm={() => {
            const { kind, id } = confirming;
            setConfirming(null);
            onDelete(kind, id);
          }}
        />
      )}
    </nav>
  );
}

function Row({
  active,
  renaming,
  title,
  pinned,
  icon,
  meta,
  twoLine,
  onSelect,
  onRename,
  onCancelRename,
  menu,
  menuLabel,
}: {
  active: boolean;
  renaming: boolean;
  title: string;
  pinned: boolean;
  icon?: ReactNode;
  meta: ReactNode;
  twoLine?: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onCancelRename: () => void;
  menu: MenuItem[];
  menuLabel: string;
}) {
  const [draft, setDraft] = useState(title);
  const input = useRef<HTMLInputElement>(null);
  const skipBlur = useRef(false);

  useEffect(() => {
    if (renaming) {
      setDraft(title);
      input.current?.focus();
      input.current?.select();
    }
    // Only when rename starts; a title update mid-edit must not overwrite the draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [renaming]);

  if (renaming) {
    return (
      <form
        className={cn("grid gap-1 rounded-lg px-3 py-2", active && "bg-night-2")}
        onSubmit={(e) => {
          e.preventDefault();
          skipBlur.current = true;
          onRename(draft.trim().slice(0, 120));
        }}
      >
        <label className="sr-only" htmlFor={`rename-${title}`}>
          Name
        </label>
        <input
          id={`rename-${title}`}
          ref={input}
          value={draft}
          maxLength={120}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={() => {
            if (skipBlur.current) {
              skipBlur.current = false;
              return;
            }
            onRename(draft.trim().slice(0, 120));
          }}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              skipBlur.current = true;
              onCancelRename();
            }
          }}
          className="min-h-10 w-full rounded-md border border-ember bg-night-0/60 px-2 text-base text-sand-0 focus:outline-none"
        />
        <span className="text-micro text-sand-2">Enter to save · Esc to cancel</span>
      </form>
    );
  }

  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onSelect}
        aria-current={active ? "true" : undefined}
        className={cn(
          "grid min-h-11 w-full gap-0.5 rounded-lg py-2 pr-11 pl-3 text-left transition-colors hover:bg-night-2",
          active && "bg-night-2",
        )}
      >
        <span className={cn("text-base text-sand-0 [overflow-wrap:anywhere]", twoLine ? "line-clamp-2" : "line-clamp-1")}>
          <span className="flex items-center gap-2">
            {icon}
            <span className={cn(twoLine ? "line-clamp-2" : "line-clamp-1")}>{truncate(title, 90)}</span>
          </span>
        </span>
        <span className="flex flex-wrap items-center gap-x-2 gap-y-0.5 pl-0 text-caption text-sand-2">{meta}</span>
      </button>
      <div className="absolute top-1.5 right-1.5 z-10">
        <ActionMenu
          label={menuLabel}
          items={menu.map((item) => (item.key === "pin" ? { ...item, label: pinned ? "Unpin" : "Pin" } : item))}
          className="opacity-100 transition-opacity focus-within:opacity-100 max-lg:opacity-100 lg:opacity-0 lg:group-hover:opacity-100"
        />
      </div>
    </div>
  );
}

// Delete asks first, names the item, and says it cannot be undone. Escape and Cancel leave everything as it was.
function ConfirmDelete({
  title,
  kind,
  onCancel,
  onConfirm,
}: {
  title: string;
  kind: ItemKind;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const heading = useId();
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancel.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="fixed inset-0 z-[60] grid place-items-center bg-black/60 p-4">
      <div role="alertdialog" aria-modal="true" aria-labelledby={heading} className="grid w-full max-w-sm gap-4 rounded-xl border border-sand-2/30 bg-night-1 p-5 shadow-2xl">
        <div className="flex items-start justify-between gap-3">
          <h3 id={heading} className="text-base font-semibold text-sand-0">
            Delete {kind === "notebook" ? "this notebook" : "this forecast"}?
          </h3>
          <IconButton label="Close" icon={<X aria-hidden="true" className="size-4" />} onClick={onCancel} className="size-8" />
        </div>
        <p className="text-caption text-sand-1">
          “{title}” and its {kind === "notebook" ? "files and questions" : "results"} will be removed. This cannot be undone.
        </p>
        <div className="flex justify-end gap-2">
          <Button ref={cancel} variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <button
            type="button"
            onClick={onConfirm}
            className="inline-flex min-h-10 items-center justify-center gap-2 rounded-lg bg-bad px-4 text-base font-semibold text-night-0 hover:brightness-110"
          >
            <Trash2 aria-hidden="true" className="size-4" />
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}
