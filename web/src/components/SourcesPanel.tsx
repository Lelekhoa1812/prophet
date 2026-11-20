import { AlertCircle, CheckCircle2, ChevronRight, Download, FileSpreadsheet, Plus, X } from "lucide-react";
import { useRef, useState, type DragEvent, type ReactNode } from "react";
import { api, tableDownloadUrl, type KeptTable, type Source } from "../lib/api";
import { errorMessage, truncate } from "../lib/format";
import { cn, formatBytes } from "../lib/utils";
import { Button, Spinner } from "./ui";

type UploadNote = { id: number; name: string; message: string };

// Every file in the notebook, with its own state. A file that cannot be used says why, in words, and is never dropped.
export function SourcesPanel({
  notebookId,
  sources,
  tables,
  onChanged,
}: {
  notebookId: string;
  sources: Source[];
  tables: KeptTable[];
  onChanged: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<{ done: number; total: number; name: string } | null>(null);
  const [notes, setNotes] = useState<UploadNote[]>([]);
  const [busyId, setBusyId] = useState<string | null>(null);

  async function addFiles(files: File[]) {
    if (files.length === 0) return;
    const failed: UploadNote[] = [];
    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      setUploading({ done: i, total: files.length, name: file.name });
      try {
        await api.uploadSource(notebookId, file);
      } catch (err) {
        failed.push({ id: Date.now() + i, name: file.name, message: errorMessage(err, "Couldn't add this file.") });
      }
    }
    setUploading(null);
    setNotes((prev) => [...failed, ...prev].slice(0, 6));
    onChanged();
  }

  function onDrop(e: DragEvent<HTMLElement>) {
    e.preventDefault();
    setDragging(false);
    void addFiles(Array.from(e.dataTransfer.files));
  }

  async function toggle(s: Source) {
    setBusyId(s.id);
    try {
      await api.patchSource(notebookId, s.id, { included: !s.included });
      onChanged();
    } catch (err) {
      setNotes((prev) => [{ id: Date.now(), name: s.name, message: errorMessage(err, "Couldn't change this file.") }, ...prev]);
    } finally {
      setBusyId(null);
    }
  }

  async function remove(s: Source) {
    setBusyId(s.id);
    try {
      await api.patchSource(notebookId, s.id, { removed: true });
      onChanged();
    } catch (err) {
      setNotes((prev) => [{ id: Date.now(), name: s.name, message: errorMessage(err, "Couldn't remove this file.") }, ...prev]);
    } finally {
      setBusyId(null);
    }
  }

  const usable = sources.filter((s) => s.included).length;

  return (
    <section
      aria-label="Sources"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
      }}
      onDrop={onDrop}
      className={cn("grid gap-4 rounded-xl border p-4 transition-colors", dragging ? "border-ember bg-night-2" : "border-sand-2/25 bg-night-1")}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="grid gap-0.5">
          <h2 className="text-lg font-bold">Sources</h2>
          <p className="text-[0.9375rem] text-sand-2">
            {sources.length === 0
              ? "Add spreadsheets or data files. Drop them here or use Add files."
              : `${usable} of ${sources.length} in use for questions. Only files in use are read.`}
          </p>
        </div>
        <input
          ref={input}
          type="file"
          multiple
          className="sr-only"
          tabIndex={-1}
          aria-hidden="true"
          onChange={(e) => {
            void addFiles(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
        <Button variant="secondary" onClick={() => input.current?.click()} disabled={uploading !== null}>
          <Plus aria-hidden="true" className="size-4" />
          Add files
        </Button>
      </div>

      {uploading && (
        <p role="status" className="flex items-center gap-2 text-[0.9375rem] text-sand-1">
          <Spinner />
          Adding {uploading.name} ({uploading.done + 1} of {uploading.total})
        </p>
      )}

      {notes.length > 0 && (
        <ul className="grid gap-2" aria-label="File messages">
          {notes.map((n) => (
            <li key={n.id} role="alert" className="flex items-start justify-between gap-3 rounded-lg border border-bad/40 p-3 text-[0.9375rem]">
              <span className="min-w-0 [overflow-wrap:anywhere]">
                <span className="font-semibold text-sand-0">{n.name}: </span>
                <span className="text-bad">{n.message}</span>
              </span>
              <button
                type="button"
                aria-label={`Dismiss message about ${n.name}`}
                onClick={() => setNotes((prev) => prev.filter((x) => x.id !== n.id))}
                className="grid size-9 shrink-0 place-items-center rounded-md text-sand-1 hover:bg-night-2 hover:text-sand-0"
              >
                <X aria-hidden="true" className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {sources.length > 0 && (
        <ul className="grid gap-2">
          {sources.map((s) => (
            <SourceRow key={s.id} source={s} busy={busyId === s.id} onToggle={() => void toggle(s)} onRemove={() => void remove(s)} />
          ))}
        </ul>
      )}

      {tables.length > 0 && (
        <div className="grid gap-2 border-t border-sand-2/20 pt-4">
          <h3 className="text-[0.9375rem] font-semibold text-sand-2">Kept tables</h3>
          <ul className="grid gap-2">
            {tables.map((t) => (
              <li key={t.id} className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-night-0/40 p-3">
                <span className="grid min-w-0 gap-0.5">
                  <span className="font-mono text-base text-sand-0 [overflow-wrap:anywhere]">{t.name}</span>
                  <span className="text-[0.9375rem] text-sand-2">
                    Version {t.version} · {t.rows.toLocaleString("en-AU")} rows · {t.columns.length} columns
                    {t.description ? ` · ${truncate(t.description, 120)}` : ""}
                  </span>
                </span>
                <a
                  href={tableDownloadUrl(notebookId, t.name)}
                  download={`${t.name}.csv`}
                  className="inline-flex min-h-10 items-center gap-2 rounded-lg px-3 text-[0.9375rem] text-sand-0 hover:bg-night-2"
                >
                  <Download aria-hidden="true" className="size-4" />
                  Download CSV
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function SourceRow({ source: s, busy, onToggle, onRemove }: { source: Source; busy: boolean; onToggle: () => void; onRemove: () => void }) {
  const rows = s.card?.sheets.reduce((sum, sh) => sum + sh.rows, 0) ?? 0;
  const firstIssue = s.card?.sheets.flatMap((sh) => sh.issues)[0];

  let state: { icon: ReactNode; text: string; tone: string };
  switch (s.status) {
    case "profiling":
      state = { icon: <Spinner />, text: "Checking the file", tone: "text-sand-1" };
      break;
    case "ready":
      state = { icon: <CheckCircle2 aria-hidden="true" className="size-4" />, text: `Ready · ${rows.toLocaleString("en-AU")} rows`, tone: "text-ok" };
      break;
    case "needs_attention":
      state = { icon: <AlertCircle aria-hidden="true" className="size-4" />, text: "Needs attention", tone: "text-ember" };
      break;
    default:
      state = { icon: <AlertCircle aria-hidden="true" className="size-4" />, text: "Couldn't read this file", tone: "text-bad" };
  }

  return (
    <li className={cn("grid gap-2 rounded-lg border border-sand-2/20 p-3", !s.included && "opacity-70")}>
      <div className="flex items-start gap-3">
        <FileSpreadsheet aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-sand-2" />
        <div className="grid min-w-0 flex-1 gap-0.5">
          <span className="text-base font-semibold text-sand-0 [overflow-wrap:anywhere]">{s.name}</span>
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.9375rem] text-sand-2">
            <span>{formatBytes(s.bytes)}</span>
            <span aria-hidden="true">·</span>
            <span className={cn("inline-flex items-center gap-1.5", state.tone)}>
              {state.icon}
              {state.text}
            </span>
            {s.duplicate && <span>· already in this notebook</span>}
          </span>
          {s.status === "needs_attention" && firstIssue && (
            <span className="text-[0.9375rem] text-sand-1 [overflow-wrap:anywhere]">{truncate(firstIssue, 220)}</span>
          )}
          {(s.status === "failed" || s.status === "needs_attention") && s.error && (
            <span className="text-[0.9375rem] text-bad [overflow-wrap:anywhere]">{truncate(s.error, 220)}</span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 pl-8">
        <label className="inline-flex min-h-10 cursor-pointer items-center gap-2 text-[0.9375rem] text-sand-0">
          <input
            type="checkbox"
            checked={s.included}
            disabled={busy}
            onChange={onToggle}
            className="size-5 accent-[var(--color-ember)]"
          />
          Use in questions
        </label>
        <Button variant="ghost" size="sm" onClick={onRemove} disabled={busy} aria-label={`Remove ${s.name}`}>
          Remove
        </Button>
      </div>

      {s.card && s.card.sheets.length > 0 && (
        <details className="pl-8">
          <summary className="inline-flex min-h-10 cursor-pointer items-center gap-1 text-[0.9375rem] text-sand-2 hover:text-sand-0">
            <ChevronRight aria-hidden="true" className="size-3.5 transition-transform [details[open]>summary_&]:rotate-90" />
            What Prophet found
          </summary>
          <div className="mt-2 grid gap-3 text-[0.9375rem]">
            {s.card.sheets.map((sh, i) => (
              <div key={`${sh.sheet ?? "table"}-${i}`} className="grid gap-1">
                <span className="font-semibold text-sand-0">
                  {sh.sheet ? `Sheet “${sh.sheet}”` : "Table"} · {sh.rows.toLocaleString("en-AU")} rows
                  {sh.header_row ? ` · header on row ${sh.header_row}` : ""}
                </span>
                {sh.columns.length > 0 && (
                  <span className="text-sand-1 [overflow-wrap:anywhere]">
                    {sh.columns.slice(0, 12).join(" · ")}
                    {more(sh) > 0 ? ` · and ${more(sh)} more` : ""}
                  </span>
                )}
                {sh.issues.map((issue) => (
                  <span key={issue} className="text-ember [overflow-wrap:anywhere]">
                    {issue}
                  </span>
                ))}
              </div>
            ))}
            {s.card.hidden_sheets && s.card.hidden_sheets.length > 0 && (
              <span className="text-sand-2">Hidden sheets not read: {s.card.hidden_sheets.join(", ")}</span>
            )}
          </div>
        </details>
      )}
    </li>
  );
}

// Columns the card did not list: those cut to keep the card short, plus any left out by the profiler.
function more(sh: { columns: string[]; columns_omitted: number }): number {
  return Math.max(0, sh.columns.length - 12) + sh.columns_omitted;
}
