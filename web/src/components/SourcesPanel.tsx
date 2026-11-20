import { AlertCircle, CheckCircle2, ChevronRight, Download, FileSpreadsheet } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api, tableDownloadUrl, type KeptTable, type Source } from "../lib/api";
import { errorMessage, truncate } from "../lib/format";
import type { Uploads } from "../lib/useUploads";
import { cn, formatBytes } from "../lib/utils";
import { Button, Spinner } from "./ui";

// Every file in the notebook, with its own state. A file that cannot be used says why, in words, and is never dropped.
// Adding files happens elsewhere (drop anywhere, or the attach button); this is where you decide which ones are read.
export function SourcesPanel({
  notebookId,
  sources,
  tables,
  uploads,
  onChanged,
}: {
  notebookId: string;
  sources: Source[];
  tables: KeptTable[];
  uploads: Uploads;
  onChanged: () => void;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);

  async function change(s: Source, body: { included?: boolean; removed?: boolean }, fallback: string) {
    setBusyId(s.id);
    try {
      await api.patchSource(notebookId, s.id, body);
      onChanged();
    } catch (err) {
      uploads.note(s.name, errorMessage(err, fallback));
    } finally {
      setBusyId(null);
    }
  }

  const usable = sources.filter((s) => s.included).length;

  return (
    <section aria-label="Files in this notebook" className="grid gap-4 rounded-2xl border border-sand-2/20 bg-night-1/70 p-4 sm:p-5">
      <p className="text-[0.9375rem] text-sand-2">
        {usable} of {sources.length} in use for questions. Only files in use are read.
      </p>

      {sources.length > 0 && (
        <ul className="grid gap-2">
          {sources.map((s) => (
            <SourceRow
              key={s.id}
              source={s}
              busy={busyId === s.id}
              onToggle={() => void change(s, { included: !s.included }, "Couldn't change this file.")}
              onRemove={() => void change(s, { removed: true }, "Couldn't remove this file.")}
            />
          ))}
        </ul>
      )}

      {tables.length > 0 && (
        <div className="grid gap-2 border-t border-sand-2/20 pt-4">
          <h3 className="text-[0.9375rem] font-semibold text-sand-2">Kept tables</h3>
          <ul className="grid gap-2">
            {tables.map((t) => (
              <li key={t.id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-night-0/40 p-3">
                <span className="grid min-w-0 gap-0.5">
                  <span className="text-base font-semibold text-sand-0 [overflow-wrap:anywhere]">{t.name}</span>
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
