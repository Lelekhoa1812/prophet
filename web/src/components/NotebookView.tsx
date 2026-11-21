import { AlertCircle, CheckCircle2, ChevronDown, FileSpreadsheet, Paperclip, Upload, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type DragEvent, type ReactNode } from "react";
import { api, type Source, type ToolInfo } from "../lib/api";
import { errorMessage } from "../lib/format";
import { useNotebook } from "../lib/useNotebook";
import { useUploads } from "../lib/useUploads";
import { cn, isWorking } from "../lib/utils";
import { Composer } from "./Composer";
import { NotebookTurn } from "./NotebookTurn";
import { SourcesPanel } from "./SourcesPanel";
import { ResearchNotice, Spinner } from "./ui";

export const THREAD_SCROLL_ID = "thread-scroll";

// Starting points once a file is ready. They only fill the message box; the person can edit before sending.
const STARTERS = [
  "Give me an overview of this data",
  "What stands out or looks unusual?",
  "Forecast the next period and show the range",
];

function chipState(s: Source): { icon: ReactNode; text: string; tone: string } {
  const rows = s.card?.sheets.reduce((sum, sh) => sum + sh.rows, 0) ?? 0;
  switch (s.status) {
    case "profiling":
      return { icon: <Spinner />, text: "Checking", tone: "text-sand-1" };
    case "ready":
      return { icon: <CheckCircle2 aria-hidden="true" className="size-4" />, text: `${rows.toLocaleString("en-AU")} rows`, tone: "text-ok" };
    case "needs_attention":
      return { icon: <AlertCircle aria-hidden="true" className="size-4" />, text: "Needs attention", tone: "text-ember" };
    default:
      return { icon: <AlertCircle aria-hidden="true" className="size-4" />, text: "Couldn't read", tone: "text-bad" };
  }
}

// A notebook: its files as chips at the top, its questions as a thread, and one message box pinned below.
export function NotebookView({
  notebookId,
  tools,
  onChanged,
  onError,
}: {
  notebookId: string;
  tools: Record<string, ToolInfo>;
  onChanged: () => void;
  onError: (message: string) => void;
}) {
  const { notebook, error, refresh } = useNotebook(notebookId);
  const [stopping, setStopping] = useState(false);
  const [manage, setManage] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [draft, setDraft] = useState<string | undefined>(undefined);
  // True while a question is being sent, so the send button cannot fire a second copy of the same question.
  const [sending, setSending] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const turnCount = notebook?.turns.length ?? 0;

  const changed = useCallback(() => {
    refresh();
    onChanged();
  }, [refresh, onChanged]);
  const uploads = useUploads(notebookId, changed);

  // Follow the newest question down the page.
  useEffect(() => {
    const el = document.getElementById(THREAD_SCROLL_ID);
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
    el?.scrollTo({ top: el.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });
  }, [turnCount]);

  useEffect(() => {
    setDraft(undefined);
    setManage(false);
  }, [notebookId]);

  if (!notebook) {
    return (
      <p role="status" className="flex items-center gap-2 px-6 py-8 text-base text-sand-1">
        {error ? <span className="text-bad">{error}</span> : <><Spinner /> Opening notebook</>}
      </p>
    );
  }

  const last = notebook.turns[notebook.turns.length - 1] ?? null;
  const working = last !== null && isWorking(last.status);
  const waiting = last?.status === "waiting_user";
  const blockedReason = working
    ? "Prophet is working. Press Stop to interrupt it."
    : waiting
      ? "Answer the question above to continue."
      : null;
  const hasFiles = notebook.sources.length > 0;
  // Usable: at least one included file has been read. Checking: some file is still being read. Otherwise nothing is in use.
  const ready = notebook.sources.some((s) => s.included && (s.status === "ready" || s.status === "needs_attention"));
  const checking = notebook.sources.some((s) => s.status === "profiling");

  // The composer keeps the draft until the send succeeds; a failure stays visible in the composer (thrown) and the
  // retry path from a question card reports through the notice slot.
  async function send(goal: string) {
    setSending(true);
    try {
      await api.createTurn(notebookId, goal);
    } catch (err) {
      throw new Error(errorMessage(err, "Couldn't send that. Try again."));
    } finally {
      setSending(false);
    }
    setDraft(undefined);
    changed();
  }

  async function stop() {
    if (!last) return;
    setStopping(true);
    try {
      await api.cancel(last.id);
    } catch (err) {
      onError(errorMessage(err, "Couldn't stop Prophet. Try again."));
    } finally {
      setStopping(false);
      changed();
    }
  }

  function onDrop(e: DragEvent<HTMLElement>) {
    e.preventDefault();
    setDragging(false);
    void uploads.addFiles(Array.from(e.dataTransfer.files));
  }

  const attach = (
    <>
      <input
        ref={input}
        type="file"
        multiple
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={(e) => {
          void uploads.addFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      <button
        type="button"
        onClick={() => input.current?.click()}
        disabled={uploads.uploading !== null}
        aria-label="Add files"
        title="Add spreadsheets or data files"
        className="grid size-11 shrink-0 place-items-center rounded-full text-sand-1 transition-colors hover:bg-night-2 hover:text-sand-0 disabled:opacity-50"
      >
        {uploads.uploading ? <Spinner /> : <Paperclip aria-hidden="true" className="size-5" />}
      </button>
    </>
  );

  return (
    <div
      className="relative flex h-full min-h-0 flex-col"
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <div id={THREAD_SCROLL_ID} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto grid w-full max-w-3xl gap-6 px-4 pt-6 pb-8 sm:px-6">
          <header className="grid gap-3">
            <h1 className="text-2xl leading-tight font-bold [overflow-wrap:anywhere] sm:text-3xl">{notebook.title}</h1>

            {hasFiles && (
              <div className="flex flex-wrap items-center gap-2" aria-label="Files">
                {notebook.sources.map((s) => {
                  const st = chipState(s);
                  return (
                    <button
                      key={s.id}
                      type="button"
                      onClick={() => setManage(true)}
                      title={`${s.name} · ${st.text}${s.included ? "" : " · not in use"}`}
                      className={cn(
                        "inline-flex min-h-10 max-w-full items-center gap-2 rounded-full border border-sand-2/30 bg-night-1 px-3.5 text-[0.9375rem] text-sand-0 transition-colors hover:border-sand-1 hover:bg-night-2",
                        !s.included && "opacity-55",
                      )}
                    >
                      <FileSpreadsheet aria-hidden="true" className="size-4 shrink-0 text-sand-1" />
                      <span className="truncate">{s.name}</span>
                      <span className={cn("inline-flex shrink-0 items-center gap-1", st.tone)}>
                        {st.icon}
                        <span className="text-sand-1">{st.text}</span>
                      </span>
                    </button>
                  );
                })}
                <button
                  type="button"
                  onClick={() => setManage((v) => !v)}
                  aria-expanded={manage}
                  className="inline-flex min-h-10 items-center gap-1 rounded-full px-3 text-[0.9375rem] text-sand-1 hover:bg-night-2 hover:text-sand-0"
                >
                  Manage
                  <ChevronDown aria-hidden="true" className={cn("size-4 transition-transform", manage && "rotate-180")} />
                </button>
              </div>
            )}
          </header>

          {uploads.uploading && (
            <p role="status" className="flex items-center gap-2 text-[0.9375rem] text-sand-1">
              <Spinner />
              Adding {uploads.uploading.name} ({uploads.uploading.done + 1} of {uploads.uploading.total})
            </p>
          )}

          {uploads.notes.length > 0 && (
            <ul className="grid gap-2" aria-label="File messages">
              {uploads.notes.map((n) => (
                <li key={n.id} role="alert" className="flex items-start justify-between gap-3 rounded-xl border border-bad/40 p-3 text-[0.9375rem]">
                  <span className="min-w-0 [overflow-wrap:anywhere]">
                    <span className="font-semibold text-sand-0">{n.name}: </span>
                    <span className="text-bad">{n.message}</span>
                  </span>
                  <button
                    type="button"
                    aria-label={`Dismiss message about ${n.name}`}
                    onClick={() => uploads.dismiss(n.id)}
                    className="grid size-9 shrink-0 place-items-center rounded-lg text-sand-1 hover:bg-night-2 hover:text-sand-0"
                  >
                    <X aria-hidden="true" className="size-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {manage && hasFiles && (
            <SourcesPanel notebookId={notebookId} sources={notebook.sources} tables={notebook.tables} uploads={uploads} onChanged={changed} />
          )}

          {!hasFiles && notebook.turns.length === 0 && (
            <button
              type="button"
              onClick={() => input.current?.click()}
              className="grid justify-items-center gap-2 rounded-3xl border border-dashed border-sand-2/40 px-6 py-14 text-center transition-colors hover:border-sand-1 hover:bg-night-1"
            >
              <Upload aria-hidden="true" className="size-8 text-sand-1" />
              <span className="text-xl font-bold text-sand-0">Drop your spreadsheets here</span>
              <span className="text-base text-sand-1">or choose files. Excel, CSV, ODS, Parquet and more. Add as many as you like.</span>
            </button>
          )}

          {hasFiles && notebook.turns.length === 0 && (
            <div className="grid gap-3 pt-2">
              <p className="text-lg text-sand-1">
                {ready
                  ? "Ask anything about your files. Prophet shows the steps it took and the code it ran."
                  : checking
                    ? "Prophet is checking your files."
                    : "None of your files are in use. Include a file under Manage to ask questions."}
              </p>
              {ready && (
                <ul className="flex flex-wrap gap-2">
                  {STARTERS.map((s) => (
                    <li key={s}>
                      <button
                        type="button"
                        onClick={() => setDraft(s)}
                        className="min-h-10 rounded-full border border-sand-2/35 px-4 text-base text-sand-0 transition-colors hover:border-sand-1 hover:bg-night-2"
                      >
                        {s}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {notebook.turns.length > 0 && (
            <ol aria-label="Questions" className="grid gap-10">
              {notebook.turns.map((t) => (
                <li key={t.id}>
                  <NotebookTurn
                    missionId={t.id}
                    tools={tools}
                    onRetry={(goal) => void send(goal).catch((err) => onError(errorMessage(err, "Couldn't send that. Try again.")))}
                    onEdit={setDraft}
                  />
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <div className="px-4 pt-1 pb-3 sm:px-6">
        <div className="mx-auto w-full max-w-3xl">
          <Composer
            onSend={send}
            busy={sending}
            blockedReason={blockedReason}
            initial={draft}
            leading={attach}
            working={working}
            onStop={() => void stop()}
            stopping={stopping}
            placeholder={notebook.turns.length === 0 ? "Ask a question about your files" : "Ask a follow-up question"}
          />
          <ResearchNotice className="px-4 pt-1.5" />
        </div>
      </div>

      {dragging && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-3 z-10 grid place-items-center rounded-3xl border-2 border-dashed border-ember bg-night-0/85 text-xl font-bold text-sand-0"
        >
          Drop to add to this notebook
        </div>
      )}
    </div>
  );
}
