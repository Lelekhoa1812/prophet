import { Square } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type ToolInfo } from "../lib/api";
import { errorMessage } from "../lib/format";
import { useNotebook } from "../lib/useNotebook";
import { isWorking } from "../lib/utils";
import { Composer } from "./Composer";
import { NotebookTurn } from "./NotebookTurn";
import { SourcesPanel } from "./SourcesPanel";
import { Button, Spinner } from "./ui";

export const THREAD_SCROLL_ID = "thread-scroll";

// A notebook: its files on top, its questions as a thread, and a message box that stays in view while the thread scrolls.
export function NotebookView({
  notebookId,
  tools,
  onChanged,
}: {
  notebookId: string;
  tools: Record<string, ToolInfo>;
  onChanged: () => void;
}) {
  const { notebook, error, refresh } = useNotebook(notebookId);
  const [stopping, setStopping] = useState(false);
  const turnCount = notebook?.turns.length ?? 0;

  // Follow the newest question down the page.
  useEffect(() => {
    const el = document.getElementById(THREAD_SCROLL_ID);
    el?.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turnCount]);

  if (!notebook) {
    return (
      <p role="status" className="flex items-center gap-2 text-base text-sand-1">
        {error ? <span className="text-bad">{error}</span> : <><Spinner /> Opening notebook</>}
      </p>
    );
  }

  const last = notebook.turns[notebook.turns.length - 1] ?? null;
  const working = last !== null && isWorking(last.status);
  const waiting = last?.status === "waiting_user";
  const blockedReason = working
    ? "Prophet is still working on the last question. You can stop it, or wait."
    : waiting
      ? "Answer the question above to continue."
      : null;
  const usable = notebook.sources.filter((s) => s.included && (s.status === "ready" || s.status === "needs_attention")).length;

  async function send(goal: string) {
    try {
      await api.createTurn(notebookId, goal);
    } catch (err) {
      throw new Error(errorMessage(err, "Couldn't send that. Try again."));
    }
    refresh();
    onChanged();
  }

  async function stop() {
    if (!last) return;
    setStopping(true);
    try {
      await api.cancel(last.id);
    } finally {
      setStopping(false);
      refresh();
      onChanged();
    }
  }

  return (
    <div className="grid gap-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="grid gap-1">
          <h1 className="text-2xl leading-tight font-bold [overflow-wrap:anywhere] sm:text-3xl">{notebook.title}</h1>
          <p className="text-[0.9375rem] text-sand-2">
            {usable} {usable === 1 ? "file" : "files"} ready for questions · {notebook.turns.length}{" "}
            {notebook.turns.length === 1 ? "question" : "questions"}
          </p>
        </div>
        {working && (
          <Button variant="secondary" onClick={() => void stop()} disabled={stopping} aria-label="Stop the current question">
            <Square aria-hidden="true" className="size-4" />
            {stopping ? "Stopping" : "Stop"}
          </Button>
        )}
      </header>

      <details open className="group">
        <summary className="mb-3 inline-flex min-h-10 cursor-pointer items-center text-[0.9375rem] font-semibold text-sand-2 hover:text-sand-0">
          {notebook.sources.length === 0 ? "Sources" : `Sources (${notebook.sources.length})`}
        </summary>
        <SourcesPanel
          notebookId={notebookId}
          sources={notebook.sources}
          tables={notebook.tables}
          onChanged={() => {
            refresh();
            onChanged();
          }}
        />
      </details>

      {notebook.turns.length === 0 ? (
        <p className="text-lg text-sand-1">
          Add a spreadsheet, then ask anything about it: totals, comparisons, what changed, or what comes next. Prophet
          shows the working and where each number came from.
        </p>
      ) : (
        <ol aria-label="Questions" className="grid gap-8">
          {notebook.turns.map((t) => (
            <li key={t.id}>
              <NotebookTurn missionId={t.id} tools={tools} onRetry={(goal) => void send(goal).catch(() => {})} />
            </li>
          ))}
        </ol>
      )}

      <div className="sticky bottom-0 -mx-4 bg-night-0/95 px-4 pt-3 pb-3 backdrop-blur sm:-mx-6 sm:px-6">
        <Composer
          onSend={send}
          busy={false}
          blockedReason={blockedReason}
          placeholder={notebook.turns.length === 0 ? "Ask about your files, or describe a forecast" : "Ask a follow-up"}
        />
      </div>
    </div>
  );
}
