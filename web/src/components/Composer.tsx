import { ArrowUp, Square } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { cn } from "../lib/utils";
import { Spinner } from "./ui";

const MAX = 2000;

// One message box, like the one people already know: attach on the left, send on the right.
// Enter sends, Shift+Enter adds a line, and Enter during IME composition is left alone. While the agent works the
// send button becomes Stop, so stopping is where the eye already is. The hint under the box appears only when it helps.
export function Composer({
  onSend,
  busy,
  blockedReason,
  initial,
  placeholder,
  leading,
  working = false,
  onStop,
  stopping = false,
}: {
  onSend: (text: string) => Promise<void> | void;
  busy: boolean;
  blockedReason: string | null;
  initial?: string;
  placeholder: string;
  leading?: ReactNode;
  working?: boolean;
  onStop?: () => void;
  stopping?: boolean;
}) {
  const [text, setText] = useState(initial ?? "");
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (initial !== undefined) {
      setText(initial);
      ref.current?.focus();
    }
  }, [initial]);

  // Grow with the text up to about eight lines, then scroll inside the box.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [text]);

  const trimmed = text.trim();
  const canSend = trimmed.length >= 3 && trimmed.length <= MAX && !busy && !blockedReason;
  const showStop = working && onStop !== undefined;

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    if (showStop || !canSend) return;
    setError(null);
    try {
      await onSend(trimmed);
      setText("");
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : "Couldn't send that. Try again.");
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    }
  }

  const note = error ?? blockedReason ?? (trimmed.length > 0 ? "Enter to send · Shift+Enter for a new line" : null);

  return (
    <form onSubmit={submit} className="grid gap-1.5">
      <div
        className={cn(
          "flex items-end gap-1.5 rounded-[1.75rem] border bg-night-1 p-2 shadow-[0_6px_24px_-12px_rgba(0,0,0,0.8)] transition-colors",
          "border-sand-2/30 focus-within:border-sand-1/70",
        )}
      >
        {leading}
        <label htmlFor="composer" className="sr-only">
          Message to Prophet
        </label>
        <textarea
          id="composer"
          ref={ref}
          value={text}
          rows={1}
          maxLength={MAX}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          className="max-h-56 min-h-11 flex-1 resize-none bg-transparent px-2 py-2.5 text-[1.0625rem] leading-snug text-sand-0 placeholder:text-sand-2 focus:outline-none"
        />
        {showStop ? (
          <button
            type="button"
            onClick={onStop}
            disabled={stopping}
            aria-label={stopping ? "Stopping" : "Stop"}
            title="Stop"
            className="grid size-11 shrink-0 place-items-center rounded-full bg-sand-0 text-night-0 transition-colors hover:bg-sand-1 disabled:opacity-60"
          >
            {stopping ? <Spinner /> : <Square aria-hidden="true" className="size-4 fill-current" />}
          </button>
        ) : (
          <button
            type="submit"
            disabled={!canSend}
            aria-label={busy ? "Sending" : "Send"}
            title="Send"
            className="grid size-11 shrink-0 place-items-center rounded-full bg-ember text-night-0 transition-colors hover:bg-ember-strong disabled:cursor-not-allowed disabled:bg-night-3 disabled:text-sand-2"
          >
            {busy ? <Spinner /> : <ArrowUp aria-hidden="true" className="size-5" />}
          </button>
        )}
      </div>
      <p className="min-h-5 px-4 text-[0.9375rem] text-sand-2" role={error ? "alert" : undefined}>
        {error ? <span className="text-bad">{error}</span> : note}
      </p>
    </form>
  );
}
