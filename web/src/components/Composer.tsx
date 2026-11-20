import { ArrowUp } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { cn } from "../lib/utils";
import { Spinner } from "./ui";

const MAX = 2000;

// One message box. Enter sends, Shift+Enter adds a line, and Enter during IME composition is left alone.
// While the agent works or waits for an answer the box stays readable but cannot send, and says why.
export function Composer({
  onSend,
  busy,
  blockedReason,
  initial,
  placeholder,
}: {
  onSend: (text: string) => Promise<void> | void;
  busy: boolean;
  blockedReason: string | null;
  initial?: string;
  placeholder: string;
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

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    if (!canSend) return;
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

  return (
    <form onSubmit={submit} className="grid gap-2">
      <div
        className={cn(
          "flex items-end gap-2 rounded-2xl border bg-night-1 p-2 transition-colors",
          "border-sand-2/40 focus-within:border-ember",
        )}
      >
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
          className="max-h-56 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-base leading-snug text-sand-0 placeholder:text-sand-2 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!canSend}
          aria-label={busy ? "Sending" : "Send"}
          className="grid size-10 shrink-0 place-items-center rounded-xl bg-ember text-night-0 transition-colors hover:bg-ember-strong disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? <Spinner /> : <ArrowUp aria-hidden="true" className="size-5" />}
        </button>
      </div>
      <p className="min-h-5 px-2 text-[0.9375rem] text-sand-2" role={error ? "alert" : undefined}>
        {error ? <span className="text-bad">{error}</span> : (blockedReason ?? "Enter to send · Shift+Enter for a new line")}
      </p>
    </form>
  );
}
