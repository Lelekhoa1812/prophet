import { Check, Copy, GitFork, Undo2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { copyText } from "../lib/checkpoint";
import { IconButton } from "./ui";

// What Fork and Rewind need from the page that owns the conversation. Copy needs nothing: it is the text itself.
export type Checkpoint = {
  // A new conversation from the point before this message. The original is not changed.
  onFork: () => Promise<void> | void;
  // The same conversation, back at the point before this message. Later messages are hidden, and Undo brings them back.
  onRewind: () => Promise<void> | void;
  // Why Rewind cannot be used right now (for example, Prophet is working), or null.
  rewindBlocked?: string | null;
};

// Three icon buttons under a sent message, in this order: Fork, Rewind, Copy. They show on hover and keyboard focus, and
// always on touch screens, which have no hover.
export function MessageActions({ text, checkpoint }: { text: string; checkpoint?: Checkpoint }) {
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState<"fork" | "rewind" | null>(null);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  async function copy() {
    const ok = await copyText(text);
    setCopied(ok);
    window.clearTimeout(timer.current);
    if (ok) timer.current = window.setTimeout(() => setCopied(false), 1800);
  }

  async function run(which: "fork" | "rewind", fn: () => Promise<void> | void) {
    if (busy) return;
    setBusy(which);
    try {
      await fn();
    } finally {
      setBusy(null);
    }
  }

  const rewindBlocked = checkpoint?.rewindBlocked ?? null;
  return (
    <div
      role="group"
      aria-label="Message actions"
      className="flex items-center opacity-0 transition-opacity duration-150 group-focus-within/msg:opacity-100 group-hover/msg:opacity-100 focus-within:opacity-100 motion-reduce:transition-none [@media(hover:none)]:opacity-100"
    >
      {checkpoint && (
        <>
          <IconButton
            label="Fork"
            aria-description="Start a new conversation from before this message. This one is not changed."
            className="size-9"
            disabled={busy !== null}
            icon={<GitFork aria-hidden="true" className="size-4" />}
            onClick={() => void run("fork", checkpoint.onFork)}
          />
          <IconButton
            label={rewindBlocked ? "Rewind (stop first)" : "Rewind"}
            aria-description={rewindBlocked ?? "Go back to before this message. Later messages are hidden, and Undo brings them back."}
            className="size-9"
            disabled={busy !== null || rewindBlocked !== null}
            icon={<Undo2 aria-hidden="true" className="size-4" />}
            onClick={() => void run("rewind", checkpoint.onRewind)}
          />
        </>
      )}
      <IconButton
        label={copied ? "Copied" : "Copy"}
        aria-description="Copy this message to the clipboard"
        className="size-9"
        icon={copied ? <Check aria-hidden="true" className="size-4 text-ok" /> : <Copy aria-hidden="true" className="size-4" />}
        onClick={() => void copy()}
      />
      <span role="status" className="sr-only">
        {copied ? "Copied to the clipboard" : ""}
      </span>
    </div>
  );
}
