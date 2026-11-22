import { FileText, Sparkles, Wrench } from "lucide-react";
import { useEffect, useId, useRef, useState } from "react";
import { loadSkill, loadTools } from "../lib/catalog";
import { parseGoal, type Mention, type MentionKind } from "../lib/mentions";
import { cn } from "../lib/utils";

const ICON = { skill: Sparkles, tool: Wrench, file: FileText } as const;
const KIND_WORD: Record<MentionKind, string> = { skill: "Playbook", tool: "Tool", file: "File" };

// What the opened chip says: a playbook's summary and first lines, a tool's description, a file's place in the question.
function useDetail(m: Mention, open: boolean): { text: string; more?: string } {
  const [detail, setDetail] = useState<{ text: string; more?: string }>({ text: "" });
  useEffect(() => {
    if (!open) return;
    let live = true;
    const done = (d: { text: string; more?: string }) => live && setDetail(d);
    if (m.kind === "skill") {
      loadSkill(m.id)
        .then((s) => done({ text: s.summary, more: s.body.split("\n").slice(0, 8).join("\n") }))
        .catch(() => done({ text: "This playbook is no longer in the library." }));
    } else if (m.kind === "tool") {
      loadTools()
        .then((l) => {
          const t = l.find((x) => x.name === m.id);
          done({ text: t ? t.description || t.activity : "This tool is no longer available." });
        })
        .catch(() => done({ text: "" }));
    } else {
      done({ text: "Attached to this question. The agent reads it first." });
    }
    return () => {
      live = false;
    };
  }, [open, m.kind, m.id]);
  return detail;
}

// A pick inside a sent message: a link-styled chip that opens a short card. It is a button, so the keyboard reaches it,
// and the card closes on Escape or a click elsewhere.
export function MentionChip({ mention }: { mention: Mention }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const cardId = useId();
  const Icon = ICON[mention.kind];
  const detail = useDetail(mention, open);

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  return (
    <span ref={ref} className="relative inline-block align-baseline">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={open ? cardId : undefined}
        data-mention={mention.kind}
        title={`${KIND_WORD[mention.kind]}: ${mention.label.replace(/^[/@]/, "")}`}
        className={cn(
          "inline-flex max-w-full items-center gap-1 rounded-md px-1.5 align-baseline text-[0.95em] leading-[1.4] underline decoration-dotted underline-offset-4 transition-colors",
          mention.kind === "file"
            ? "bg-night-3 text-sand-0 hover:bg-night-1"
            : "bg-ember-soft text-ember-strong hover:bg-night-3",
        )}
      >
        <Icon aria-hidden="true" className="size-3.5 shrink-0" />
        <span className="truncate">{mention.label}</span>
      </button>
      {open && (
        <span
          id={cardId}
          role="dialog"
          aria-label={`${KIND_WORD[mention.kind]} ${mention.label}`}
          className="absolute top-full left-0 z-30 mt-1.5 grid w-72 max-w-[75vw] gap-1 rounded-xl border border-sand-2/30 bg-night-2 p-3 text-left text-caption leading-snug whitespace-normal text-sand-1 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)]"
        >
          <span className="text-micro font-semibold text-sand-2">{KIND_WORD[mention.kind]}</span>
          {detail.text && <span>{detail.text}</span>}
          {detail.more && <span className="mt-1 max-h-32 overflow-y-auto font-mono text-micro whitespace-pre-wrap text-sand-2">{detail.more}</span>}
        </span>
      )}
    </span>
  );
}

// A sent message: words as they were typed, each pick as a chip.
export function MentionText({ text }: { text: string }) {
  return (
    <>
      {parseGoal(text).map((s, i) =>
        s.type === "text" ? <span key={i}>{s.text}</span> : <MentionChip key={i} mention={s} />,
      )}
    </>
  );
}
