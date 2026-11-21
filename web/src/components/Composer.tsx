import { AlertCircle, ArrowUp, FileText, Folder, Paperclip, Plus, Sparkles, Square, Wrench, X } from "lucide-react";
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import type { ToolInfo } from "../lib/api";
import { SKILLS, type Skill } from "../lib/skills";
import { cn } from "../lib/utils";
import { ActionMenu } from "./ActionMenu";
import { Spinner } from "./ui";

const MAX = 2000;
// From this length the counter shows, so people see the limit before they reach it.
const NEAR_LIMIT = 1800;

// Phones and tablets: Enter adds a line, because the on-screen keyboard has no Shift and people expect Return to break a line.
function isTouchPointer(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches === true;
}

// A file added to a quick forecast. Ready files are sent with the question as data sets.
export type Attachment = { key: string; name: string; status: "uploading" | "ready" | "failed"; datasetId?: string; message?: string };

// Everything the quick-forecast composer adds on top of the plain message box. Notebooks leave it out.
export type ComposerQuick = {
  attachments: Attachment[];
  tools: ToolInfo[];
  skills?: Skill[];
  onRemoveAttachment: (key: string) => void;
  onAddFiles: (files: File[]) => void;
  onOpenFiles: () => void;
  onOpenFolder: () => void;
};

type Pick = { kind: "file" | "command" | "skill" | "tool"; anchor: number; trigger: boolean; query: string };
type Option = { id: string; group: string; label: string; detail: string; icon: ReactNode; insert: string };

// One message box, like the one people already know: attach on the left, send on the right.
// Enter sends, Shift+Enter adds a line, and Enter during IME composition is left alone. While the agent works the
// send button becomes Stop, so stopping is where the eye already is. The hint under the box appears only when it helps.
// In a quick forecast, "+" adds files, folders, skills or tools, "@" mentions an attached file and "/" picks a skill or tool.
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
  quick,
  footer,
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
  quick?: ComposerQuick;
  // Controls shown along the bottom of the box, right-aligned (the model switcher and usage meter).
  footer?: ReactNode;
}) {
  const [text, setText] = useState(initial ?? "");
  const [error, setError] = useState<string | null>(null);
  const [pick, setPick] = useState<Pick | null>(null);
  const [active, setActive] = useState(0);
  const [dragging, setDragging] = useState(false);
  const ref = useRef<HTMLTextAreaElement>(null);
  const listId = useId();

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

  const options = quick ? optionsFor(pick, quick) : [];
  const listOpen = pick !== null && options.length > 0;

  const trimmed = text.trim();
  const canSend = trimmed.length >= 3 && trimmed.length <= MAX && !busy && !blockedReason;
  const showStop = working && onStop !== undefined;
  const near = text.length >= NEAR_LIMIT;
  const full = text.length >= MAX;

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    if (showStop || !canSend) return;
    setError(null);
    try {
      await onSend(trimmed);
      setText("");
      setPick(null);
    } catch (err) {
      setError(err instanceof Error && err.message ? err.message : "Couldn't send that. Try again.");
    }
  }

  function choose(option: Option) {
    const el = ref.current;
    if (!el || !pick) return;
    const caret = el.selectionStart ?? text.length;
    const next = text.slice(0, pick.anchor) + option.insert + text.slice(caret);
    const pos = pick.anchor + option.insert.length;
    setText(next);
    setPick(null);
    requestAnimationFrame(() => {
      el.focus();
      el.setSelectionRange(pos, pos);
    });
  }

  // Opens a list from the "+" menu at the caret. Skills and tools appear as a list; they are not typed.
  function openFromMenu(kind: "skill" | "tool") {
    const el = ref.current;
    const caret = el?.selectionStart ?? text.length;
    setPick({ kind, anchor: caret, trigger: false, query: "" });
    setActive(0);
    el?.focus();
  }

  function onChange(e: ChangeEvent<HTMLTextAreaElement>) {
    const value = e.target.value;
    const caret = e.target.selectionStart ?? value.length;
    setText(value);
    setActive(0);
    const typed = quick ? /(^|\s)([@/])([^\s@/]*)$/.exec(value.slice(0, caret)) : null;
    if (typed) {
      const kind = typed[2] === "@" ? "file" : "command";
      setPick({ kind, anchor: caret - typed[3].length - 1, trigger: true, query: typed[3] });
    } else if (pick?.trigger) {
      setPick(null);
    } else if (pick) {
      setPick({ ...pick, query: value.slice(pick.anchor, caret) });
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (listOpen) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        setActive((i) => (e.key === "ArrowDown" ? (i + 1) % options.length : (i - 1 + options.length) % options.length));
        return;
      }
      if ((e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) || e.key === "Tab") {
        e.preventDefault();
        choose(options[active] ?? options[0]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setPick(null);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && !isTouchPointer()) {
      e.preventDefault();
      void submit();
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    if (!quick) return;
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files ?? []);
    if (files.length) quick.onAddFiles(files);
  }

  const hint = isTouchPointer() ? "Tap the arrow to send" : "Enter to send · Shift+Enter for a new line";
  const note = error ?? blockedReason ?? (trimmed.length > 0 ? hint : null);
  const activeId = listOpen ? `${listId}-${active}` : undefined;
  // Options keep one flat index (for the arrow keys) but render under their group heading.
  const groups = options.reduce<{ name: string; items: { option: Option; index: number }[] }[]>((acc, option, index) => {
    const last = acc[acc.length - 1];
    if (last && last.name === option.group) last.items.push({ option, index });
    else acc.push({ name: option.group, items: [{ option, index }] });
    return acc;
  }, []);

  return (
    <form onSubmit={submit} className="relative grid gap-1.5">
      {listOpen && (
        <div
          id={listId}
          role="listbox"
          aria-label={pick?.kind === "file" ? "Attached files" : pick?.kind === "tool" ? "Tools" : "Skills and tools"}
          className="absolute inset-x-0 bottom-full z-30 mb-2 grid max-h-80 gap-0.5 overflow-y-auto rounded-xl border border-sand-2/30 bg-night-2 p-1 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)]"
        >
          {groups.map((g) => (
            <div key={g.name} role="group" aria-label={g.name} className="grid gap-0.5">
              <p aria-hidden="true" className="px-3 pt-2 pb-1 text-micro font-semibold text-sand-2">
                {g.name}
              </p>
              {g.items.map(({ option: o, index: i }) => (
                <div
                  key={o.id}
                  id={`${listId}-${i}`}
                  role="option"
                  aria-selected={i === active}
                  onMouseDown={(e) => e.preventDefault()}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => choose(o)}
                  className={cn(
                    "flex min-h-11 cursor-pointer items-center gap-3 rounded-lg px-3 py-1.5 sm:min-h-10",
                    i === active ? "bg-night-3" : "hover:bg-night-3",
                  )}
                >
                  <span aria-hidden="true" className="grid size-7 shrink-0 place-items-center rounded-md bg-night-1 text-sand-1">
                    {o.icon}
                  </span>
                  <span className="grid min-w-0">
                    <span className="truncate text-base text-sand-0">{o.label}</span>
                    {o.detail && <span className="truncate text-caption text-sand-2">{o.detail}</span>}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      <div
        onDragOver={(e) => {
          if (!quick) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cn(
          "grid gap-1.5 rounded-[1.75rem] border bg-night-1 p-2 shadow-[0_6px_24px_-12px_rgba(0,0,0,0.8)] transition-colors",
          "border-sand-2/30 focus-within:border-ember focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-ember",
          dragging && "border-ember border-dashed bg-ember-soft",
        )}
      >
        {quick && quick.attachments.length > 0 && (
          <ul aria-label="Attached files" className="flex flex-wrap gap-1.5 px-1 pt-1">
            {quick.attachments.map((a) => (
              <li
                key={a.key}
                className={cn(
                  "flex max-w-full items-center gap-2 rounded-lg border bg-night-2 py-1 pr-1 pl-2 text-caption",
                  a.status === "failed" ? "border-bad/50" : "border-sand-2/30",
                )}
              >
                {a.status === "uploading" ? <Spinner /> : a.status === "failed" ? <AlertCircle aria-hidden="true" className="size-4 text-bad" /> : <FileText aria-hidden="true" className="size-4 text-sand-2" />}
                <span className="truncate text-sand-0" title={a.message ?? a.name}>
                  {a.name}
                </span>
                <span className="sr-only">{a.status === "uploading" ? "uploading" : a.status === "failed" ? `failed: ${a.message ?? "could not add"}` : "ready"}</span>
                <button
                  type="button"
                  onClick={() => quick.onRemoveAttachment(a.key)}
                  aria-label={`Remove ${a.name}`}
                  className="grid size-7 shrink-0 place-items-center rounded-md text-sand-2 hover:bg-night-3 hover:text-sand-0"
                >
                  <X aria-hidden="true" className="size-3.5" />
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center gap-1.5">
          {quick ? (
            <ActionMenu
              label="Add files, folders, skills or tools"
              trigger={<Plus aria-hidden="true" className="size-5" />}
              items={[
                { key: "files", label: "Add files", icon: <Paperclip className="size-4" />, onSelect: quick.onOpenFiles },
                { key: "folder", label: "Add folder", icon: <Folder className="size-4" />, onSelect: quick.onOpenFolder },
                { key: "skills", label: "Skills", icon: <Sparkles className="size-4" />, onSelect: () => openFromMenu("skill") },
                { key: "tools", label: "Tools", icon: <Wrench className="size-4" />, onSelect: () => openFromMenu("tool") },
              ]}
            />
          ) : (
            leading
          )}
          <label htmlFor="composer" className="sr-only">
            Your question
          </label>
          <textarea
            id="composer"
            ref={ref}
            value={text}
            rows={1}
            onChange={onChange}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            role={quick ? "combobox" : undefined}
            aria-expanded={quick ? listOpen : undefined}
            aria-controls={quick && listOpen ? listId : undefined}
            aria-activedescendant={activeId}
            aria-autocomplete={quick ? "list" : undefined}
            className="max-h-56 min-h-11 flex-1 resize-none bg-transparent px-2 py-2.5 text-body leading-snug text-sand-0 placeholder:text-sand-2 focus:outline-none"
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

        {footer && <div className="flex min-w-0 items-center justify-end gap-1 px-1 pb-0.5">{footer}</div>}
      </div>
      <div className="flex min-h-5 flex-wrap items-baseline justify-between gap-x-4 px-4 text-caption text-sand-2">
        <p className="min-w-0" role={error ? "alert" : undefined}>
          {error ? <span className="text-bad">{error}</span> : note}
        </p>
        {near && (
          <p className={cn("tabular-nums", full && "text-ember")}>
            {text.length}/{MAX}
            {full ? ". For longer data, attach a file." : ""}
          </p>
        )}
      </div>
    </form>
  );
}

// What the list offers for the current pick. "@" lists attached files; "/" and the "+" menu list skills and tools.
function optionsFor(pick: Pick | null, quick: ComposerQuick): Option[] {
  if (!pick) return [];
  const q = pick.query.trim().toLowerCase();
  const match = (s: string) => !q || s.toLowerCase().includes(q);

  if (pick.kind === "file") {
    const files = quick.attachments.filter((a) => a.status === "ready" && match(a.name));
    return files.map((a) => ({
      id: `file-${a.key}`,
      group: "Attached files",
      label: a.name,
      detail: "Mention this file",
      icon: <FileText aria-hidden="true" className="size-4" />,
      insert: `@${a.name} `,
    }));
  }

  const skills = (quick.skills ?? SKILLS)
    .filter((s) => match(s.title))
    .map<Option>((s) => ({ id: `skill-${s.key}`, group: "Skills", label: s.title, detail: s.description, icon: <s.icon aria-hidden="true" className="size-4" />, insert: s.prompt }));
  const tools = quick.tools
    .filter((t) => match(t.title || t.name))
    .map<Option>((t) => ({
      id: `tool-${t.name}`,
      group: "Tools",
      label: t.title || t.name,
      detail: t.description || t.activity,
      icon: <Wrench aria-hidden="true" className="size-4" />,
      insert: `Use the ${t.title || t.name} tool. `,
    }));

  if (pick.kind === "skill") return skills;
  if (pick.kind === "tool") return tools;
  return [...skills, ...tools];
}
