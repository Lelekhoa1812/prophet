import { AlertCircle, ArrowUp, FileText, Folder, Paperclip, Plus, Sparkles, Square, Wrench, X } from "lucide-react";
import Document from "@tiptap/extension-document";
import HardBreak from "@tiptap/extension-hard-break";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import { Placeholder, UndoRedo } from "@tiptap/extensions";
import { EditorContent, useEditor } from "@tiptap/react";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type FormEvent,
  type ReactNode,
} from "react";
import type { SkillInfo, ToolInfo } from "../lib/api";
import { usePlaybooks } from "../lib/catalog";
import { mentionExtensions, type EditorHost, type PickOption, type Trigger } from "../lib/mentionEditor";
import { docFromGoal, goalFromDoc, plainFromDoc } from "../lib/mentions";
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

// Everything the "+" menu, "@" and "/" add on top of the plain message box. Quick forecasts and notebooks both use it.
export type ComposerQuick = {
  // Files attached to the message itself. Their chips sit inside the box, and their X removes them when given.
  attachments: Attachment[];
  // Files that already belong to the conversation (a notebook's included files). "@" can mention them, but they are
  // shown elsewhere, so they get no chip.
  mentions?: { key: string; name: string }[];
  tools: ToolInfo[];
  skills?: Skill[];
  onRemoveAttachment?: (key: string) => void;
  onAddFiles: (files: File[]) => void;
  onOpenFiles: () => void;
  onOpenFolder: () => void;
};

// What the open list shows: the rows for what was typed and how to turn the chosen one into a chip. "only" narrows "/" to
// playbooks or tools when the list was opened from the "+" menu.
type Sug = { trigger: Trigger; items: PickOption[]; choose: (o: PickOption) => void };

// One message box, like the one people already know: "+" on the left, send on the right.
// Enter sends, Shift+Enter adds a line, and Enter during IME composition is left alone. While the agent works the
// send button becomes Stop, so stopping is where the eye already is. The hint under the box appears only when it helps.
// "+" adds files, folders, skills or tools, "@" mentions a file and "/" picks a skill or tool. Quick forecasts and
// notebooks share this box; each passes its own files, footer and submit handler.
export function Composer({
  onSend,
  busy,
  blockedReason,
  initial,
  placeholder,
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
  working?: boolean;
  onStop?: () => void;
  stopping?: boolean;
  quick: ComposerQuick;
  // Controls shown along the bottom of the box, right-aligned (the model switcher and usage meter).
  footer?: ReactNode;
}) {
  const [error, setError] = useState<string | null>(null);
  const [sug, setSug] = useState<Sug | null>(null);
  const [active, setActive] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [doc, setDoc] = useState<Parameters<typeof goalFromDoc>[0]>(() => docFromGoal(initial ?? ""));
  const listId = useId();
  const playbooks = usePlaybooks();

  // The editor is made once, so what it reads (files, tools, playbooks, what to do on Enter) lives in refs the host reads.
  const live = useRef({ quick, playbooks, only: null as "skill" | "tool" | null, sug: null as Sug | null, active: 0, submit: () => {} });
  live.current.quick = quick;
  live.current.playbooks = playbooks;
  live.current.sug = sug;
  live.current.active = active;
  const placeholderRef = useRef(placeholder);
  placeholderRef.current = placeholder;

  const host = useMemo<EditorHost>(
    () => ({
      items: (trigger, query) => pickOptions(trigger, query, live.current.quick, live.current.playbooks, live.current.only),
      show: (trigger, items, choose) => {
        setSug({ trigger, items, choose });
        setActive((a) => (a < items.length ? a : 0));
      },
      hide: () => {
        live.current.only = null;
        setSug(null);
      },
      key: (e) => {
        const cur = live.current.sug;
        if (!cur || cur.items.length === 0) return false;
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          const n = cur.items.length;
          setActive((i) => (e.key === "ArrowDown" ? (i + 1) % n : (i - 1 + n) % n));
          return true;
        }
        if ((e.key === "Enter" && !e.shiftKey && !e.isComposing) || e.key === "Tab") {
          cur.choose(cur.items[live.current.active] ?? cur.items[0]);
          return true;
        }
        if (e.key === "Escape") {
          setSug(null);
          return true;
        }
        return false;
      },
      open: () => (live.current.sug?.items.length ?? 0) > 0,
      submit: () => {
        live.current.submit();
        return true;
      },
      coarse: isTouchPointer,
    }),
    [],
  );

  const editor = useEditor({
    extensions: [
      Document,
      Paragraph,
      Text,
      HardBreak,
      UndoRedo,
      Placeholder.configure({ placeholder: () => placeholderRef.current }),
      ...mentionExtensions(host),
    ],
    content: docFromGoal(initial ?? ""),
    editorProps: {
      attributes: {
        id: "composer",
        role: "combobox",
        "aria-label": "Your question",
        "aria-autocomplete": "list",
        "aria-multiline": "true",
        class: "composer-editor",
      },
    },
    onUpdate: ({ editor: ed }) => {
      setDoc(ed.getJSON());
      setActive(0);
    },
  });

  useEffect(() => {
    if (initial !== undefined && editor) {
      editor.commands.setContent(docFromGoal(initial));
      setDoc(editor.getJSON());
      editor.commands.focus("end");
    }
  }, [initial, editor]);

  const options = sug?.items ?? [];
  const listOpen = sug !== null && options.length > 0;

  // The stored message keeps each pick as a link; the person's count and the minimum length use the words they see.
  const goal = goalFromDoc(doc).trim();
  const shown = plainFromDoc(doc).trim();
  const textLength = plainFromDoc(doc).length;
  const canSend = shown.length >= 3 && shown.length <= MAX && !busy && !blockedReason;
  const showStop = working && onStop !== undefined;
  const near = textLength >= NEAR_LIMIT;
  const full = textLength >= MAX;
  const activeId = listOpen ? `${listId}-${active}` : undefined;

  useEffect(() => {
    const dom = editor?.view.dom;
    if (!dom) return;
    dom.setAttribute("aria-expanded", String(listOpen));
    if (listOpen) dom.setAttribute("aria-controls", listId);
    else dom.removeAttribute("aria-controls");
    if (activeId) dom.setAttribute("aria-activedescendant", activeId);
    else dom.removeAttribute("aria-activedescendant");
  }, [editor, listOpen, listId, activeId]);

  const submit = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      if (showStop || !canSend) return;
      setError(null);
      try {
        await onSend(goal);
        editor?.commands.clearContent(true);
        setDoc(editor?.getJSON() ?? docFromGoal(""));
        setSug(null);
      } catch (err) {
        setError(err instanceof Error && err.message ? err.message : "Couldn't send that. Try again.");
      }
    },
    [showStop, canSend, onSend, goal, editor],
  );
  live.current.submit = () => void submit();

  // The "+" menu opens the "/" list at the caret, narrowed to playbooks or tools. The list is the same one typing "/" opens.
  function openFromMenu(kind: "skill" | "tool") {
    if (!editor) return;
    live.current.only = kind;
    const before = editor.state.doc.textBetween(Math.max(0, editor.state.selection.from - 1), editor.state.selection.from, "\n");
    editor.chain().focus().insertContent(before === "" || before === " " || before === "\n" ? "/" : " /").run();
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    // A notebook's page is also a drop zone; stop here so one drop adds its files once.
    e.stopPropagation();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files ?? []);
    if (files.length) quick.onAddFiles(files);
  }

  // Only errors and blocking reasons are shown under the box. The send hint is hidden.
  const note = error ?? blockedReason ?? null;
  const removable = quick.onRemoveAttachment;
  // Options keep one flat index (for the arrow keys) but render under their group heading.
  const groups = options.reduce<{ name: string; items: { option: PickOption; index: number }[] }[]>((acc, option, index) => {
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
          aria-label={sug?.trigger === "file" ? "Files" : "Skills and tools"}
          className="absolute inset-x-0 bottom-full z-30 mb-2 grid max-h-80 gap-0.5 overflow-y-auto rounded-xl border border-sand-2/30 bg-night-2 p-1 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)]"
        >
          {groups.map((g) => (
            <div key={g.name} role="group" aria-label={g.name} className="grid gap-0.5">
              <p aria-hidden="true" className="px-3 pt-2 pb-1 text-micro font-semibold text-sand-2">
                {g.name}
              </p>
              {g.items.map(({ option: o, index: i }) => (
                <div
                  key={o.key}
                  id={`${listId}-${i}`}
                  role="option"
                  aria-selected={i === active}
                  onMouseDown={(e) => e.preventDefault()}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => sug?.choose(o)}
                  className={cn(
                    "flex min-h-11 cursor-pointer items-center gap-3 rounded-lg px-3 py-1.5 sm:min-h-10",
                    i === active ? "bg-night-3" : "hover:bg-night-3",
                  )}
                >
                  <span aria-hidden="true" className="grid size-7 shrink-0 place-items-center rounded-md bg-night-1 text-sand-1">
                    <PickIcon icon={o.icon} />
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
        {quick.attachments.length > 0 && (
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
                {removable && (
                  <button
                    type="button"
                    onClick={() => removable(a.key)}
                    aria-label={`Remove ${a.name}`}
                    className="grid size-7 shrink-0 place-items-center rounded-md text-sand-2 hover:bg-night-3 hover:text-sand-0"
                  >
                    <X aria-hidden="true" className="size-3.5" />
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center gap-1">
          <ActionMenu
            label="Add files, folders, skills or tools"
            triggerClassName="shrink-0 rounded-full bg-night-2 text-sand-1 hover:bg-night-3 hover:text-sand-0 sm:size-8"
            trigger={<Plus aria-hidden="true" className="size-4" />}
            items={[
              { key: "files", label: "Add files", icon: <Paperclip className="size-4" />, onSelect: quick.onOpenFiles },
              { key: "folder", label: "Add folder", icon: <Folder className="size-4" />, onSelect: quick.onOpenFolder },
              { key: "skills", label: "Skills", icon: <Sparkles className="size-4" />, onSelect: () => openFromMenu("skill") },
              { key: "tools", label: "Tools", icon: <Wrench className="size-4" />, onSelect: () => openFromMenu("tool") },
            ]}
          />
          <EditorContent editor={editor} className="min-w-0 flex-1" />
          {showStop ? (
            <button
              type="button"
              onClick={onStop}
              disabled={stopping}
              aria-label={stopping ? "Stopping" : "Stop"}
              title="Stop"
              className="grid size-10 shrink-0 place-items-center rounded-full bg-sand-0 text-night-0 transition-colors hover:bg-sand-1 disabled:opacity-60 sm:size-8"
            >
              {stopping ? <Spinner /> : <Square aria-hidden="true" className="size-3.5 fill-current" />}
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              aria-label={busy ? "Sending" : "Send"}
              title="Send"
              className="grid size-10 shrink-0 place-items-center rounded-full bg-ember text-night-0 transition-colors hover:bg-ember-strong disabled:cursor-not-allowed disabled:bg-night-3 disabled:text-sand-2 sm:size-8"
            >
              {busy ? <Spinner /> : <ArrowUp aria-hidden="true" className="size-4" />}
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
            {textLength}/{MAX}
            {full ? ". For longer data, attach a file." : ""}
          </p>
        )}
      </div>
    </form>
  );
}

function PickIcon({ icon }: { icon: PickOption["icon"] }) {
  if (icon === "file") return <FileText aria-hidden="true" className="size-4" />;
  if (icon === "tool") return <Wrench aria-hidden="true" className="size-4" />;
  return <Sparkles aria-hidden="true" className="size-4" />;
}

// What the list offers. "@" lists the files that can be pointed at (attached and ready, or already in the notebook) and
// "/" lists the analysis playbooks, the tools and the quick starts. A playbook, tool or file becomes a chip; a quick start
// puts its sentence in the box.
function pickOptions(trigger: Trigger, query: string, quick: ComposerQuick, playbooks: SkillInfo[], only: "skill" | "tool" | null): PickOption[] {
  const q = query.trim().toLowerCase();
  const match = (...s: string[]) => !q || s.some((x) => x.toLowerCase().includes(q));

  if (trigger === "file") {
    const attached = quick.attachments.filter((a) => a.status === "ready").map((a) => ({ id: a.datasetId ?? a.key, name: a.name }));
    const files = [...attached, ...(quick.mentions ?? []).map((m) => ({ id: m.key, name: m.name }))];
    const seen = new Set<string>();
    return files
      .filter((f) => match(f.name) && !seen.has(f.id) && seen.add(f.id))
      .map((f) => ({
        key: `file-${f.id}`,
        group: "Files",
        label: f.name,
        detail: "Point the agent at this file",
        icon: "file" as const,
        attrs: { kind: "file" as const, id: f.id, label: f.name },
      }));
  }

  // A name that contains what was typed comes first; the one-line summaries are searched only when no name matches.
  const byName = playbooks.filter((s) => match(s.title, s.name));
  const skills: PickOption[] = (byName.length || !q ? byName : playbooks.filter((s) => match(s.summary)))
    .map((s) => ({
      key: `skill-${s.name}`,
      group: "Skills",
      label: s.title,
      detail: s.summary,
      icon: "skill" as const,
      attrs: { kind: "skill" as const, id: s.name, label: s.title },
    }));
  const tools: PickOption[] = quick.tools
    .filter((t) => match(t.title || t.name, t.name))
    .map((t) => ({
      key: `tool-${t.name}`,
      group: "Tools",
      label: t.title || t.name,
      detail: t.description || t.activity,
      icon: "tool" as const,
      attrs: { kind: "tool" as const, id: t.name, label: t.title || t.name },
    }));
  const starts: PickOption[] = (quick.skills ?? SKILLS)
    .filter((s) => match(s.title, s.description))
    .map((s: Skill) => ({ key: `start-${s.key}`, group: "Quick starts", label: s.title, detail: s.description, icon: "start" as const, text: s.prompt }));

  if (only === "skill") return skills;
  if (only === "tool") return tools;
  return [...skills, ...tools, ...starts];
}
