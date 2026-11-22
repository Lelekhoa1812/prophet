import { Extension, mergeAttributes } from "@tiptap/core";
import Mention from "@tiptap/extension-mention";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import type { SuggestionProps } from "@tiptap/suggestion";
import { SIGIL, type MentionKind } from "./mentions";

// One row in the "/" or "@" list. A row either becomes a chip (attrs) or inserts plain words (text, for the quick starts).
export type PickOption = {
  key: string;
  group: string;
  label: string;
  detail: string;
  icon: "skill" | "tool" | "file" | "start";
  attrs?: { kind: MentionKind; id: string; label: string };
  text?: string;
};

export type Trigger = "file" | "command";

// What the editor needs from the page: the rows for what was typed, somewhere to show them, the keys the list takes,
// and what Enter does when no list is open. The page owns the list, so the editor stays a plain text field with chips.
export type EditorHost = {
  items: (trigger: Trigger, query: string) => PickOption[];
  show: (trigger: Trigger, items: PickOption[], choose: (o: PickOption) => void) => void;
  hide: () => void;
  key: (event: KeyboardEvent) => boolean;
  open: () => boolean;
  submit: () => boolean;
  coarse: () => boolean;
};

function chip(name: string, char: string, trigger: Trigger, defaultKind: MentionKind, host: EditorHost) {
  return Mention.extend({
    name,
    addAttributes() {
      return {
        id: { default: null },
        label: { default: null },
        kind: { default: defaultKind },
      };
    },
    renderText({ node }) {
      return `${SIGIL[node.attrs.kind as MentionKind]}${node.attrs.label ?? node.attrs.id}`;
    },
  }).configure({
    deleteTriggerWithBackspace: false,
    HTMLAttributes: {},
    renderHTML({ node }) {
      const kind = node.attrs.kind as MentionKind;
      return [
        "span",
        mergeAttributes({
          class: "composer-chip",
          "data-mention": kind,
          "data-id": String(node.attrs.id),
          contenteditable: "false",
        }),
        `${SIGIL[kind]}${node.attrs.label ?? node.attrs.id}`,
      ];
    },
    suggestion: {
      char,
      pluginKey: new PluginKey(`${name}-suggestion`),
      allowedPrefixes: [" ", "\n"],
      items: ({ query }: { query: string }) => host.items(trigger, query),
      command: ({ editor, range, props }) => {
        const o = props as unknown as PickOption;
        if (o.attrs) {
          editor
            .chain()
            .focus()
            .insertContentAt(range, [
              { type: o.attrs.kind === "file" ? "fileMention" : "commandMention", attrs: o.attrs },
              { type: "text", text: " " },
            ])
            .run();
        } else {
          editor.chain().focus().insertContentAt(range, `${o.text ?? ""} `).run();
        }
      },
      render: () => ({
        onStart: (p: SuggestionProps<PickOption>) => host.show(trigger, p.items, (o) => p.command(o)),
        onUpdate: (p: SuggestionProps<PickOption>) => host.show(trigger, p.items, (o) => p.command(o)),
        onKeyDown: ({ event }: { event: KeyboardEvent }) => host.key(event),
        onExit: () => host.hide(),
      }),
    },
  });
}

export function mentionExtensions(host: EditorHost) {
  return [
    chip("fileMention", "@", "file", "file", host),
    chip("commandMention", "/", "command", "skill", host),
    // Enter sends and Shift+Enter breaks the line. On a touch screen Enter breaks the line, and Enter that confirms an
    // input-method candidate is left alone. The plugin runs before the editor's own Enter (which would split the
    // paragraph) and hands Enter to the open list first, so choosing a row never sends.
    Extension.create({
      name: "sendOnEnter",
      priority: 1000,
      addProseMirrorPlugins() {
        const editor = this.editor;
        return [
          new Plugin({
            key: new PluginKey("sendOnEnter"),
            props: {
              handleKeyDown: (view, event) => {
                if (event.key !== "Enter" || event.isComposing || view.composing) return false;
                if (host.open()) return host.key(event);
                if (event.shiftKey || host.coarse()) return editor.commands.setHardBreak();
                return host.submit();
              },
            },
          }),
        ];
      },
    }),
  ];
}
