// A skill, tool or file picked with "/" or "@" is stored inside the message as a markdown link with a prophet: scheme,
//   [/Choose model by backtest](prophet:skill/choose-model-by-backtest)   [@sales.csv](prophet:file/<id>)
// The label carries its sigil, so removing the link syntax leaves the words the person saw. The server reads these links
// itself (src/prophet_agent/mentions.py) and checks each against what exists: a skill loads its playbook, a tool is
// requested by name and a file is attached. Keep LINK in step with the server's.
export type MentionKind = "skill" | "tool" | "file";
export type Mention = { kind: MentionKind; id: string; label: string };
export type Segment = { type: "text"; text: string } | ({ type: "mention" } & Mention);

const LINK = /\[([^[\]\n]{1,120})\]\(prophet:(skill|tool|file)\/([A-Za-z0-9_.:-]{1,80})\)/g;

export const SIGIL: Record<MentionKind, string> = { skill: "/", tool: "/", file: "@" };

export function parseGoal(text: string): Segment[] {
  const out: Segment[] = [];
  let at = 0;
  for (const m of text.matchAll(LINK)) {
    const start = m.index ?? 0;
    if (start > at) out.push({ type: "text", text: text.slice(at, start) });
    out.push({ type: "mention", kind: m[2] as MentionKind, id: m[3], label: m[1] });
    at = start + m[0].length;
  }
  if (at < text.length) out.push({ type: "text", text: text.slice(at) });
  return out;
}

// The message as the person read it.
export function plainGoal(text: string): string {
  return text.replace(LINK, (_, label: string) => label);
}

export function hasMentions(text: string): boolean {
  return new RegExp(LINK.source).test(text);
}

// Brackets and line breaks cannot sit in a label.
export function cleanLabel(label: string): string {
  return label.replace(/[[\]\n\r]+/g, " ").trim().slice(0, 120);
}

export function mentionLink(kind: MentionKind, id: string, label: string): string {
  return `[${cleanLabel(label)}](prophet:${kind}/${id})`;
}

// Editor documents. One paragraph; a line break is a hard break.
type Node = { type: string; text?: string; attrs?: Record<string, unknown> };
export type Doc = { type: "doc"; content: { type: "paragraph"; content?: Node[] }[] };

export function docFromGoal(text: string): Doc {
  const content: Node[] = [];
  const pushText = (t: string) => {
    t.split("\n").forEach((line, i) => {
      if (i > 0) content.push({ type: "hardBreak" });
      if (line) content.push({ type: "text", text: line });
    });
  };
  for (const seg of parseGoal(text)) {
    if (seg.type === "text") pushText(seg.text);
    else content.push({ type: seg.kind === "file" ? "fileMention" : "commandMention", attrs: { kind: seg.kind, id: seg.id, label: seg.label.replace(/^[/@]/, "") } });
  }
  return { type: "doc", content: [{ type: "paragraph", ...(content.length ? { content } : {}) }] };
}

// Serialise an editor document back to the stored text, and to the words alone (for the length count).
type Json = { type?: string; text?: string; attrs?: Record<string, unknown>; content?: Json[] };

function walk(node: Json, mention: (n: Json) => string): string {
  if (node.type === "text") return node.text ?? "";
  if (node.type === "hardBreak") return "\n";
  if (node.type === "fileMention" || node.type === "commandMention") return mention(node);
  const parts = (node.content ?? []).map((c) => walk(c, mention));
  return node.type === "doc" ? parts.join("\n") : parts.join("");
}

export function goalFromDoc(doc: Json): string {
  return walk(doc, (n) => {
    const a = n.attrs ?? {};
    const kind = a.kind as MentionKind;
    return mentionLink(kind, String(a.id), `${SIGIL[kind]}${String(a.label ?? a.id)}`);
  });
}

export function plainFromDoc(doc: Json): string {
  return walk(doc, (n) => {
    const a = n.attrs ?? {};
    return `${SIGIL[a.kind as MentionKind]}${String(a.label ?? a.id)}`;
  });
}
