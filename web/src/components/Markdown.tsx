import type { ReactNode } from "react";

// A deliberately small renderer for the agent's answers: paragraphs, "- " and "1. " lists, ## headings and **bold**.
// It builds React elements, never HTML strings, so nothing the model writes can inject markup.
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2)
      return (
        <code key={i} className="rounded bg-night-2 px-1 py-0.5 text-[0.95em]">
          {part.slice(1, -1)}
        </code>
      );
    return part;
  });
}

export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const heading = /^#{1,3}\s+(.*)$/.exec(line);
    if (heading) {
      blocks.push(
        <h3 key={key++} className="mt-1 text-lg font-bold text-sand-0">
          {inline(heading[1])}
        </h3>,
      );
      i++;
      continue;
    }
    const bullet = /^\s*[-*•]\s+/.test(line);
    const numbered = /^\s*\d+[.)]\s+/.test(line);
    if (bullet || numbered) {
      const items: string[] = [];
      const re = bullet ? /^\s*[-*•]\s+(.*)$/ : /^\s*\d+[.)]\s+(.*)$/;
      while (i < lines.length && re.test(lines[i])) {
        items.push(re.exec(lines[i])![1]);
        i++;
      }
      const Tag = bullet ? "ul" : "ol";
      blocks.push(
        <Tag key={key++} className={`grid gap-1 pl-6 ${bullet ? "list-disc" : "list-decimal"} marker:text-sand-2`}>
          {items.map((t, n) => (
            <li key={n}>{inline(t)}</li>
          ))}
        </Tag>,
      );
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^#{1,3}\s/.test(lines[i]) && !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="[overflow-wrap:anywhere]">
        {inline(para.join(" "))}
      </p>,
    );
  }
  return <div className="grid gap-3 text-sand-0">{blocks}</div>;
}
