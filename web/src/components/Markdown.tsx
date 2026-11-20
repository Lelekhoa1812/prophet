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

export function Markdown({ text, className = "" }: { text: string; className?: string }) {
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
    if (/^\s*```/.test(line)) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      i++;
      blocks.push(
        <pre key={key++} className="max-h-80 overflow-auto rounded-xl bg-night-1 p-4 text-[0.9375rem] leading-snug whitespace-pre-wrap [overflow-wrap:anywhere] text-sand-1">
          {code.join("\n")}
        </pre>,
      );
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
      const cells = (l: string) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
      const head = cells(line);
      i += 2;
      const body: string[][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        body.push(cells(lines[i]));
        i++;
      }
      blocks.push(
        <div key={key++} className="overflow-x-auto rounded-xl border border-sand-2/20">
          <table className="w-full min-w-max border-collapse text-left text-[0.9375rem]">
            <thead>
              <tr className="bg-night-1 text-sand-1">
                {head.map((h, n) => (
                  <th key={n} scope="col" className="px-3 py-2 font-semibold">
                    {inline(h)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {body.map((r, n) => (
                <tr key={n} className="border-t border-sand-2/15">
                  {r.map((c, k) => (
                    <td key={k} className="px-3 py-2 tabular-nums">
                      {inline(c)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
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
    while (i < lines.length && lines[i].trim() && !/^#{1,3}\s/.test(lines[i]) && !/^\s*```/.test(lines[i]) && !/^\s*\|.*\|\s*$/.test(lines[i]) && !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    blocks.push(
      <p key={key++} className="[overflow-wrap:anywhere]">
        {inline(para.join(" "))}
      </p>,
    );
  }
  return <div className={`grid gap-3.5 text-sand-0 ${className}`}>{blocks}</div>;
}
