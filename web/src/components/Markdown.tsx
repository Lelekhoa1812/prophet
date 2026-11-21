import { isValidElement, useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type ComponentPropsWithoutRef, type ReactElement, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { Check, Copy } from "lucide-react";

// Answers are written as Markdown by the composer. This renders it as a document: bold, italic, links, lists, tables,
// footnote references, fenced snippets with a copy button, and Mermaid diagrams when the answer contains one.
// Nothing the model writes becomes live HTML: raw HTML is not parsed, the output is sanitised, links open with
// rel="noopener noreferrer", and Mermaid runs in its strict security level.

// Answers stored before the composer existed arrive as one long line. Where the structure is still visible, put the
// line breaks back so lists and paragraphs read as they were meant to. Text that already has line breaks is left alone.
export function repairFlat(text: string): string {
  const t = text.replace(/\r\n/g, "\n");
  if (t.includes("\n") || t.length < 160) return t;
  let out = t.replace(/ {2,}(?=(?:[-*•]|\d+[.)])\s)/g, "\n");
  out = out.replace(/ {2,}(?=#{1,4}\s)/g, "\n\n");
  out = out.replace(/ {2,}/g, "\n\n");
  return out;
}

const SAFE_URL = /^(https?:|mailto:|#|\/)/i;

const TEXT = "[overflow-wrap:anywhere]";

function CopyButton({ text, label }: { text: string; label: string }) {
  const [done, setDone] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return (
    <button
      type="button"
      aria-label={label}
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => setDone(false), 1600);
        }).catch(() => {
          // The browser refused the clipboard write: show nothing as copied.
          setDone(false);
        });
      }}
      className="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2 text-sm text-sand-1 transition-colors hover:bg-night-3 hover:text-sand-0"
    >
      {done ? <Check size={14} aria-hidden /> : <Copy size={14} aria-hidden />}
      <span aria-live="polite">{done ? "Copied" : "Copy"}</span>
    </button>
  );
}

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  return (
    <figure className="m-0 overflow-hidden rounded-xl border border-sand-2/20 bg-night-1">
      <figcaption className="flex items-center justify-between gap-3 border-b border-sand-2/15 bg-night-2 py-0.5 pr-1 pl-3 text-sm text-sand-1">
        <span>{lang || "text"}</span>
        <CopyButton text={code} label={`Copy ${lang || "text"} snippet`} />
      </figcaption>
      <pre className="m-0 max-h-96 overflow-auto p-4 text-[0.9375rem] leading-snug text-sand-0">
        <code>{code}</code>
      </pre>
    </figure>
  );
}

let mermaidReady: Promise<typeof import("mermaid").default> | null = null;
function loadMermaid() {
  mermaidReady ??= import("mermaid").then((m) => {
    m.default.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      fontFamily: '"IBM Plex Sans", system-ui, sans-serif',
      themeVariables: {
        darkMode: true,
        background: "#171410",
        primaryColor: "#2b241c",
        primaryTextColor: "#f3e9d2",
        primaryBorderColor: "#8d7e65",
        lineColor: "#c2b392",
        secondaryColor: "#211c16",
        tertiaryColor: "#171410",
        textColor: "#f3e9d2",
        noteBkgColor: "#34201a",
        noteTextColor: "#f3e9d2",
      },
    });
    return m.default;
  });
  return mermaidReady;
}

function Diagram({ code }: { code: string }) {
  const id = "mmd" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    setSvg(null);
    setFailed(false);
    loadMermaid()
      .then((m) => m.render(id, code))
      .then((r) => live && setSvg(r.svg))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
      document.getElementById("d" + id)?.remove();
    };
  }, [code, id]);
  if (failed)
    return (
      <div className="grid gap-2">
        <p className="m-0 text-sm text-sand-1">This diagram could not be drawn. Its source is shown instead.</p>
        <CodeBlock code={code} lang="mermaid" />
      </div>
    );
  if (!svg)
    return (
      <div role="status" className="rounded-xl border border-sand-2/20 bg-night-1 p-4 text-sand-1">
        Drawing the diagram…
      </div>
    );
  return (
    <figure className="m-0 overflow-x-auto rounded-xl border border-sand-2/20 bg-night-1 p-4" aria-label="Diagram">
      <div className="mx-auto w-fit [&_svg]:h-auto [&_svg]:max-w-full" dangerouslySetInnerHTML={{ __html: svg }} />
    </figure>
  );
}

function textOf(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (isValidElement(node)) return textOf((node.props as { children?: ReactNode }).children);
  return "";
}

const NUMERIC = /^[\s\-+−~≈$€£¥]*[\d][\d,.\s]*(?:%|[a-zA-Z]{0,3})?\s*$/;

// A column whose cells are nearly all numbers is right-aligned, header included, unless the author set an alignment.
function Table({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLTableElement>(null);
  useLayoutEffect(() => {
    const table = ref.current;
    if (!table) return;
    const rows = [...table.querySelectorAll("tr")].map((r) => [...r.children] as HTMLElement[]);
    const width = Math.max(0, ...rows.map((r) => r.length));
    for (let c = 0; c < width; c++) {
      const body = rows.slice(1).map((r) => r[c]).filter((el): el is HTMLElement => !!el && !!el.textContent?.trim());
      if (!body.length || body.some((el) => el.style.textAlign)) continue;
      const numeric = body.filter((el) => NUMERIC.test(el.textContent ?? "")).length / body.length >= 0.8;
      for (const r of rows) if (numeric && r[c]) r[c].style.textAlign = "right";
    }
  });
  return (
    <div className="overflow-x-auto rounded-xl border border-sand-2/20" tabIndex={0} role="region" aria-label="Table">
      <table ref={ref} className="w-full min-w-max border-collapse text-left text-[0.9375rem]">
        {children}
      </table>
    </div>
  );
}

const components: Components = {
  a({ href, children }) {
    const url = href ?? "";
    if (!SAFE_URL.test(url)) return <span>{children}</span>;
    const external = /^https?:/i.test(url);
    return (
      <a
        href={url}
        {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
        className="text-ember-strong underline decoration-ember/50 underline-offset-2 hover:decoration-ember-strong"
      >
        {children}
      </a>
    );
  },
  pre({ children }) {
    const child = (Array.isArray(children) ? children[0] : children) as ReactElement<{ className?: string; children?: ReactNode }> | undefined;
    const props = child && isValidElement(child) ? child.props : undefined;
    const code = textOf(props?.children).replace(/\n$/, "");
    const lang = /language-([\w+-]+)/.exec(props?.className ?? "")?.[1] ?? "";
    if (lang === "mermaid") return <Diagram code={code} />;
    return <CodeBlock code={code} lang={lang} />;
  },
  code({ children, className }) {
    return <code className={`rounded bg-night-2 px-1.5 py-0.5 text-[0.95em] text-sand-0 ${className ?? ""}`}>{children}</code>;
  },
  h1: ({ children }) => <h2 className="mt-2 mb-0 text-xl font-bold text-sand-0">{children}</h2>,
  h2: ({ children, className }) =>
    className?.includes("sr-only") ? (
      <h3 className="sr-only">{children}</h3>
    ) : (
      <h3 className="mt-2 mb-0 text-lg font-bold text-sand-0">{children}</h3>
    ),
  h3: ({ children }) => <h4 className="mt-1 mb-0 text-base font-bold text-sand-0">{children}</h4>,
  h4: ({ children }) => <h5 className="mt-1 mb-0 text-base font-bold text-sand-1">{children}</h5>,
  p: ({ children }) => <p className={`m-0 ${TEXT}`}>{children}</p>,
  strong: ({ children }) => <strong className="font-bold text-sand-0">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  del: ({ children }) => <del className="text-sand-2">{children}</del>,
  ul: ({ children, className }) => (
    <ul className={`m-0 grid list-disc gap-1.5 pl-6 marker:text-sand-2 ${className?.includes("contains-task-list") ? "list-none pl-1" : ""}`}>{children}</ul>
  ),
  ol: ({ children }) => <ol className="m-0 grid list-decimal gap-1.5 pl-6 marker:text-sand-2">{children}</ol>,
  li: ({ children }) => <li className={`${TEXT} [&>ol]:mt-1.5 [&>p]:inline [&>ul]:mt-1.5`}>{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="m-0 grid gap-2 border-l-2 border-ember/60 pl-4 text-sand-1">{children}</blockquote>
  ),
  hr: () => <hr className="my-1 border-0 border-t border-sand-2/25" />,
  input: (props: ComponentPropsWithoutRef<"input">) => <input {...props} disabled className="mr-2 accent-[var(--color-ember)]" />,
  table: ({ children }) => <Table>{children}</Table>,
  thead: ({ children }) => <thead className="bg-night-1 text-sand-0">{children}</thead>,
  tr: ({ children }) => <tr className="border-t border-sand-2/15 first:border-t-0">{children}</tr>,
  th: ({ children, style }) => (
    <th scope="col" style={style} className="px-3 py-2 font-bold">
      {children}
    </th>
  ),
  td: ({ children, style }) => (
    <td style={style} className="px-3 py-2 align-top tabular-nums">
      {children}
    </td>
  ),
  section({ children, className }) {
    if (className?.includes("footnotes"))
      return <section className="mt-1 grid gap-1 border-t border-sand-2/25 pt-3 text-[0.9375rem] text-sand-1">{children}</section>;
    return <section>{children}</section>;
  },
  sup: ({ children }) => <sup className="text-[0.75em] leading-none">{children}</sup>,
};

export function Markdown({ text, className = "" }: { text: string; className?: string }) {
  const source = useMemo(() => repairFlat(text), [text]);
  return (
    <div className={`grid gap-3.5 text-sand-0 ${className}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]} components={components}>
        {source}
      </ReactMarkdown>
    </div>
  );
}
