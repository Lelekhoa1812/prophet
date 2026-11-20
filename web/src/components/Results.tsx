import { Download, File, FileImage, FileSpreadsheet, FileText } from "lucide-react";
import { artifactUrl, type ArtifactFile, type ArtifactRun } from "../lib/api";
import { formatNumber } from "../lib/format";
import { formatBytes } from "../lib/utils";
import { Card } from "./ui";

const W = 640;
const H = 250;
const PAD = { l: 56, r: 12, t: 12, b: 28 };

type Fc = NonNullable<ArtifactRun["forecast"]>;

// The forecast, drawn from the numbers in the run: the series so far in sand, the forecast in ember, and the
// shaded band where 8 in 10 outcomes are expected (the P10 to P90 range).
export function ForecastChart({ history, fc }: { history: number[]; fc: Fc }) {
  const point = fc.point ?? [];
  const lo10 = fc.q10 ?? [];
  const hi90 = fc.q90 ?? [];
  const n = Math.min(point.length, lo10.length, hi90.length);
  if (n < 1) return null;
  const hist = history.slice(-Math.max(n * 3, 24));
  const total = hist.length + n;
  const all = [...hist, ...point.slice(0, n), ...lo10.slice(0, n), ...hi90.slice(0, n)];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const x = (i: number) => PAD.l + (i / Math.max(total - 1, 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - lo) / span) * (H - PAD.t - PAD.b);
  const path = (vals: number[], offset: number) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${x(offset + i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const off = hist.length;
  const bridge = hist.length > 0 ? [hist[hist.length - 1]] : [];
  const band =
    `M${x(off).toFixed(1)},${y(hi90[0]).toFixed(1)} ` +
    hi90.slice(0, n).map((v, i) => `L${x(off + i).toFixed(1)},${y(v).toFixed(1)}`).join(" ") +
    " " +
    lo10.slice(0, n).map((_, i) => { const j = n - 1 - i; return `L${x(off + j).toFixed(1)},${y(lo10[j]).toFixed(1)}`; }).join(" ") +
    " Z";
  const ticks = [lo, lo + span / 3, lo + (2 * span) / 3, hi];
  const first = point[0];
  const last = point[n - 1];
  const change = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : null;
  const reading =
    n === 1
      ? `The forecast is ${formatNumber(first)}, most likely between ${formatNumber(lo10[0])} and ${formatNumber(hi90[0])}.`
      : `The forecast moves from ${formatNumber(first)} to ${formatNumber(last)} over ${n} periods${
          change !== null ? ` (${change >= 0 ? "+" : ""}${change.toFixed(1)}%)` : ""
        }. By the end, 8 in 10 outcomes are expected between ${formatNumber(lo10[n - 1])} and ${formatNumber(hi90[n - 1])}.`;

  return (
    <figure className="grid gap-3">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={reading}
        className="h-auto w-full"
        style={{ fontFamily: "inherit" }}
      >
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="#2b241c" strokeWidth="1" />
            <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" fontSize="13" fill="#c2b392">
              {formatNumber(t)}
            </text>
          </g>
        ))}
        {hist.length > 0 && <line x1={x(off)} x2={x(off)} y1={PAD.t} y2={H - PAD.b} stroke="#8d7e65" strokeDasharray="3 4" />}
        <path d={band} fill="#e8793a" fillOpacity="0.2" />
        {hist.length > 1 && <path d={path(hist, 0)} fill="none" stroke="#c2b392" strokeWidth="1.75" strokeLinejoin="round" />}
        <path d={path([...bridge, ...point.slice(0, n)], bridge.length ? off - 1 : off)} fill="none" stroke="#e8793a" strokeWidth="2.5" strokeLinejoin="round" />
        <text x={x(off)} y={H - 8} textAnchor={hist.length ? "start" : "start"} fontSize="13" fill="#c2b392" dx={hist.length ? 6 : 0}>
          {hist.length ? "Forecast starts" : "Forecast"}
        </text>
      </svg>
      <figcaption className="grid gap-2 text-[0.9375rem] text-sand-1">
        <span className="flex flex-wrap gap-x-5 gap-y-1">
          {hist.length > 0 && (
            <span><span aria-hidden="true" className="mr-1.5 inline-block h-0.5 w-5 bg-sand-1 align-middle" />Your data</span>
          )}
          <span><span aria-hidden="true" className="mr-1.5 inline-block h-0.5 w-5 bg-ember align-middle" />Forecast</span>
          <span><span aria-hidden="true" className="mr-1.5 inline-block h-2.5 w-5 bg-ember/25 align-middle" />Likely range (8 in 10)</span>
        </span>
        <span className="text-sand-0">{reading}</span>
      </figcaption>
    </figure>
  );
}

const KIND: Record<string, { label: string; Icon: typeof File }> = {
  png: { label: "Chart", Icon: FileImage },
  svg: { label: "Chart (vector)", Icon: FileImage },
  xlsx: { label: "Workbook", Icon: FileSpreadsheet },
  csv: { label: "Data", Icon: FileSpreadsheet },
  docx: { label: "Report", Icon: FileText },
  pdf: { label: "PDF report", Icon: FileText },
};

function FileRow({ runId, file }: { runId: string; file: ArtifactFile }) {
  const name = file.path.replace(/^out\//, "");
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  const kind = KIND[ext] ?? { label: "File", Icon: File };
  return (
    <li>
      <a
        href={artifactUrl(runId, file.path)}
        download={name.split("/").pop()}
        title={`Download ${name}`}
        className="group flex min-h-14 items-center gap-3 rounded-lg border border-sand-2/25 bg-night-0/40 px-3 py-2 hover:border-sand-1 hover:bg-night-2"
      >
        <kind.Icon aria-hidden="true" className="size-6 shrink-0 text-sand-1" />
        <span className="grid min-w-0 flex-1">
          <span className="truncate text-base text-sand-0">{name}</span>
          <span className="text-[0.9375rem] text-sand-2">
            {kind.label} · {ext.toUpperCase()} · {formatBytes(file.bytes)}
          </span>
        </span>
        <Download aria-label="Download" className="size-5 shrink-0 text-sand-1 group-hover:text-ember" />
      </a>
    </li>
  );
}

// A chart the analysis drew is shown in the answer itself, on its own white page, with the full-size image one click away.
function ChartFigure({ runId, file }: { runId: string; file: ArtifactFile }) {
  const name = file.path.replace(/^out\//, "");
  const url = artifactUrl(runId, file.path);
  return (
    <figure className="m-0 grid gap-2">
      <a href={url} target="_blank" rel="noopener noreferrer" title="Open the chart full size" className="block overflow-hidden rounded-xl border border-sand-2/25 bg-white">
        <img src={url} alt={`Chart ${name}`} loading="lazy" className="block h-auto w-full" />
      </a>
      <figcaption className="text-[0.9375rem] text-sand-2">{name}</figcaption>
    </figure>
  );
}

export function Results({ runs }: { runs: ArtifactRun[] }) {
  if (runs.length === 0) return null;
  return (
    <div className="grid gap-4">
      {runs.map((run) => (
        <Card key={run.run_id} className="grid gap-5 p-4 sm:p-5">
          {run.forecast?.point && run.forecast.q10 && run.forecast.q90 && (
            <ForecastChart history={run.history ?? []} fc={run.forecast} />
          )}
          {!run.forecast &&
            run.files
              .filter((f) => f.path.toLowerCase().endsWith(".png"))
              .map((f) => <ChartFigure key={f.path} runId={run.run_id} file={f} />)}
          <div className="grid gap-2">
            <h3 className="text-lg font-bold text-sand-0">Files</h3>
            <ul className="grid gap-2 sm:grid-cols-2">
              {run.files.map((f) => (
                <FileRow key={f.path} runId={run.run_id} file={f} />
              ))}
            </ul>
          </div>
        </Card>
      ))}
    </div>
  );
}
