import { Download, File, FileImage, FileSpreadsheet, FileText } from "lucide-react";
import { artifactUrl, type ArtifactFile, type ArtifactRun } from "../lib/api";
import { formatNumber, shortHash } from "../lib/format";
import { formatBytes } from "../lib/utils";
import { Card } from "./ui";

const W = 640;
const H = 260;
// Room for axis labels and the caption line at a readable size: on a 375 px phone the chart scales to about 0.59,
// so a 20-unit label is about 12 px.
const PAD = { l: 84, r: 16, t: 14, b: 40 };
const LABEL = 20;

type Fc = NonNullable<ArtifactRun["forecast"]>;

// Where the shaded band came from, read from the forecast's config (A7-13). A band is called calibrated only when the
// config says so, with its held-out coverage when that is given. Otherwise it is the raw model band, which has not been
// checked against past forecasts, so the chart must not say it covers a fixed share of outcomes.
export function bandSource(config: Record<string, unknown> | null): string {
  if (config?.band_source !== "calibrated") return "raw TimesFM band, not yet checked against past forecasts";
  const c = config.band_coverage;
  if (typeof c !== "number" || !Number.isFinite(c)) return "calibrated band";
  const pct = c <= 1 ? c * 100 : c;
  return `calibrated band (held-out coverage ${pct.toFixed(1)}%)`;
}

// The forecast, drawn from the numbers in the run: the series so far in sand, the forecast in ember, and the shaded band
// between the 10th and 90th percentile of the model's outcomes. The figcaption says in words what the band is and where it
// came from, so the drawing itself is hidden from screen readers.
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
  // The "Forecast starts" label sits to the left of its line when there is no room to its right.
  const flip = hist.length > 0 && x(off) + 160 > W - PAD.r;
  const first = point[0];
  const last = point[n - 1];
  const change = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : null;
  const source = bandSource(fc.config);
  const reading =
    n === 1
      ? `The forecast is ${formatNumber(first)}. The likely range is ${formatNumber(lo10[0])} to ${formatNumber(hi90[0])}.`
      : `The forecast moves from ${formatNumber(first)} to ${formatNumber(last)} over ${n} periods${
          change !== null ? ` (${change >= 0 ? "+" : ""}${change.toFixed(1)}%)` : ""
        }. By the end, the likely range is ${formatNumber(lo10[n - 1])} to ${formatNumber(hi90[n - 1])}.`;

  return (
    <figure className="grid gap-3">
      <svg viewBox={`0 0 ${W} ${H}`} aria-hidden="true" className="h-auto w-full" style={{ fontFamily: "inherit" }}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="stroke-night-3" strokeWidth="1" />
            <text x={PAD.l - 8} y={y(t) + 6} textAnchor="end" fontSize={LABEL} className="fill-sand-1">
              {formatNumber(t)}
            </text>
          </g>
        ))}
        {hist.length > 0 && <line x1={x(off)} x2={x(off)} y1={PAD.t} y2={H - PAD.b} className="stroke-sand-2" strokeDasharray="3 4" />}
        <path d={band} className="fill-ember/20" />
        {hist.length > 1 && <path d={path(hist, 0)} className="fill-none stroke-sand-1" strokeWidth="1.75" strokeLinejoin="round" />}
        <path d={path([...bridge, ...point.slice(0, n)], bridge.length ? off - 1 : off)} className="fill-none stroke-ember" strokeWidth="2.5" strokeLinejoin="round" />
        <text
          x={flip ? x(off) - 6 : x(off) + (hist.length ? 6 : 0)}
          y={H - 12}
          textAnchor={flip ? "end" : "start"}
          fontSize={LABEL}
          className="fill-sand-1"
        >
          {hist.length ? "Forecast starts" : "Forecast"}
        </text>
      </svg>
      <figcaption className="grid gap-2 text-caption text-sand-1">
        <span className="flex flex-wrap gap-x-5 gap-y-1">
          {hist.length > 0 && (
            <span><span aria-hidden="true" className="mr-1.5 inline-block h-0.5 w-5 bg-sand-1 align-middle" />Your data</span>
          )}
          <span><span aria-hidden="true" className="mr-1.5 inline-block h-0.5 w-5 bg-ember align-middle" />Forecast</span>
          <span><span aria-hidden="true" className="mr-1.5 inline-block h-2.5 w-5 bg-ember/25 align-middle" />Likely range (10th to 90th percentile)</span>
        </span>
        <span className="text-sand-0">{reading}</span>
        <span className="text-sand-2">Band source: {source}.</span>
      </figcaption>
    </figure>
  );
}

const KIND: Record<string, { label: string; Icon: typeof File }> = {
  png: { label: "Chart", Icon: FileImage },
  svg: { label: "Chart (vector)", Icon: FileImage },
  jpg: { label: "Image", Icon: FileImage },
  jpeg: { label: "Image", Icon: FileImage },
  gif: { label: "Image", Icon: FileImage },
  webp: { label: "Image", Icon: FileImage },
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
          <span className="text-caption text-sand-2">
            {kind.label} · {ext.toUpperCase()} · {formatBytes(file.bytes)}
            {file.sha256 && (
              <>
                {" · "}
                <span className="font-mono text-meta" title={`sha256 ${file.sha256}`}>
                  sha256 {shortHash(file.sha256)}
                </span>
              </>
            )}
          </span>
        </span>
        <Download aria-label="Download" className="size-5 shrink-0 text-sand-1 group-hover:text-ember" />
      </a>
    </li>
  );
}

const IMAGE_EXT = new Set(["png", "jpg", "jpeg", "gif", "webp", "svg"]);

export function isImagePath(path: string): boolean {
  const dot = path.lastIndexOf(".");
  if (dot <= 0 || dot === path.length - 1) return false;
  return IMAGE_EXT.has(path.slice(dot + 1).toLowerCase());
}

// One picture per chart. A chart saved as both .png and .svg (same folder, same name) is one image: the PNG is shown and the
// vector file stays in the file list to download. Different names are different charts and all show.
const IMAGE_RANK: Record<string, number> = { png: 0, webp: 1, jpg: 2, jpeg: 2, gif: 3, svg: 4 };
const imageStem = (path: string) => path.replace(/\.[^./]+$/, "").toLowerCase();
const imageRank = (path: string) => IMAGE_RANK[path.slice(path.lastIndexOf(".") + 1).toLowerCase()] ?? 9;

export function imagesToShow(files: ArtifactFile[]): ArtifactFile[] {
  const best = new Map<string, ArtifactFile>();
  for (const f of files) {
    if (!isImagePath(f.path)) continue;
    const key = imageStem(f.path);
    const have = best.get(key);
    if (!have || imageRank(f.path) < imageRank(have.path)) best.set(key, f);
  }
  return files.filter((f) => best.get(imageStem(f.path)) === f);
}

// An image is shown on a white page. A click opens the full-size file.
export function ImagePreview({ src, name }: { src: string; name: string }) {
  return (
    <figure className="m-0 grid gap-2">
      <a href={src} target="_blank" rel="noopener noreferrer" title={`Open ${name}`} className="block overflow-hidden rounded-xl border border-sand-2/50 bg-white shadow-[var(--shadow-composer)]">
        <img src={src} alt={`Image ${name}`} loading="lazy" className="block h-auto w-full" />
      </a>
      <figcaption className="text-caption text-sand-2">{name}</figcaption>
    </figure>
  );
}

function ChartFigure({ runId, file }: { runId: string; file: ArtifactFile }) {
  const name = file.path.replace(/^out\//, "");
  return <ImagePreview src={artifactUrl(runId, file.path)} name={name} />;
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
          {imagesToShow(run.files).map((f) => (
            <ChartFigure key={f.path} runId={run.run_id} file={f} />
          ))}
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
