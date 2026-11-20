import { artifactUrl, type ArtifactRun } from "../lib/api";
import { shortHash } from "../lib/format";
import { formatBytes } from "../lib/utils";
import { Card, Label } from "./ui";

const W = 320;
const H = 170;
const PAD = { l: 44, r: 8, t: 10, b: 24 };

// The Operate signature move: the forecast fan. The P10 to P90 band is shaded, the median dashed and the point
// forecast solid, all drawn from the numbers in the run's result. It is not a picture of a file.
export function FanChart({ point, q10, q90, q50 }: { point: number[]; q10: number[]; q90: number[]; q50: number[] | null }) {
  const n = Math.min(point.length, q10.length, q90.length);
  if (n < 2) return <p className="text-sm text-sand-2">Not enough points to draw the forecast.</p>;

  const all = [...point.slice(0, n), ...q10.slice(0, n), ...q90.slice(0, n)];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const span = hi - lo || 1;
  const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - lo) / span) * (H - PAD.t - PAD.b);
  const line = (vals: number[]) =>
    vals
      .slice(0, n)
      .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join(" ");
  const band =
    q90
      .slice(0, n)
      .map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`)
      .join(" ") +
    " " +
    q10
      .slice(0, n)
      .map((_, i, arr) => {
        const j = arr.length - 1 - i;
        return `L${x(j).toFixed(1)},${y(q10[j]).toFixed(1)}`;
      })
      .join(" ") +
    " Z";

  const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(1));

  return (
    <figure className="grid gap-2">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Forecast fan over ${n} periods: point forecast from ${fmt(point[0])} to ${fmt(point[n - 1])}, P10 to P90 band shaded.`}
        className="h-auto w-full"
      >
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke="#7d6d55" strokeWidth="1" />
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} stroke="#7d6d55" strokeWidth="1" />
        <text x={PAD.l - 6} y={PAD.t + 4} textAnchor="end" fontSize="10" fill="#b9a682" className="tabular-nums">
          {fmt(hi)}
        </text>
        <text x={PAD.l - 6} y={H - PAD.b} textAnchor="end" fontSize="10" fill="#b9a682" className="tabular-nums">
          {fmt(lo)}
        </text>
        <text x={PAD.l} y={H - 6} fontSize="10" fill="#b9a682">
          1
        </text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" fontSize="10" fill="#b9a682">
          {n}
        </text>
        <path d={band} fill="#ef7a3a" fillOpacity="0.22" stroke="none" />
        {q50 && <path d={line(q50)} fill="none" stroke="#b9a682" strokeWidth="1" strokeDasharray="3 3" />}
        <path d={line(point)} fill="none" stroke="#ef7a3a" strokeWidth="2" strokeLinejoin="round" />
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-sand-2">
        <span>
          <span aria-hidden="true" className="mr-1.5 inline-block h-0.5 w-4 bg-ember align-middle" />
          Point forecast
        </span>
        <span>
          <span aria-hidden="true" className="mr-1.5 inline-block h-2 w-4 bg-ember/25 align-middle" />
          P10 to P90
        </span>
        {q50 && (
          <span>
            <span aria-hidden="true" className="mr-1.5 inline-block w-4 border-t border-dashed border-sand-1 align-middle" />
            Median
          </span>
        )}
      </figcaption>
    </figure>
  );
}

function ConfigList({ config }: { config: Record<string, unknown> }) {
  const entries = Object.entries(config).slice(0, 12);
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-sand-2">{k}</dt>
          <dd className="text-sand-1 tabular-nums [overflow-wrap:anywhere]">
            {typeof v === "object" ? JSON.stringify(v) : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function Artifacts({ runs }: { runs: ArtifactRun[] }) {
  if (runs.length === 0) {
    return (
      <Card className="p-4">
        <Label>Artefacts</Label>
        <p className="mt-2 text-sm text-sand-2">
          Forecast packs appear here when a run produces them: charts, a workbook, a report and a hash manifest.
        </p>
      </Card>
    );
  }

  return (
    <div className="grid gap-4">
      {runs.map((run) => {
        const fc = run.forecast;
        return (
          <Card key={run.run_id} className="grid gap-4 p-4">
            <div className="grid gap-1">
              <Label>Forecast pack</Label>
              <span className="font-mono text-xs text-sand-2 [overflow-wrap:anywhere]" title={run.run_id}>
                run {shortHash(run.run_id, 8)}
              </span>
              <span className="text-xs text-sand-2">Research use only. Not for commercial decisions.</span>
            </div>

            {fc?.point && fc.q10 && fc.q90 && (
              <FanChart point={fc.point} q10={fc.q10} q90={fc.q90} q50={fc.q50} />
            )}

            {fc?.config && Object.keys(fc.config).length > 0 && <ConfigList config={fc.config} />}

            <div className="grid gap-2">
              <Label>Files</Label>
              <ul className="grid gap-2">
                {run.files.map((f) => {
                  const name = f.path.replace(/^out\//, "");
                  return (
                    <li key={f.path} className="grid gap-1 rounded-md border border-sand-2/30 bg-night-0/60 p-2.5">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <a
                          href={artifactUrl(run.run_id, f.path)}
                          download={name.split("/").pop()}
                          className="break-all text-sm text-sand-0 underline decoration-ember/60 underline-offset-4 hover:decoration-ember"
                        >
                          {name}
                        </a>
                        <span className="text-xs text-sand-2 tabular-nums">{formatBytes(f.bytes)}</span>
                      </div>
                      <span className="font-mono text-[11px] text-sand-2 [overflow-wrap:anywhere]" title={f.sha256}>
                        sha256 {shortHash(f.sha256)}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>

            {run.files.some((f) => f.path.endsWith("forecast.png")) && (
              <img
                src={artifactUrl(run.run_id, run.files.find((f) => f.path.endsWith("forecast.png"))!.path)}
                alt="Forecast chart produced by the sandboxed run, showing the point forecast and P10 to P90 band."
                className="w-full rounded-md border border-sand-2/30 bg-night-0"
              />
            )}
          </Card>
        );
      })}
    </div>
  );
}
