import { ArrowDownUp, ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type PanelEntityPage, type PanelSummary } from "../lib/api";
import { formatNumber } from "../lib/format";
import { cn } from "../lib/utils";
import { Button, Card } from "./ui";

const MODEL_NAMES: Record<string, string> = {
  naive: "Last value",
  seasonal_naive: "Same period last cycle",
  global_gbm: "Shared model",
  timesfm: "TimesFM 3",
  blend: "Blend",
};
const PAGE_SIZE = 25;

function modelName(m: string | null | undefined): string {
  return m ? (MODEL_NAMES[m] ?? m) : "None";
}

// Per-entity results of a panel backtest: which model did best for each entity, and its MASE for every model (lower is
// better; 1.0 equals the naive rule). Paged and sortable, so a panel of hundreds of entities stays readable.
function PanelRun({ missionId, run }: { missionId: string; run: PanelSummary }) {
  const [page, setPage] = useState(1);
  const [sort, setSort] = useState("entity");
  const [order, setOrder] = useState<"asc" | "desc">("asc");
  const [data, setData] = useState<PanelEntityPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .panelEntities(missionId, run.step, { page, pageSize: PAGE_SIZE, sort, order })
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't load these entities. Try again.");
      });
    return () => {
      cancelled = true;
    };
  }, [missionId, run.step, page, sort, order]);

  function sortBy(next: string) {
    setPage(1);
    if (sort === next) setOrder(order === "asc" ? "desc" : "asc");
    else {
      setSort(next);
      setOrder("asc");
    }
  }

  const models = run.models;
  const ariaFor = (key: string) => (sort === key ? (order === "asc" ? "ascending" : "descending") : "none");

  return (
    <section aria-label={`Results for ${run.entities} entities`} className="grid gap-3">
      <div className="grid gap-1">
        <p className="text-base font-bold text-sand-0">
          {run.entities} {run.entities === 1 ? "entity" : "entities"} backtested
          {run.horizon ? `, ${run.horizon} steps ahead` : ""}
          {run.origins ? `, ${run.origins} test windows` : ""}
        </p>
        <p className="text-caption text-sand-1">
          Best overall: {modelName(run.best_overall)}. Measured on the saved history, not a forecast of the future.
        </p>
        {run.reading && <p className="text-caption text-sand-2">{run.reading}</p>}
      </div>

      {error && (
        <p role="alert" className="text-caption text-bad">
          {error}
        </p>
      )}

      <div className="-mx-1 overflow-x-auto rounded-xl border border-sand-2/25">
        <table className="w-full min-w-[32rem] border-collapse text-caption tabular-nums">
          <caption className="sr-only">Error (MASE, a scaled error) by entity and model: lower is better, 1.0 is the naive rule</caption>
          <thead className="bg-night-2 text-left text-sand-1">
            <tr>
              <th scope="col" aria-sort={ariaFor("entity") as "ascending" | "descending" | "none"} className="p-0">
                <button type="button" onClick={() => sortBy("entity")} className="flex min-h-10 w-full items-center gap-1 px-3 text-left hover:text-sand-0">
                  Entity <ArrowDownUp aria-hidden="true" className="size-3.5 text-sand-2" />
                </button>
              </th>
              <th scope="col" aria-sort={ariaFor("best") as "ascending" | "descending" | "none"} className="p-0">
                <button type="button" onClick={() => sortBy("best")} className="flex min-h-10 w-full items-center gap-1 px-3 text-left hover:text-sand-0">
                  Best model <ArrowDownUp aria-hidden="true" className="size-3.5 text-sand-2" />
                </button>
              </th>
              {models.map((m) => (
                <th key={m} scope="col" aria-sort={ariaFor(`mase:${m}`) as "ascending" | "descending" | "none"} className="p-0 text-right">
                  <button
                    type="button"
                    onClick={() => sortBy(`mase:${m}`)}
                    className="ml-auto flex min-h-10 w-full items-center justify-end gap-1 px-3 hover:text-sand-0"
                  >
                    {modelName(m)} <span className="sr-only">MASE</span>
                    <ArrowDownUp aria-hidden="true" className="size-3.5 text-sand-2" />
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data?.rows.map((row) => (
              <tr key={row.entity} className="border-t border-sand-2/15">
                <th scope="row" className="px-3 py-2 text-left font-normal text-sand-0 [overflow-wrap:anywhere]">
                  {row.entity}
                </th>
                <td className="px-3 py-2 text-sand-1">{modelName(row.best_model)}</td>
                {models.map((m) => {
                  const v = row.mase[m];
                  const isBest = row.best_model === m;
                  return (
                    <td key={m} className={cn("px-3 py-2 text-right", isBest ? "text-ok" : "text-sand-1")}>
                      {typeof v === "number" ? formatNumber(v) : "n/a"}
                      {isBest && <span className="sr-only"> (best)</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && data.pages > 1 && (
        <nav aria-label="Entity pages" className="flex items-center justify-between gap-2 text-caption text-sand-1">
          <Button variant="secondary" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)} aria-label="Previous page">
            <ChevronLeft aria-hidden="true" className="size-4" />
          </Button>
          <span className="tabular-nums" aria-live="polite">
            Page {data.page} of {data.pages} · {data.total} entities
          </span>
          <Button variant="secondary" size="sm" disabled={page >= data.pages} onClick={() => setPage(page + 1)} aria-label="Next page">
            <ChevronRight aria-hidden="true" className="size-4" />
          </Button>
        </nav>
      )}
    </section>
  );
}

// Lists every panel backtest a mission ran. Shown once the work has stopped, so the table does not move under a reader.
export function PanelTable({ missionId, working }: { missionId: string; working: boolean }) {
  const [runs, setRuns] = useState<PanelSummary[]>([]);

  useEffect(() => {
    if (working) return;
    let cancelled = false;
    api
      .panels(missionId)
      .then((r) => {
        if (!cancelled) setRuns(r);
      })
      .catch(() => {
        if (!cancelled) setRuns([]);
      });
    return () => {
      cancelled = true;
    };
  }, [missionId, working]);

  if (runs.length === 0) return null;
  return (
    <Card className="grid gap-5 p-4 sm:p-5">
      {runs.map((run) => (
        <PanelRun key={run.step} missionId={missionId} run={run} />
      ))}
    </Card>
  );
}
