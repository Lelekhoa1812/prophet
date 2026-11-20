import { Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import type { MissionRow } from "../lib/api";
import { relativeTime, truncate } from "../lib/format";
import { cn, isWorking, statusInfo } from "../lib/utils";
import { Button, StatusBadge } from "./ui";

// Recent forecasts, newest first. Search narrows by the words in the request. Status is icon plus word.
export function Sidebar({
  missions,
  selectedId,
  onSelect,
  onNew,
}: {
  missions: MissionRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const [query, setQuery] = useState("");
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? missions.filter((m) => m.goal.toLowerCase().includes(q)) : missions;
  }, [missions, query]);

  return (
    <nav aria-label="Forecasts" className="flex h-full min-h-0 flex-col gap-4">
      <Button onClick={onNew} className="w-full justify-start" variant="secondary">
        <Plus aria-hidden="true" className="size-4" />
        New forecast
      </Button>

      {missions.length > 4 && (
        <label className="relative block">
          <span className="sr-only">Search your forecasts</span>
          <Search aria-hidden="true" className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-sand-2" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search"
            className="min-h-10 w-full rounded-lg border border-sand-2/30 bg-night-0/50 pr-3 pl-9 text-base text-sand-0 placeholder:text-sand-2 focus:border-ember focus:outline-none"
          />
        </label>
      )}

      <div className="grid min-h-0 flex-1 content-start gap-1 overflow-y-auto">
        <h2 className="px-2 pb-1 text-[0.9375rem] font-semibold text-sand-2">Recents</h2>
        {missions.length === 0 && <p className="px-2 text-[0.9375rem] text-sand-2">Nothing yet. Your forecasts will appear here.</p>}
        {missions.length > 0 && shown.length === 0 && <p className="px-2 text-[0.9375rem] text-sand-2">No forecasts match “{query}”.</p>}
        <ul className="grid gap-1">
          {shown.map((m) => {
            const active = m.id === selectedId;
            const tone = statusInfo(m.status).tone;
            return (
              <li key={m.id}>
                <button
                  type="button"
                  onClick={() => onSelect(m.id)}
                  aria-current={active ? "true" : undefined}
                  className={cn(
                    "grid min-h-11 w-full gap-0.5 rounded-lg px-3 py-2 text-left transition-colors hover:bg-night-2",
                    active && "bg-night-2",
                  )}
                >
                  <span className="line-clamp-2 text-base text-sand-0 [overflow-wrap:anywhere]">{truncate(m.goal, 90)}</span>
                  <span className="flex items-center gap-2 text-[0.9375rem] text-sand-2">
                    <span className="tabular-nums">{relativeTime(m.updated || m.created)}</span>
                    {(tone !== "ok" || isWorking(m.status)) && <StatusBadge status={m.status} />}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
}
