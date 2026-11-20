import { useEffect, useState } from "react";
import { api, type ArtifactRun, type ToolInfo } from "../lib/api";
import { useMission } from "../lib/useMission";
import { Turn } from "./Turn";
import { Spinner } from "./ui";

// One question in a notebook. It is a mission: the same live view as a one-off forecast, loaded by its id.
export function NotebookTurn({
  missionId,
  tools,
  onRetry,
}: {
  missionId: string;
  tools: Record<string, ToolInfo>;
  onRetry: (goal: string) => void;
}) {
  const { mission, error, refresh } = useMission(missionId);
  const [runs, setRuns] = useState<ArtifactRun[]>([]);
  const eventCount = mission?.events.length ?? 0;

  useEffect(() => {
    let cancelled = false;
    api
      .artifacts(missionId)
      .then((list) => !cancelled && setRuns(list))
      .catch(() => !cancelled && setRuns([]));
    return () => {
      cancelled = true;
    };
  }, [missionId, mission?.status, eventCount]);

  if (!mission) {
    return (
      <p role="status" className="flex items-center gap-2 text-base text-sand-1">
        {error ? <span className="text-bad">{error}</span> : <><Spinner /> Loading this question</>}
      </p>
    );
  }
  return <Turn mission={mission} tools={tools} runs={runs} onAnswered={refresh} onRetry={onRetry} />;
}
