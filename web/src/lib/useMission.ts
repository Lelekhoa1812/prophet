import { useEffect, useRef, useState } from "react";
import { api, type Mission } from "./api";
import { errorMessage } from "./format";

// What the console has already shown of one mission: the last event seq and the last step index. step stays null until
// the first full load, so a poll with a null step asks for the whole snapshot.
type Cursor = { id: string | null; seq: number; step: number };

// A poll after the first load: only the events and steps newer than the cursor come back. The API client has no step
// cursor, so this request names both values itself.
async function fetchNewer(id: string, seq: number, step: number): Promise<Mission> {
  const res = await fetch(`/api/missions/${encodeURIComponent(id)}?after=${seq}&after_step=${step}`, { cache: "no-store" });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* not JSON: keep the status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as Mission;
}

// Rows already shown stay; a row that comes back again replaces its copy, so a repeat is harmless. Rows stay in key order.
function mergeRows<T>(shown: T[], fresh: T[], key: (row: T) => number): T[] {
  const byKey = new Map<number, T>();
  for (const row of [...shown, ...fresh]) byKey.set(key(row), row);
  return [...byKey.values()].sort((a, b) => key(a) - key(b));
}

// Polls one mission. After the first load each poll fetches only the events and steps newer than the cursor and merges
// them in; the rest of the snapshot (status, progress, outcome, questions) is replaced each time. Polling stops on a
// finished mission and slows while the mission waits for an answer.
export function useMission(id: string | null) {
  const [mission, setMission] = useState<Mission | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  // The cursor is only set after a load that filled the mission state, and cleared whenever that state is cleared.
  const cursorRef = useRef<Cursor>({ id: null, seq: 0, step: -1 });
  const loadedRef = useRef(false);

  useEffect(() => {
    if (!id) {
      cursorRef.current = { id: null, seq: 0, step: -1 };
      loadedRef.current = false;
      setMission(null);
      setError(null);
      return;
    }
    if (cursorRef.current.id !== id) {
      cursorRef.current = { id, seq: 0, step: -1 };
      loadedRef.current = false;
      setMission(null);
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      try {
        const cursor = cursorRef.current;
        const delta = loadedRef.current;
        const snap = delta ? await fetchNewer(id, cursor.seq, cursor.step) : await api.mission(id, 0);
        if (cancelled) return;
        const lastEvent = snap.events[snap.events.length - 1];
        const lastStep = snap.steps[snap.steps.length - 1];
        cursorRef.current = {
          id,
          seq: lastEvent ? lastEvent.seq : cursor.seq,
          step: lastStep ? lastStep.idx : cursor.step,
        };
        loadedRef.current = true;
        setMission((prev) => {
          if (!delta || !prev || prev.id !== id) return snap;
          return {
            ...snap,
            events: mergeRows(prev.events, snap.events, (e) => e.seq),
            steps: mergeRows(prev.steps, snap.steps, (s) => s.idx),
          };
        });
        setError(null);
        const live = snap.status === "created" || snap.status === "running";
        const waiting = snap.status === "waiting_user";
        // Short while the mission runs, so answer text appears as the model writes it.
        if (live) timer = setTimeout(run, 300);
        else if (waiting) timer = setTimeout(run, 4000);
      } catch (err) {
        if (cancelled) return;
        setError(errorMessage(err, "Lost contact with the API; retrying"));
        timer = setTimeout(run, 4000);
      }
    };

    void run();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id, nonce]);

  return { mission, error, refresh: () => setNonce((n) => n + 1) };
}
