import { useEffect, useState } from "react";
import { api, type Notebook } from "./api";
import { errorMessage } from "./format";
import { isWorking } from "./utils";

// Loads one notebook: its sources, kept tables and turns. Polls while a source is being checked or a turn is running or
// waiting for an answer, and stops when everything is settled. Each turn's own events are followed by useMission.
export function useNotebook(id: string | null) {
  const [notebook, setNotebook] = useState<Notebook | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    setNotebook(null);
    setError(null);
  }, [id]);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      try {
        const nb = await api.notebook(id);
        if (cancelled) return;
        setNotebook(nb);
        setError(null);
        const checking = nb.sources.some((s) => s.status === "profiling");
        const active = nb.turns.some((t) => isWorking(t.status) || t.status === "waiting_user");
        if (checking || active) timer = setTimeout(run, 2000);
      } catch (err) {
        if (cancelled) return;
        setError(errorMessage(err, "Couldn't load this notebook. Retrying."));
        timer = setTimeout(run, 4000);
      }
    };

    void run();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id, nonce]);

  return { notebook, error, refresh: () => setNonce((n) => n + 1) };
}
