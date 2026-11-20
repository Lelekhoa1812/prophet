import { useEffect, useRef, useState } from "react";
import { api, type Mission } from "./api";
import { errorMessage } from "./format";

// Polls one mission. Only new events are fetched (by sequence number); the rest of the snapshot is replaced each time.
// Polling stops on a finished mission and slows while the mission waits for an answer.
export function useMission(id: string | null) {
  const [mission, setMission] = useState<Mission | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const seqRef = useRef<{ id: string | null; seq: number }>({ id: null, seq: 0 });

  useEffect(() => {
    if (!id) {
      setMission(null);
      setError(null);
      return;
    }
    if (seqRef.current.id !== id) {
      seqRef.current = { id, seq: 0 };
      setMission(null);
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const run = async () => {
      try {
        const snap = await api.mission(id, seqRef.current.seq);
        if (cancelled) return;
        const last = snap.events[snap.events.length - 1];
        if (last) seqRef.current.seq = last.seq;
        setMission((prev) => ({
          ...snap,
          events: prev && prev.id === id ? [...prev.events, ...snap.events] : snap.events,
        }));
        setError(null);
        const live = snap.status === "created" || snap.status === "running";
        const waiting = snap.status === "waiting_user";
        if (live) timer = setTimeout(run, 1500);
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
