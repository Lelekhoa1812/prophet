import { useCallback, useEffect, useState } from "react";
import { api, type UsageView } from "./api";

// Usage moves slowly, so one request a minute is plenty. Polling stops while the page is hidden.
const POLL_MS = 60_000;

export type UsageStatus = "loading" | "ready" | "error";

// Keeps the last good reading on screen if a later refresh fails; `status` says whether that reading is current.
export function useUsage(enabled: boolean) {
  const [data, setData] = useState<UsageView | null>(null);
  const [status, setStatus] = useState<UsageStatus>("loading");

  const refresh = useCallback(() => {
    api
      .usage()
      .then((u) => {
        setData({ models: Array.isArray(u?.models) ? u.models : [] });
        setStatus("ready");
      })
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    let timer: number | undefined;
    const stop = () => {
      if (timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
      }
    };
    const start = () => {
      if (timer === undefined) timer = window.setInterval(refresh, POLL_MS);
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        refresh();
        start();
      } else {
        stop();
      }
    };

    refresh();
    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [enabled, refresh]);

  return { data, status, refresh };
}
