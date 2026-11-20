// Formatting helpers for the console. Australian English, 24-hour clock, local time zone.

export function formatTime(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleString("en-AU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
}

// "just now", "5 min ago", "3 h ago", then a short date. Recognition beats a raw timestamp in a list.
export function relativeTime(unixSeconds: number, now = Date.now()): string {
  const diff = Math.max(0, now / 1000 - unixSeconds);
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} h ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString("en-AU", { day: "numeric", month: "short" });
}

export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m}:${String(s % 60).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
}

export function shortHash(hex: string, keep = 12): string {
  return hex.length > keep ? `${hex.slice(0, keep)}…` : hex;
}

export function truncate(text: string, max = 280): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export function errorMessage(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

// "values_per_series" -> "Values per series". Used only when a tool has no title of its own; code names never reach a screen.
export function humanise(name: string): string {
  const s = name.replace(/[_-]+/g, " ").trim();
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : name;
}

export function formatNumber(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1000) return Math.round(v).toLocaleString("en-AU");
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}
