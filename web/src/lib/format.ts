// Formatting helpers for the console. Australian English, 24-hour clock, local time zone.

export function formatTime(unixSeconds: number): string {
  const d = new Date(unixSeconds * 1000);
  return d.toLocaleString("en-AU", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
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
