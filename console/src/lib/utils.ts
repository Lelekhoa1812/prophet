import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const money = (n: number | null | undefined, compact = false) => {
  if (n == null || Number.isNaN(n)) return "–";
  const abs = Math.abs(n);
  if (compact && abs >= 1_000_000) return `${n < 0 ? "-" : ""}$${(abs / 1_000_000).toFixed(abs >= 1e7 ? 0 : 1)}M`;
  if (compact && abs >= 10_000) return `${n < 0 ? "-" : ""}$${(abs / 1000).toFixed(0)}k`;
  return `${n < 0 ? "-" : ""}$${abs.toLocaleString("en-AU", { maximumFractionDigits: abs < 100 ? 2 : 0 })}`;
};
export const pct = (n: number | null | undefined, digits = 0) => (n == null || Number.isNaN(n) ? "–" : `${(n * 100).toFixed(digits)}%`);
export const num = (n: number | null | undefined, digits = 0) => (n == null || Number.isNaN(n) ? "–" : n.toLocaleString("en-AU", { maximumFractionDigits: digits }));
export const dateLabel = (iso: string) => new Date(iso + "T00:00:00").toLocaleDateString("en-AU", { day: "numeric", month: "short", year: "numeric" });
export const dayShort = (iso: string) => new Date(iso + "T00:00:00").toLocaleDateString("en-AU", { day: "numeric", month: "short" });
