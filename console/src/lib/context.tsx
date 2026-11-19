import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useApi, type Meta } from "@/lib/api";

export type Lens = "executive" | "operator" | "developer" | "demo";
interface Ctx {
  meta: Meta | null; metaError: string | null; dataset: string; setDataset: (d: string) => void;
  lens: Lens; setLens: (l: Lens) => void; dark: boolean; toggleDark: () => void; reveal: boolean; setReveal: (r: boolean) => void;
  revealed: Record<string, string>; remember: (codes: string[]) => void;
}
const C = createContext<Ctx>(null as unknown as Ctx);
export const useApp = () => useContext(C);

const store = {
  get: (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

export function AppProvider({ children }: { children: ReactNode }) {
  const { data: meta, error: metaError } = useApi<Meta>("/meta");
  const [dataset, setDatasetState] = useState(store.get("ds", "twin"));
  const [lens, setLensState] = useState<Lens>(store.get("lens", "executive") as Lens);
  const [dark, setDark] = useState(store.get("dark", "0") === "1");
  const [reveal, setRevealState] = useState(store.get("reveal", "0") === "1");
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  useEffect(() => { document.documentElement.classList.toggle("dark", dark); }, [dark]);
  const value = useMemo<Ctx>(() => ({
    meta, metaError, dataset, lens, dark, reveal, revealed,
    setDataset: (d) => { setDatasetState(d); store.set("ds", d); },
    setLens: (l) => { setLensState(l); store.set("lens", l); },
    toggleDark: () => setDark((v) => { store.set("dark", v ? "0" : "1"); return !v; }),
    setReveal: (r) => { setRevealState(r); store.set("reveal", r ? "1" : "0"); },
    remember: (codes) => {
      const need = codes.filter((c) => !(c in revealed));
      if (!need.length) return;
      fetch(`/api/reveal?codes=${need.join(",")}`).then((r) => r.json()).then((m) => setRevealed((p) => ({ ...p, ...m }))).catch(() => undefined);
    },
  }), [meta, metaError, dataset, lens, dark, reveal, revealed]);
  return <C.Provider value={value}>{children}</C.Provider>;
}
