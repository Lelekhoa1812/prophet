import { useCallback, useEffect, useState } from "react";
import { api, type CatalogProvider } from "./api";

export type SelectableModel = { id: string; label: string; provider: string; providerLabel: string };

// A model can be chosen only when its provider is enabled and ready and the model itself is enabled.
export function selectableModels(providers: CatalogProvider[]): SelectableModel[] {
  return providers.flatMap((p) =>
    p.enabled && p.ready
      ? p.models.filter((m) => m.enabled).map((m) => ({ id: m.id, label: m.label, provider: p.name, providerLabel: p.label }))
      : [],
  );
}

export function useModelCatalog() {
  const [state, setState] = useState<{ status: "loading" | "ready" | "error"; providers: CatalogProvider[] }>({
    status: "loading",
    providers: [],
  });

  const reload = useCallback(() => {
    api
      .modelCatalog()
      .then((c) => setState({ status: "ready", providers: c.providers }))
      .catch(() => setState((s) => ({ status: "error", providers: s.providers })));
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return { status: state.status, selectable: selectableModels(state.providers), reload };
}

const STORE_KEY = "prophet.model";

// The choice is a per-browser convenience. Storage can be blocked, so every access is guarded.
function readStoredModel(): string | null {
  try {
    return localStorage.getItem(STORE_KEY);
  } catch {
    return null;
  }
}

function writeStoredModel(id: string): void {
  try {
    localStorage.setItem(STORE_KEY, id);
  } catch {
    /* storage unavailable: the choice lasts for this page only */
  }
}

// The model for the next forecast. If the remembered model is no longer selectable, the first selectable one is used.
export function useSelectedModel(selectable: SelectableModel[]) {
  const [storedId, setStoredId] = useState<string | null>(readStoredModel);
  const model = selectable.find((m) => m.id === storedId) ?? selectable[0] ?? null;

  const choose = useCallback((id: string) => {
    setStoredId(id);
    writeStoredModel(id);
  }, []);

  return { model, choose };
}
