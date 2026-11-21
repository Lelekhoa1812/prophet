import type { SelectableModel } from "../lib/useModels";
import { ModelSwitcher } from "./ModelSwitcher";
import { UsageMeter } from "./UsageMeter";

// The controls under every message box: the model switcher and the usage meter. A quick forecast and a notebook both
// render this, so the model choice and its budget reading look and behave the same wherever a question is asked.
export function ComposerControls({
  models,
  catalogStatus,
  selected,
  onSelect,
  onOpenSettings,
}: {
  models: SelectableModel[];
  catalogStatus: "loading" | "ready" | "error";
  selected: SelectableModel | null;
  onSelect: (id: string) => void;
  onOpenSettings: () => void;
}) {
  return (
    <>
      <ModelSwitcher models={models} selected={selected} catalogStatus={catalogStatus} onSelect={onSelect} onOpenSettings={onOpenSettings} />
      <UsageMeter modelId={selected?.id ?? null} modelLabel={selected?.label ?? null} onManageBudgets={onOpenSettings} />
    </>
  );
}
