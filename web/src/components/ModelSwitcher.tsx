import { Check, ChevronDown } from "lucide-react";
import { useEffect, useId, useRef, useState, type KeyboardEvent } from "react";
import { useDismiss } from "../lib/useDismiss";
import type { SelectableModel } from "../lib/useModels";
import { cn } from "../lib/utils";

type ProviderGroup = { provider: string; providerLabel: string; items: SelectableModel[] };

// Models arrive in provider order, so a provider's models sit together.
function groupByProvider(models: SelectableModel[]): ProviderGroup[] {
  return models.reduce<ProviderGroup[]>((groups, m) => {
    const last = groups[groups.length - 1];
    if (last && last.provider === m.provider) last.items.push(m);
    else groups.push({ provider: m.provider, providerLabel: m.providerLabel, items: [m] });
    return groups;
  }, []);
}

// The model button under the chat input. The menu lists only selectable models, grouped by provider, with a check on the
// current one. On a phone the menu becomes a full-width sheet at the bottom of the screen.
export function ModelSwitcher({
  models,
  selected,
  catalogStatus,
  onSelect,
  onOpenSettings,
}: {
  models: SelectableModel[];
  selected: SelectableModel | null;
  catalogStatus: "loading" | "ready" | "error";
  onSelect: (id: string) => void;
  onOpenSettings: () => void;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const menuId = useId();

  const closeMenu = (restoreFocus: boolean) => {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus();
  };
  useDismiss(open, wrap, (reason) => closeMenu(reason === "escape"));

  // Opening the menu moves focus to the current model, so the keyboard starts where the choice is.
  useEffect(() => {
    if (!open) return;
    const first =
      menu.current?.querySelector<HTMLElement>('[aria-checked="true"]') ??
      menu.current?.querySelector<HTMLElement>('[role="menuitemradio"]');
    first?.focus();
  }, [open]);

  function onMenuKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const items = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]') ?? []);
    if (items.length === 0) return;
    const at = items.indexOf(document.activeElement as HTMLButtonElement);
    let next: number | null = null;
    if (e.key === "ArrowDown") next = at < 0 ? 0 : (at + 1) % items.length;
    else if (e.key === "ArrowUp") next = at <= 0 ? items.length - 1 : at - 1;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = items.length - 1;
    if (next !== null) {
      e.preventDefault();
      items[next]?.focus();
    }
  }

  if (models.length === 0) {
    if (catalogStatus === "loading") {
      return (
        <span role="status" className="inline-flex min-h-10 items-center px-3 text-caption text-sand-2 sm:min-h-8">
          Loading models
        </span>
      );
    }
    if (catalogStatus === "error") {
      return <span className="inline-flex min-h-10 items-center px-3 text-caption text-sand-2 sm:min-h-8">Models unavailable</span>;
    }
    return (
      <button
        type="button"
        onClick={onOpenSettings}
        className="inline-flex min-h-10 min-w-0 flex-wrap items-center gap-x-2 rounded-lg px-3 text-caption font-semibold text-ember transition-colors hover:bg-night-2 hover:text-ember-strong sm:min-h-8"
      >
        No model enabled
        <span className="font-normal text-sand-1 underline underline-offset-4">Open Settings</span>
      </button>
    );
  }

  const groups = groupByProvider(models);

  return (
    <div ref={wrap} className="relative min-w-0">
      <button
        ref={trigger}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        title="Choose a model"
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "inline-flex min-h-10 max-w-44 min-w-0 items-center gap-1 rounded-lg px-3 text-caption font-semibold transition-colors hover:bg-night-2 hover:text-sand-0 sm:min-h-8",
          open ? "bg-night-2 text-sand-0" : "text-sand-1",
        )}
      >
        <span className="truncate">{selected?.label}</span>
        <ChevronDown aria-hidden="true" className={cn("size-4 shrink-0 transition-transform motion-reduce:transition-none", open && "rotate-180")} />
      </button>

      {open && (
        <div
          ref={menu}
          id={menuId}
          role="menu"
          aria-label="Choose a model"
          onKeyDown={onMenuKeyDown}
          className="absolute right-0 bottom-full z-30 mb-2 grid max-h-80 w-64 gap-0.5 overflow-y-auto rounded-xl border border-sand-2/30 bg-night-2 p-1 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)] max-sm:fixed max-sm:inset-x-0 max-sm:bottom-0 max-sm:mb-0 max-sm:w-auto max-sm:max-h-[70dvh] max-sm:rounded-b-none max-sm:pb-[max(0.25rem,env(safe-area-inset-bottom))]"
        >
          {groups.map((g, gi) => {
            const headingId = `${menuId}-group-${gi}`;
            return (
              <div key={headingId} role="group" aria-labelledby={headingId} className="grid gap-0.5">
                <p id={headingId} className="px-3 pt-2 pb-1 text-micro font-semibold text-sand-2">
                  {g.providerLabel}
                </p>
                {g.items.map((m) => {
                  const isSelected = selected?.id === m.id;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      role="menuitemradio"
                      aria-checked={isSelected}
                      onClick={() => {
                        onSelect(m.id);
                        closeMenu(true);
                      }}
                      className="flex min-h-11 w-full items-center justify-between gap-3 rounded-lg px-3 text-left text-base text-sand-0 transition-colors hover:bg-night-3 focus-visible:bg-night-3 sm:min-h-10"
                    >
                      <span className="truncate">{m.label}</span>
                      {isSelected && <Check aria-hidden="true" className="size-4 shrink-0 text-ember" />}
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
