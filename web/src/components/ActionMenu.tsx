import { MoreHorizontal } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { cn } from "../lib/utils";

export type MenuItem = {
  key: string;
  label: string;
  icon?: ReactNode;
  tone?: "neutral" | "danger";
  onSelect: () => void;
};

// A small popover menu with a trigger. The panel is positioned with fixed coordinates, so a sidebar that scrolls
// never clips it. Escape, a click outside, or choosing an item closes it and returns focus to the trigger.
export function ActionMenu({
  label,
  items,
  trigger,
  className,
  triggerClassName,
  panelClassName,
  onOpenChange,
}: {
  label: string;
  items: MenuItem[];
  trigger?: ReactNode;
  className?: string;
  triggerClassName?: string;
  panelClassName?: string;
  onOpenChange?: (open: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const uid = useId();

  const setOpenState = (next: boolean) => {
    setOpen(next);
    if (!next) setPos(null);
    onOpenChange?.(next);
  };

  // Place the panel under the trigger, or above it when there is no room below.
  useLayoutEffect(() => {
    if (!open || !button.current || !panel.current) return;
    const r = button.current.getBoundingClientRect();
    const h = panel.current.offsetHeight;
    const w = panel.current.offsetWidth;
    const below = r.bottom + 4;
    const top = below + h > window.innerHeight - 8 ? Math.max(8, r.top - h - 4) : below;
    const left = Math.min(Math.max(8, r.right - w), window.innerWidth - w - 8);
    setPos({ top, left });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!panel.current?.contains(t) && !button.current?.contains(t)) setOpenState(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpenState(false);
        button.current?.focus();
      }
    };
    const close = () => setOpenState(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Focus the first item once the panel is placed, so keyboard users start inside the menu.
  useEffect(() => {
    // preventScroll: focusing an item must not scroll the sidebar, which would close the menu.
    if (open && pos) panel.current?.querySelector<HTMLButtonElement>('[role="menuitem"]')?.focus({ preventScroll: true });
  }, [open, pos]);

  function onPanelKey(e: KeyboardEvent<HTMLDivElement>) {
    const nodes = Array.from(panel.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]') ?? []);
    const at = nodes.indexOf(document.activeElement as HTMLButtonElement);
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = e.key === "ArrowDown" ? (at + 1) % nodes.length : (at - 1 + nodes.length) % nodes.length;
      nodes[next]?.focus();
    } else if (e.key === "Tab") {
      setOpenState(false);
    }
  }

  return (
    <div className={cn("relative inline-flex shrink-0", className)}>
      <button
        ref={button}
        type="button"
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? `${uid}-menu` : undefined}
        title={label}
        onClick={(e) => {
          e.stopPropagation();
          setOpenState(!open);
        }}
        className={cn(
          "grid size-10 place-items-center rounded-md text-sand-2 transition-colors hover:bg-night-3 hover:text-sand-0 focus-visible:text-sand-0 sm:size-8",
          triggerClassName,
        )}
      >
        {trigger ?? <MoreHorizontal aria-hidden="true" className="size-4" />}
      </button>
      {open && (
        <div
          ref={panel}
          id={`${uid}-menu`}
          role="menu"
          aria-label={label}
          onKeyDown={onPanelKey}
          style={{ position: "fixed", top: pos?.top ?? -9999, left: pos?.left ?? -9999, visibility: pos ? "visible" : "hidden" }}
          className={cn(
            "z-50 grid min-w-44 gap-0.5 rounded-xl border border-sand-2/30 bg-night-2 p-1 shadow-[0_18px_40px_-12px_rgba(0,0,0,0.9)]",
            panelClassName,
          )}
        >
          {items.map((item) => (
            <button
              key={item.key}
              type="button"
              role="menuitem"
              onClick={() => {
                setOpenState(false);
                item.onSelect();
              }}
              className={cn(
                "flex min-h-10 w-full items-center gap-3 rounded-lg px-3 text-left text-base transition-colors hover:bg-night-3 focus-visible:bg-night-3 focus-visible:outline-none sm:min-h-9 sm:text-body",
                item.tone === "danger" ? "text-bad" : "text-sand-0",
              )}
            >
              {item.icon && <span aria-hidden="true" className="grid size-4 place-items-center">{item.icon}</span>}
              {item.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
