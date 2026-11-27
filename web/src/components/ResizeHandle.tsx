import type { KeyboardEvent, PointerEvent } from "react";
import { cn } from "../lib/utils";

// The strip on a panel's edge that the person drags to resize it. A separator in the accessibility tree, so the width can be
// read and changed with the keyboard too. Double-click puts the panel back to its default width.
export function ResizeHandle({
  label,
  side,
  width,
  min,
  max,
  dragging,
  handle,
}: {
  label: string;
  side: "left" | "right";
  width: number;
  min: number;
  max: number;
  dragging: boolean;
  handle: {
    onPointerDown: (e: PointerEvent<HTMLElement>) => void;
    onPointerMove: (e: PointerEvent<HTMLElement>) => void;
    onPointerUp: (e: PointerEvent<HTMLElement>) => void;
    onPointerCancel: (e: PointerEvent<HTMLElement>) => void;
    onKeyDown: (e: KeyboardEvent<HTMLElement>) => void;
    onDoubleClick: () => void;
  };
}) {
  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title="Drag to resize. Double-click to reset."
      {...handle}
      className={cn(
        "group absolute inset-y-0 z-20 w-2 cursor-col-resize touch-none select-none outline-none",
        side === "left" ? "-right-1" : "-left-1",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors duration-150",
          dragging ? "w-0.5 bg-ember" : "bg-transparent group-hover:bg-sand-2/60 group-focus-visible:w-0.5 group-focus-visible:bg-ember",
        )}
      />
    </div>
  );
}
