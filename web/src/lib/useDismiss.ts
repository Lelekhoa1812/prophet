import { useEffect, type RefObject } from "react";

// While `open`, calls onDismiss on Escape, or on a pointer press outside the element. Keyboard users close with Escape.
export function useDismiss(open: boolean, ref: RefObject<HTMLElement | null>, onDismiss: (reason: "escape" | "outside") => void) {
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: PointerEvent) => {
      if (ref.current && e.target instanceof Node && !ref.current.contains(e.target)) onDismiss("outside");
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss("escape");
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, ref, onDismiss]);
}
