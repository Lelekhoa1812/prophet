import { useCallback, useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

// A side panel whose width the person drags. The width is remembered per panel in this browser (a convenience only: when
// storage is blocked the panel starts at its default width). side says which edge the handle sits on: a left panel's handle
// is on its right edge, so dragging right widens it; a right panel's handle is on its left edge, so dragging left widens it.
export function useResizable(storageKey: string, { initial, min, max, side }: { initial: number; min: number; max: number; side: "left" | "right" }) {
  const clamp = useCallback((w: number) => Math.round(Math.min(max, Math.max(min, w))), [min, max]);
  const [width, setWidth] = useState<number>(() => {
    try {
      const saved = Number(window.localStorage.getItem(storageKey));
      if (Number.isFinite(saved) && saved > 0) return clamp(saved);
    } catch {
      /* storage blocked: default width */
    }
    return initial;
  });
  const [dragging, setDragging] = useState(false);
  const start = useRef<{ x: number; w: number } | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, String(width));
    } catch {
      /* storage blocked: the width lasts for this page only */
    }
  }, [storageKey, width]);

  const onPointerDown = useCallback(
    (e: PointerEvent<HTMLElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      start.current = { x: e.clientX, w: width };
      setDragging(true);
    },
    [width],
  );
  const onPointerMove = useCallback(
    (e: PointerEvent<HTMLElement>) => {
      if (!start.current) return;
      const dx = e.clientX - start.current.x;
      setWidth(clamp(start.current.w + (side === "left" ? dx : -dx)));
    },
    [clamp, side],
  );
  const onPointerUp = useCallback((e: PointerEvent<HTMLElement>) => {
    if (!start.current) return;
    start.current = null;
    setDragging(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  }, []);
  // Keyboard: arrows move 16 px (64 with Shift), Home and End go to the limits. Arrows follow the screen direction.
  const onKeyDown = useCallback(
    (e: KeyboardEvent<HTMLElement>) => {
      const step = e.shiftKey ? 64 : 16;
      const grow = side === "left" ? 1 : -1;
      if (e.key === "ArrowRight") setWidth((w) => clamp(w + step * grow));
      else if (e.key === "ArrowLeft") setWidth((w) => clamp(w - step * grow));
      else if (e.key === "Home") setWidth(min);
      else if (e.key === "End") setWidth(max);
      else return;
      e.preventDefault();
    },
    [clamp, side, min, max],
  );
  const reset = useCallback(() => setWidth(initial), [initial]);

  return { width, dragging, min, max, handle: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onKeyDown, onDoubleClick: reset } };
}
