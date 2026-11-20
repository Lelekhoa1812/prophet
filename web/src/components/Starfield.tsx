import { useEffect, useRef } from "react";

// A drawn starfield, written for this project in the spirit of the React Bits background effects. It is decoration
// only: aria-hidden, pointer-events none, and static under prefers-reduced-motion.
export function Starfield() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let w = 0;
    let h = 0;
    let stars: { x: number; y: number; r: number; a: number; s: number }[] = [];
    let px = 0;
    let py = 0;
    let raf = 0;

    const seed = () => {
      const count = Math.round(Math.min(160, Math.max(60, (w * h) / 9000)));
      stars = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.1 + 0.2,
        a: Math.random() * 0.6 + 0.25,
        s: Math.random() * 0.12 + 0.02,
      }));
    };

    const resize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      seed();
      draw(0);
    };

    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      const parX = w > 1024 ? px * 14 : 0;
      const parY = w > 1024 ? py * 14 : 0;
      for (const s of stars) {
        const twinkle = reduce ? 0 : Math.sin(t * s.s * 0.002 + s.x) * 0.25;
        ctx.globalAlpha = Math.min(1, Math.max(0.1, s.a + twinkle));
        ctx.fillStyle = "#f1e4c8";
        ctx.beginPath();
        ctx.arc(s.x - parX * s.r, s.y - parY * s.r, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };

    const loop = (t: number) => {
      draw(t);
      raf = requestAnimationFrame(loop);
    };

    const onMove = (e: PointerEvent) => {
      px = e.clientX / w - 0.5;
      py = e.clientY / h - 0.5;
      if (reduce) draw(0);
    };

    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", onMove);
    if (!reduce) raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      window.removeEventListener("pointermove", onMove);
    };
  }, []);

  return <canvas ref={ref} aria-hidden="true" className="pointer-events-none fixed inset-0 -z-10 opacity-70" />;
}
