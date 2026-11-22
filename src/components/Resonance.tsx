import { useEffect, useRef } from "react";
import type { PerformanceStore } from "../lib/live";

export function Resonance({
  store,
  quality,
}: {
  store: PerformanceStore;
  quality: "full" | "reduced" | "off";
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    let raf = 0;
    let last = 0;
    let pulse = 0;
    let phase = 0;
    /** What the canvas shows; null = it must be redrawn. */
    let shown: { pulse: number; color: string; w: number; h: number } | null =
      null;
    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      const w = canvas.clientWidth,
        h = canvas.clientHeight;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      if (
        canvas.width !== Math.round(w * dpr) ||
        canvas.height !== Math.round(h * dpr)
      ) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        shown = null; // Resizing clears the canvas.
      }
      const frame = store.frame;
      const present = !!frame?.note && now - store.receivedAt < 500;
      const energy = present
        ? Math.max(0, Math.min(1, ((frame?.levelDb ?? -80) + 65) / 50))
        : 0;
      const dt = Math.min(0.05, (now - last) / 1000 || 0.016);
      last = now;
      pulse += (energy - pulse) * Math.min(1, dt * 5);
      const motion = quality === "full" && !media.matches;
      if (motion) phase += dt;
      const cx = w / 2,
        cy = h * 0.49,
        radius = Math.min(w * 0.27, h * 0.36);
      const color =
        present && Math.abs(frame?.note?.cents ?? 99) < 10
          ? "214,183,122"
          : "101,171,166";
      // Silent and settled: ripples and string swing (pulse × 6 px) are
      // invisible and the picture is unchanged, so skip the redraw.
      if (
        shown &&
        shown.w === w &&
        shown.h === h &&
        shown.color === color &&
        Math.abs(shown.pulse - pulse) < 1e-3 &&
        (!motion || pulse < 0.005)
      )
        return;
      shown = { pulse, color, w, h };
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 1.9);
      glow.addColorStop(0, `rgba(${color},${0.05 + pulse * 0.12})`);
      glow.addColorStop(1, `rgba(${color},0)`);
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, w, h);
      for (let i = 0; i < 7; i++) {
        ctx.beginPath();
        for (let k = 0; k <= 180; k++) {
          const a = (k / 180) * Math.PI * 2;
          const ripple = motion
            ? Math.sin(a * 6 + phase * (i % 2 ? 0.4 : -0.5) + i) * pulse * 6
            : 0;
          const r = radius * (0.5 + i * 0.115) + ripple;
          const x = cx + Math.cos(a) * r,
            y = cy + Math.sin(a) * r * 0.82;
          if (k === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = `rgba(${color},${0.12 + i * 0.017 + pulse * 0.06})`;
        ctx.lineWidth = i === 3 ? 1.4 : 0.6;
        ctx.stroke();
      }
      // Six courses, with vibration driven by the measured acoustic level.
      for (let i = 0; i < 6; i++) {
        const x = cx + (i - 2.5) * 12;
        ctx.beginPath();
        ctx.moveTo(x, cy - radius * 1.17);
        ctx.bezierCurveTo(
          x + (motion ? Math.sin(phase * 12 + i) * pulse * 6 : 0),
          cy - radius * 0.2,
          x,
          cy + radius * 0.5,
          x,
          cy + radius * 1.17,
        );
        ctx.strokeStyle = `rgba(${color},${0.18 + pulse * 0.18})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.fillStyle = `rgba(${color},0.65)`;
      ctx.beginPath();
      ctx.arc(cx, cy, 3 + pulse * 3, 0, Math.PI * 2);
      ctx.fill();
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [store, quality]);
  return <canvas ref={ref} className="resonance" aria-hidden="true" />;
}
