"use client";

import { useEffect, useRef, useState } from "react";
import rough from "roughjs";

type Props = { children: React.ReactNode; className?: string; seed?: number; fill?: string; as?: "div" | "section" | "aside" };

type PathInfo = { d: string; stroke: string; strokeWidth: number; fill?: string };

const generator = rough.generator();

// A box outlined like an Excalidraw sketch; the seed keeps each box's stroke stable between renders.
export function RoughBox({ children, className = "", seed = 1, fill, as: Tag = "div" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize((s) => (s && Math.abs(s.w - width) < 1 && Math.abs(s.h - height) < 1 ? s : { w: width, h: height }));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  let paths: PathInfo[] = [];
  if (size && size.w > 4 && size.h > 4) {
    const shape = generator.rectangle(2, 2, size.w - 4, size.h - 4, {
      seed,
      roughness: 1.1,
      bowing: 0.8,
      stroke: "#1c1c1a",
      strokeWidth: 1.3,
      fill,
      fillStyle: "solid",
    });
    paths = generator.toPaths(shape) as PathInfo[];
  }

  return (
    <Tag className={`relative ${className}`}>
      <div ref={ref} className="pointer-events-none absolute inset-0" aria-hidden>
        {size && (
          <svg width={size.w} height={size.h} className="absolute inset-0 overflow-visible">
            {paths.map((p, i) => (
              <path key={i} d={p.d} stroke={p.stroke} strokeWidth={p.strokeWidth} fill={p.fill ?? "none"} strokeLinecap="round" />
            ))}
          </svg>
        )}
      </div>
      <div className="relative h-full">{children}</div>
    </Tag>
  );
}

// Hand-drawn underline used under the brand and the active tab.
export function Scribble({ className = "", width = 120 }: { className?: string; width?: number }) {
  return (
    <svg viewBox="0 0 120 8" width={width} height={(width / 120) * 8} className={className} aria-hidden>
      <path d="M2 5.5 C 25 2, 45 7, 66 4 S 104 2.5, 118 4.8" fill="none" stroke="#1c1c1a" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
